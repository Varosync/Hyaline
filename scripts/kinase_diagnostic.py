#!/usr/bin/env python3
"""
Diagnostic: Why RF beats GNN by 40 points
Fix: Hybrid model with RF features + GNN
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats
import json


# Reuse data generation from ablation
class KinaseDataset:
    def __init__(self, n=1500, seed=42):
        np.random.seed(seed)
        self.samples = []
        for _ in range(n):
            pocket_seq = np.random.randint(0, 20, 85).astype(np.int64)
            coords_in = np.random.randn(85, 3).astype(np.float32) * 12
            coords_out = coords_in.copy()
            dfg_flip = 3.0 + np.random.rand() * 8.0
            coords_out[79:84] += [dfg_flip, dfg_flip*0.5, dfg_flip*0.3]
            chelix_shift = np.random.rand() * 5.0
            coords_out[20:31] += [0, -chelix_shift, chelix_shift*0.5]
            drug_fp = (np.random.rand(2048) > 0.9).astype(np.float32)
            drug_size = drug_fp[:256].sum() / 256.0
            drug_flex = drug_fp[256:512].sum() / 256.0
            struct_sel = dfg_flip / 10.0
            delta_pki = -(drug_size-0.1)*struct_sel*12 + (drug_flex-0.1)*(1-struct_sel)*8 + np.random.randn()*0.2
            self.samples.append({
                'pocket_seq': pocket_seq, 'coords_in': coords_in, 'coords_out': coords_out,
                'drug_fp': drug_fp, 'delta_pki': float(delta_pki),
                'dfg_flip': dfg_flip, 'chelix_shift': chelix_shift,
                'drug_size': drug_size, 'drug_flex': drug_flex
            })


def extract_rf_features(samples):
    """Hand-crafted features that encode domain knowledge."""
    X, y = [], []
    for s in samples:
        diff = s['coords_out'] - s['coords_in']
        dfg_mag = np.sqrt((diff[79:84]**2).sum(axis=-1)).mean()
        chelix_mag = np.sqrt((diff[20:31]**2).sum(axis=-1)).mean()
        rmsd = np.sqrt((diff**2).mean())
        drug_size = s['drug_fp'][:256].mean()
        drug_flex = s['drug_fp'][256:512].mean()
        # These interaction terms encode the DOMAIN KNOWLEDGE
        X.append([dfg_mag, chelix_mag, rmsd, drug_size, drug_flex, 
                  dfg_mag*drug_size, chelix_mag*drug_flex])
        y.append(s['delta_pki'])
    return np.array(X), np.array(y)


# ============= HYBRID MODEL =============
class HybridModel(nn.Module):
    """
    Hybrid: RF features + simple neural network
    Gives NN the "hints" from domain knowledge
    """
    def __init__(self, n_rf_features=7, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rf_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Linear(hidden//2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_hybrid(X_train, y_train, X_val, y_val, epochs=100):
    """Train hybrid model."""
    # Normalize features
    X_mean, X_std = X_train.mean(0), X_train.std(0) + 1e-6
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-6
    
    X_tr_norm = (X_train - X_mean) / X_std
    X_va_norm = (X_val - X_mean) / X_std
    y_tr_norm = (y_train - y_mean) / y_std
    
    X_tr_t = torch.FloatTensor(X_tr_norm)
    y_tr_t = torch.FloatTensor(y_tr_norm)
    X_va_t = torch.FloatTensor(X_va_norm)
    
    model = HybridModel(X_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_r2 = -float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = F.mse_loss(pred, y_tr_t)
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            pred_val = model(X_va_t).numpy() * y_std + y_mean
            r2 = r2_score(y_val, pred_val)
        
        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Final prediction
    model.eval()
    with torch.no_grad():
        pred_final = model(X_va_t).numpy() * y_std + y_mean
    
    return r2_score(y_val, pred_final), stats.pearsonr(y_val, pred_final)[0]


def main():
    print("="*70)
    print("DIAGNOSTIC: Why RF beats GNN")
    print("="*70)
    
    # Create dataset
    dataset = KinaseDataset(n=2000, seed=42)
    
    # Extract RF features
    X, y = extract_rf_features(dataset.samples)
    
    print("\n[1] RF Feature Importance Analysis")
    print("-"*50)
    
    # Train RF to get feature importances
    rf = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    feature_names = ['DFG_mag', 'CHelix_mag', 'RMSD', 'Drug_size', 'Drug_flex', 
                     'DFG*Size', 'CHelix*Flex']
    
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\nFeature Importances (what RF learned):")
    for i in sorted_idx:
        bar = "█" * int(importances[i] * 50)
        print(f"  {feature_names[i]:<15}: {importances[i]:.3f} {bar}")
    
    print("\n[2] Cross-validation Comparison")
    print("-"*50)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_r2s, hybrid_r2s = [], []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        # RF
        rf = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_tr, y_tr)
        rf_r2 = r2_score(y_va, rf.predict(X_va))
        rf_r2s.append(rf_r2)
        
        # Hybrid (NN on RF features)
        hybrid_r2, _ = train_hybrid(X_tr, y_tr, X_va, y_va)
        hybrid_r2s.append(hybrid_r2)
        
        print(f"  Fold {fold+1}: RF={rf_r2:.4f}, Hybrid={hybrid_r2:.4f}")
    
    print("\n[3] RESULTS")
    print("="*70)
    print(f"\n{'Model':<25} {'R² (mean±std)':>20}")
    print("-"*50)
    print(f"{'RF (GradientBoosting)':<25} {np.mean(rf_r2s):.4f} ± {np.std(rf_r2s):.4f}")
    print(f"{'Hybrid (NN on RF feats)':<25} {np.mean(hybrid_r2s):.4f} ± {np.std(hybrid_r2s):.4f}")
    
    print("\n[4] KEY INSIGHT")
    print("="*70)
    print("""
The RF works because its features encode DOMAIN KNOWLEDGE:
- DFG_magnitude: How much the kinase flips (structural)
- DFG*Drug_size: INTERACTION between structure and drug
- CHelix*Drug_flex: INTERACTION between conformation and drug

The GNN failed because:
1. It has ~100K parameters learning from 1,500 samples
2. It must discover these interactions from scratch
3. Without enough data, it learns NOTHING useful

SOLUTION: Give the NN the RF features as hints.
The Hybrid model uses RF's domain knowledge while
learning non-linear patterns the RF might miss.
""")
    
    # Save results
    Path('checkpoints').mkdir(exist_ok=True)
    results = {
        'rf_r2_mean': float(np.mean(rf_r2s)),
        'rf_r2_std': float(np.std(rf_r2s)),
        'hybrid_r2_mean': float(np.mean(hybrid_r2s)),
        'hybrid_r2_std': float(np.std(hybrid_r2s)),
        'feature_importances': {n: float(v) for n, v in zip(feature_names, importances)},
    }
    with open('checkpoints/diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Saved to checkpoints/diagnostic_results.json")


if __name__ == "__main__":
    main()
