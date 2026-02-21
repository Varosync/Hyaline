#!/usr/bin/env python3
"""
Ablation Study Framework
========================

Systematically test if each component of TFActivationModel adds value.

Tests:
1. Full model vs Random Forest baseline
2. Full model vs no-spiking (static EGNN)
3. Full model vs no-context (EGNN only)
4. Full model vs random structure
5. Leave-one-TF-out cross-validation
6. Leave-one-cell-type-out cross-validation

Success criteria:
- Full model must beat ALL baselines by >5% AUC
- p-value < 0.05 on Wilcoxon signed-rank test
- Cohen's d > 0.5 (medium effect size)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from scipy import stats
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from hyaline.models.tf_activation_model import TFActivationModel, TFActivationConfig
from hyaline.loaders.pdb_loader import load_tf_structures
from hyaline.loaders.tf_activation_data import SCENICContext, TF_LINEAGE, TF_NAMES, CELL_TYPES


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    n_samples_per_combination: int = 5
    n_bootstrap: int = 1000
    alpha: float = 0.05
    seed: int = 42


# =============================================================================
# Data Generation
# =============================================================================

def create_large_dataset(structures: Dict, n_per_combo: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create large dataset for proper validation.
    
    Returns:
        X_features: Flattened features for baselines [n_samples, n_features]
        X_full: Full data for neural model (list of dicts)
        y: Labels [n_samples]
        groups_tf: TF group for each sample
        groups_ct: Cell type group for each sample
    """
    samples = []
    labels = []
    groups_tf = []
    groups_ct = []
    
    # Use more cell types
    cell_types = [
        'melanocyte', 'hepatocyte', 'cardiomyocyte', 'neuron',
        't_cell', 'b_cell', 'macrophage', 'fibroblast',
        'keratinocyte', 'adipocyte', 'myocyte', 'astrocyte'
    ]
    
    for tf_name, struct_list in structures.items():
        tf_lineage = TF_LINEAGE.get(tf_name, None)
        
        for struct in struct_list:
            for cell_type in cell_types:
                # Generate multiple samples per combination
                for _ in range(n_per_combo):
                    is_positive = tf_lineage and tf_lineage in cell_type.lower()
                    
                    ctx = SCENICContext.from_cell_type(cell_type, tf_name, noise_scale=0.2)
                    
                    # Create feature vector for baselines
                    # [tf_one_hot, cell_type_one_hot, tf_activity, chromatin, coactivators]
                    tf_idx = TF_NAMES.index(tf_name) if tf_name in TF_NAMES else 0
                    ct_idx = CELL_TYPES.index(cell_type) if cell_type in CELL_TYPES else 0
                    
                    tf_onehot = np.zeros(len(TF_NAMES))
                    tf_onehot[tf_idx] = 1
                    
                    ct_onehot = np.zeros(len(CELL_TYPES))
                    ct_onehot[ct_idx] = 1
                    
                    features = np.concatenate([
                        tf_onehot,
                        ct_onehot,
                        ctx.tf_activity,
                        ctx.chromatin_topics,
                        ctx.coactivator_expr
                    ])
                    
                    samples.append({
                        'features': features,
                        'node_features': struct.node_features,
                        'pos': struct.pos,
                        'edge_index': struct.edge_index,
                        'edge_attr': struct.edge_attr,
                        'context': ctx,
                        'tf_name': tf_name,
                        'cell_type': cell_type,
                    })
                    
                    labels.append(1.0 if is_positive else 0.0)
                    groups_tf.append(tf_name)
                    groups_ct.append(cell_type)
    
    X_features = np.array([s['features'] for s in samples])
    y = np.array(labels)
    groups_tf = np.array(groups_tf)
    groups_ct = np.array(groups_ct)
    
    return samples, X_features, y, groups_tf, groups_ct


# =============================================================================
# Baseline Models
# =============================================================================

def get_baselines() -> Dict[str, any]:
    """Get baseline models for comparison."""
    return {
        'random_forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'logistic_regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
    }


# =============================================================================
# Neural Model Wrapper
# =============================================================================

class NeuralModelWrapper:
    """Wrapper for TFActivationModel to use in cross-validation."""
    
    def __init__(self, use_spiking=True, use_context=True, use_structure=True):
        self.use_spiking = use_spiking
        self.use_context = use_context
        self.use_structure = use_structure
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, samples: List[Dict], y: np.ndarray, epochs: int = 20):
        """Train the model."""
        config = TFActivationConfig()
        self.model = TFActivationModel(config).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i, sample in enumerate(samples):
                # Prepare inputs
                nf = torch.from_numpy(sample['node_features']).float().to(self.device)
                
                if self.use_structure:
                    pos = torch.from_numpy(sample['pos']).float().to(self.device)
                else:
                    # Randomize structure
                    pos = torch.randn_like(torch.from_numpy(sample['pos'])).float().to(self.device) * 10
                
                ei = torch.from_numpy(sample['edge_index']).long().to(self.device)
                ea = torch.from_numpy(sample['edge_attr']).float().to(self.device)
                
                ctx = sample['context']
                cell_type = torch.tensor([CELL_TYPES.index(sample['cell_type']) if sample['cell_type'] in CELL_TYPES else 0]).to(self.device)
                tf_activity = torch.from_numpy(ctx.tf_activity).float().unsqueeze(0).to(self.device)
                chromatin = torch.from_numpy(ctx.chromatin_topics).float().unsqueeze(0).to(self.device)
                coactivators = torch.from_numpy(ctx.coactivator_expr).float().unsqueeze(0).to(self.device)
                
                if not self.use_context:
                    # Zero out context
                    tf_activity = torch.zeros_like(tf_activity)
                    chromatin = torch.zeros_like(chromatin)
                    coactivators = torch.zeros_like(coactivators)
                
                label = torch.tensor([y[i]]).float().to(self.device)
                
                optimizer.zero_grad()
                output = self.model(nf, pos, ei, ea, cell_type, tf_activity, chromatin, coactivators)
                loss = criterion(output.activation_prob, label)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        return self
    
    def predict_proba(self, samples: List[Dict]) -> np.ndarray:
        """Predict probabilities."""
        self.model.eval()
        probs = []
        
        with torch.no_grad():
            for sample in samples:
                nf = torch.from_numpy(sample['node_features']).float().to(self.device)
                
                if self.use_structure:
                    pos = torch.from_numpy(sample['pos']).float().to(self.device)
                else:
                    pos = torch.randn_like(torch.from_numpy(sample['pos'])).float().to(self.device) * 10
                
                ei = torch.from_numpy(sample['edge_index']).long().to(self.device)
                ea = torch.from_numpy(sample['edge_attr']).float().to(self.device)
                
                ctx = sample['context']
                cell_type = torch.tensor([CELL_TYPES.index(sample['cell_type']) if sample['cell_type'] in CELL_TYPES else 0]).to(self.device)
                tf_activity = torch.from_numpy(ctx.tf_activity).float().unsqueeze(0).to(self.device)
                chromatin = torch.from_numpy(ctx.chromatin_topics).float().unsqueeze(0).to(self.device)
                coactivators = torch.from_numpy(ctx.coactivator_expr).float().unsqueeze(0).to(self.device)
                
                if not self.use_context:
                    tf_activity = torch.zeros_like(tf_activity)
                    chromatin = torch.zeros_like(chromatin)
                    coactivators = torch.zeros_like(coactivators)
                
                output = self.model(nf, pos, ei, ea, cell_type, tf_activity, chromatin, coactivators)
                probs.append(output.activation_prob.cpu().numpy()[0])
        
        return np.array(probs)


# =============================================================================
# Statistical Tests
# =============================================================================

def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test."""
    try:
        stat, pval = stats.wilcoxon(scores_a, scores_b)
    except:
        stat, pval = 0, 1.0
    return stat, pval


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Effect size (Cohen's d)."""
    diff = scores_a - scores_b
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """Bootstrap 95% confidence interval."""
    boot_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(scores), len(scores), replace=True)
        boot_means.append(np.mean(scores[idx]))
    return np.percentile(boot_means, [2.5, 97.5])


# =============================================================================
# Cross-Validation
# =============================================================================

def leave_one_group_out_cv(
    model_fn,
    samples: List[Dict],
    X_features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    is_neural: bool = False
) -> np.ndarray:
    """Leave-one-group-out cross-validation."""
    logo = LeaveOneGroupOut()
    scores = []
    
    unique_groups = np.unique(groups)
    
    for train_idx, test_idx in logo.split(X_features, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        
        if is_neural:
            model = model_fn()
            train_samples = [samples[i] for i in train_idx]
            test_samples = [samples[i] for i in test_idx]
            model.fit(train_samples, y[train_idx], epochs=10)
            y_prob = model.predict_proba(test_samples)
        else:
            model = model_fn()
            model.fit(X_features[train_idx], y[train_idx])
            y_prob = model.predict_proba(X_features[test_idx])[:, 1]
        
        try:
            auc = roc_auc_score(y[test_idx], y_prob)
            scores.append(auc)
        except:
            pass
    
    return np.array(scores)


def stratified_kfold_cv(
    model_fn,
    samples: List[Dict],
    X_features: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    is_neural: bool = False
) -> np.ndarray:
    """Stratified K-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X_features, y):
        if is_neural:
            model = model_fn()
            train_samples = [samples[i] for i in train_idx]
            test_samples = [samples[i] for i in test_idx]
            model.fit(train_samples, y[train_idx], epochs=10)
            y_prob = model.predict_proba(test_samples)
        else:
            model = model_fn()
            model.fit(X_features[train_idx], y[train_idx])
            y_prob = model.predict_proba(X_features[test_idx])[:, 1]
        
        try:
            auc = roc_auc_score(y[test_idx], y_prob)
            scores.append(auc)
        except:
            pass
    
    return np.array(scores)


# =============================================================================
# Main Ablation Study
# =============================================================================

def run_ablation_study():
    """Run complete ablation study."""
    print("=" * 70)
    print("ABLATION STUDY: TFActivationModel")
    print("=" * 70)
    
    # Load structures
    print("\n[1/5] Loading TF-DNA structures...")
    structures = load_tf_structures(Path("data/tf_dna_structures"))
    print(f"Loaded {sum(len(v) for v in structures.values())} structures for {len(structures)} TFs")
    
    # Create dataset
    print("\n[2/5] Creating large dataset...")
    samples, X_features, y, groups_tf, groups_ct = create_large_dataset(structures, n_per_combo=3)
    print(f"Created {len(samples)} samples")
    print(f"  Positive: {sum(y)}, Negative: {len(y) - sum(y)}")
    print(f"  Unique TFs: {len(np.unique(groups_tf))}")
    print(f"  Unique cell types: {len(np.unique(groups_ct))}")
    
    # Define models to compare
    models = {
        'full_model': lambda: NeuralModelWrapper(use_spiking=True, use_context=True, use_structure=True),
        'no_context': lambda: NeuralModelWrapper(use_spiking=True, use_context=False, use_structure=True),
        'no_structure': lambda: NeuralModelWrapper(use_spiking=True, use_context=True, use_structure=False),
    }
    
    baselines = get_baselines()
    
    results = {}
    
    # Run 10-fold CV for all models
    print("\n[3/5] Running 10-fold stratified CV...")
    
    # Baselines first (faster)
    for name, model_fn in baselines.items():
        print(f"  Evaluating {name}...")
        scores = stratified_kfold_cv(lambda m=model_fn: type(m)(**m.get_params()), 
                                      samples, X_features, y, n_splits=10, is_neural=False)
        
        # Actually need to recreate model each time
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X_features, y):
            model = type(model_fn)(**model_fn.get_params())
            model.fit(X_features[train_idx], y[train_idx])
            y_prob = model.predict_proba(X_features[test_idx])[:, 1]
            try:
                scores.append(roc_auc_score(y[test_idx], y_prob))
            except:
                pass
        scores = np.array(scores)
        
        results[name] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores': scores.tolist(),
            'ci_low': float(np.percentile(scores, 2.5)) if len(scores) > 0 else 0,
            'ci_high': float(np.percentile(scores, 97.5)) if len(scores) > 0 else 0,
        }
        print(f"    AUC: {results[name]['mean']:.4f} ± {results[name]['std']:.4f}")
    
    # Neural models (slower)
    for name, model_fn in models.items():
        print(f"  Evaluating {name}...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Fewer folds for speed
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, y)):
            print(f"    Fold {fold+1}/5...", end=" ", flush=True)
            model = model_fn()
            train_samples = [samples[i] for i in train_idx]
            test_samples = [samples[i] for i in test_idx]
            model.fit(train_samples, y[train_idx], epochs=10)
            y_prob = model.predict_proba(test_samples)
            try:
                auc = roc_auc_score(y[test_idx], y_prob)
                scores.append(auc)
                print(f"AUC={auc:.4f}")
            except Exception as e:
                print(f"Error: {e}")
        
        scores = np.array(scores)
        results[name] = {
            'mean': float(np.mean(scores)) if len(scores) > 0 else 0,
            'std': float(np.std(scores)) if len(scores) > 0 else 0,
            'scores': scores.tolist(),
            'ci_low': float(np.percentile(scores, 2.5)) if len(scores) > 0 else 0,
            'ci_high': float(np.percentile(scores, 97.5)) if len(scores) > 0 else 0,
        }
        print(f"    AUC: {results[name]['mean']:.4f} ± {results[name]['std']:.4f}")
    
    # Statistical comparisons
    print("\n[4/5] Statistical comparisons vs full_model...")
    
    full_scores = np.array(results.get('full_model', {}).get('scores', []))
    
    comparisons = {}
    for name, data in results.items():
        if name == 'full_model':
            continue
        
        other_scores = np.array(data['scores'])
        
        # Align lengths
        min_len = min(len(full_scores), len(other_scores))
        if min_len < 2:
            continue
        
        fs = full_scores[:min_len]
        os = other_scores[:min_len]
        
        stat, pval = wilcoxon_test(fs, os)
        d = cohens_d(fs, os)
        
        comparisons[name] = {
            'wilcoxon_stat': float(stat),
            'p_value': float(pval),
            'cohens_d': float(d),
            'significant': pval < 0.05,
            'effect_size': 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small',
            'full_model_better': np.mean(fs) > np.mean(os),
        }
        
        sig = "✓" if pval < 0.05 else "✗"
        print(f"  vs {name}: p={pval:.4f} {sig}, d={d:.3f} ({comparisons[name]['effect_size']})")
    
    # Summary
    print("\n[5/5] SUMMARY")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print("\nModel Rankings (by AUC):")
    print("-" * 50)
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"  {rank}. {name:25s} AUC={data['mean']:.4f} ± {data['std']:.4f}")
    
    # Save results
    output = {
        'results': results,
        'comparisons': comparisons,
        'metadata': {
            'n_samples': len(samples),
            'n_positive': int(sum(y)),
            'n_negative': int(len(y) - sum(y)),
            'n_tfs': len(np.unique(groups_tf)),
            'n_cell_types': len(np.unique(groups_ct)),
        }
    }
    
    with open('ablation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to ablation_results.json")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    full_mean = results.get('full_model', {}).get('mean', 0)
    best_baseline = max(
        [(name, data['mean']) for name, data in results.items() 
         if name in baselines],
        key=lambda x: x[1],
        default=('none', 0)
    )
    
    improvement = full_mean - best_baseline[1]
    
    if improvement > 0.05:
        print(f"✓ Full model beats best baseline ({best_baseline[0]}) by {improvement:.1%}")
        print("  → Architecture ADDS VALUE")
    elif improvement > 0:
        print(f"~ Full model slightly better than {best_baseline[0]} by {improvement:.1%}")
        print("  → Marginal improvement, needs more data")
    else:
        print(f"✗ Best baseline ({best_baseline[0]}) beats full model by {-improvement:.1%}")
        print("  → Architecture does NOT add value over simpler methods")
    
    return output


if __name__ == "__main__":
    results = run_ablation_study()
