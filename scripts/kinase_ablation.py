#!/usr/bin/env python3
"""
Kinase Conformational Selectivity Ablation Study
Compare: RF(seq) vs RF(struct) vs Static EGNN vs Spiking EGNN
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from scipy import stats
import json


# Dataset
class KinaseDataset(Dataset):
    def __init__(self, n_samples=1000, seed=42):
        np.random.seed(seed)
        self.samples = []
        for i in range(n_samples):
            pocket_seq = np.random.randint(0, 20, size=85).astype(np.int64)
            coords_in = np.random.randn(85, 3).astype(np.float32) * 12
            coords_out = coords_in.copy()
            
            dfg_flip = 3.0 + np.random.rand() * 8.0
            coords_out[79:84] += np.array([dfg_flip, dfg_flip*0.5, dfg_flip*0.3])
            chelix_shift = np.random.rand() * 5.0
            coords_out[20:31] += np.array([0, -chelix_shift, chelix_shift*0.5])
            
            drug_fp = (np.random.rand(2048) > 0.9).astype(np.float32)
            drug_size = drug_fp[:256].sum() / 256.0
            drug_flex = drug_fp[256:512].sum() / 256.0
            
            struct_sel = dfg_flip / 10.0
            delta_pki = -(drug_size-0.1)*struct_sel*12 + (drug_flex-0.1)*(1-struct_sel)*8 + np.random.randn()*0.2
            
            self.samples.append({
                'pocket_seq': pocket_seq, 'coords_in': coords_in, 'coords_out': coords_out,
                'drug_fp': drug_fp, 'delta_pki': float(delta_pki), 'dfg_flip': float(dfg_flip)
            })
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v, dtype=torch.float32) for k, v in s.items()}


# Feature extraction
def extract_struct_features(samples):
    X, y = [], []
    for s in samples:
        diff = s['coords_out'] - s['coords_in']
        dfg_mag = np.sqrt((diff[79:84]**2).sum(axis=-1)).mean()
        chelix_mag = np.sqrt((diff[20:31]**2).sum(axis=-1)).mean()
        drug_size = s['drug_fp'][:256].mean()
        drug_flex = s['drug_fp'][256:512].mean()
        X.append([dfg_mag, chelix_mag, drug_size, drug_flex, dfg_mag*drug_size, chelix_mag*drug_flex])
        y.append(s['delta_pki'])
    return np.array(X), np.array(y)

def extract_seq_features(samples):
    X, y = [], []
    for s in samples:
        seq_oh = np.zeros((85, 22)); seq_oh[np.arange(85), s['pocket_seq']] = 1
        X.append(np.concatenate([seq_oh.flatten(), s['drug_fp']]))
        y.append(s['delta_pki'])
    return np.array(X), np.array(y)


# Static EGNN
class StaticEGNN(nn.Module):
    def __init__(self, hdim=128, nlayers=4):
        super().__init__()
        self.enc = nn.Linear(25, hdim)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hdim*2+9, hdim), nn.SiLU(), nn.Linear(hdim, hdim)) for _ in range(nlayers)])
        self.ln = nn.LayerNorm(hdim)
    
    def forward(self, x, pos, ei, ea):
        h = F.silu(self.enc(torch.cat([x, pos], -1)))
        for layer in self.layers:
            row, col = ei
            dist = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)
            m = layer(torch.cat([h[row], h[col], ea, dist], -1))
            agg = torch.zeros_like(h); agg.index_add_(0, row, m)
            h = self.ln(h + agg)
        return h


# Spiking EGNN
class SpikingEGNN(nn.Module):
    def __init__(self, hdim=128, nlayers=4, beta=0.9):
        super().__init__()
        self.enc = nn.Linear(25, hdim)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hdim*2+9, hdim), nn.SiLU(), nn.Linear(hdim, hdim)) for _ in range(nlayers)])
        self.ln = nn.LayerNorm(hdim)
        self.beta = beta
    
    def forward(self, x, pos, ei, ea):
        h = F.silu(self.enc(torch.cat([x, pos], -1)))
        membrane = torch.zeros_like(h)
        spikes_all = []
        for layer in self.layers:
            row, col = ei
            dist = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)
            m = layer(torch.cat([h[row], h[col], ea, dist], -1))
            agg = torch.zeros_like(h); agg.index_add_(0, row, m)
            h = self.ln(h + agg)
            # Spiking dynamics
            membrane = self.beta * membrane + h
            spikes = (membrane.abs().mean(-1) > 1.0).float()
            membrane = membrane * (1 - spikes.unsqueeze(-1))
            spikes_all.append(spikes)
        return h, torch.stack(spikes_all)


# Binding predictor
class BindingPredictor(nn.Module):
    def __init__(self, egnn, hdim=128, spiking=False):
        super().__init__()
        self.egnn = egnn; self.spiking = spiking
        self.drug_enc = nn.Sequential(nn.Linear(2048, hdim), nn.GELU(), nn.Linear(hdim, hdim))
        self.pred = nn.Sequential(nn.Linear(hdim*2+3, hdim), nn.GELU(), nn.Linear(hdim, 1))
    
    def forward(self, seq, c_in, c_out, drug, ei, ea):
        B = seq.size(0)
        def enc(seq, c):
            x = F.one_hot(seq.clamp(0,21).long(), 22).float().view(B*85, 22)
            p = c.view(B*85, 3)
            if self.spiking:
                h, _ = self.egnn(x, p, ei, ea)
            else:
                h = self.egnn(x, p, ei, ea)
            return h.view(B, 85, -1).mean(1)
        e_in, e_out = enc(seq, c_in), enc(seq, c_out)
        d = self.drug_enc(drug)
        diff = c_out - c_in
        sf = torch.stack([diff[:,79:84].norm(dim=-1).mean(-1), diff[:,20:31].norm(dim=-1).mean(-1), (diff**2).mean((1,2)).sqrt()], -1)
        return self.pred(torch.cat([e_in-e_out, d, sf], -1)).squeeze(-1)


def build_edges(bs, n=85, k=6):
    edges = []
    for b in range(bs):
        off = b * n
        for i in range(n):
            for j in np.random.choice(n, k, replace=False):
                edges.append([off+i, off+j])
    return torch.tensor(edges).T


def train_nn(model, train_dl, val_dl, device, epochs=30):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for batch in train_dl:
            seq, c_in, c_out, drug, tgt = [batch[k].to(device) for k in ['pocket_seq','coords_in','coords_out','drug_fp','delta_pki']]
            ei = build_edges(seq.size(0)).to(device)
            ea = torch.ones(ei.size(1), 8, device=device)
            opt.zero_grad()
            F.mse_loss(model(seq, c_in, c_out, drug, ei, ea), tgt).backward()
            opt.step()
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for batch in val_dl:
            seq, c_in, c_out, drug, tgt = [batch[k].to(device) for k in ['pocket_seq','coords_in','coords_out','drug_fp','delta_pki']]
            ei = build_edges(seq.size(0)).to(device)
            ea = torch.ones(ei.size(1), 8, device=device)
            preds.extend(model(seq, c_in, c_out, drug, ei, ea).cpu().numpy())
            tgts.extend(tgt.cpu().numpy())
    return r2_score(tgts, preds), stats.pearsonr(tgts, preds)[0]


def main():
    print("="*70)
    print("KINASE ABLATION STUDY: Static vs Spiking EGNN")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = KinaseDataset(n_samples=1500, seed=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {k: [] for k in ['rf_seq', 'rf_struct', 'static_egnn', 'spiking_egnn']}
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\nFold {fold+1}/5")
        tr_samp = [dataset.samples[i] for i in tr_idx]
        va_samp = [dataset.samples[i] for i in va_idx]
        tr_dl = DataLoader(torch.utils.data.Subset(dataset, tr_idx), batch_size=32, shuffle=True)
        va_dl = DataLoader(torch.utils.data.Subset(dataset, va_idx), batch_size=32)
        
        # RF seq
        X_tr, y_tr = extract_seq_features(tr_samp)
        X_va, y_va = extract_seq_features(va_samp)
        rf = RandomForestRegressor(100, max_depth=10, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
        results['rf_seq'].append(r2_score(y_va, rf.predict(X_va)))
        
        # RF struct
        X_tr, _ = extract_struct_features(tr_samp)
        X_va, _ = extract_struct_features(va_samp)
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
        results['rf_struct'].append(r2_score(y_va, gb.predict(X_va)))
        
        # Static EGNN
        static = BindingPredictor(StaticEGNN(64, 3), 64, False)
        r2, r = train_nn(static, tr_dl, va_dl, device, 25)
        results['static_egnn'].append(r2)
        
        # Spiking EGNN
        spiking = BindingPredictor(SpikingEGNN(64, 3), 64, True)
        r2, r = train_nn(spiking, tr_dl, va_dl, device, 25)
        results['spiking_egnn'].append(r2)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for name, vals in results.items():
        print(f"{name:<20}: R² = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # Key comparison
    spiking_r2 = results['spiking_egnn']
    static_r2 = results['static_egnn']
    diff = np.mean(spiking_r2) - np.mean(static_r2)
    _, p = stats.wilcoxon(spiking_r2, static_r2)
    
    print(f"\nSpiking vs Static: Δ = {diff:.4f}, p = {p:.4f}")
    if diff > 0 and p < 0.1:
        print("✓ Spiking dynamics show improvement!")
    
    Path('checkpoints').mkdir(exist_ok=True)
    with open('checkpoints/kinase_ablation.json', 'w') as f:
        json.dump(results, f)
    print("\n✓ Saved to checkpoints/kinase_ablation.json")


if __name__ == "__main__":
    main()
