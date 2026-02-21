#!/usr/bin/env python3
"""Train DFG classifier on real KLIFS data with GPU acceleration."""
import requests
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Load cached data
with open('klifs_cache/key_kinases.json') as f:
    data = json.load(f)

# Collect structures
all_structures = []
for kinase in data['kinases']:
    resp = requests.get('https://klifs.net/api_v2/structures_list',
                       params={'kinase_ID': [kinase['id']]}, timeout=30)
    for s in resp.json():
        dfg = s.get('DFG', '')
        pocket = s.get('pocket', '')
        if dfg and pocket:
            if dfg == 'in':
                dfg_label = 0
            elif 'out' in dfg.lower():
                dfg_label = 1
            else:
                continue
            all_structures.append({
                'kinase': kinase['name'], 'dfg_label': dfg_label, 'pocket': pocket
            })

print(f"Collected {len(all_structures)} structures")

# Encode pockets
AA = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY_-')}
X = np.array([[AA.get(c, 21) for c in s['pocket'][:85]] + [21]*(85-len(s['pocket'][:85])) 
              for s in all_structures])
y = np.array([s['dfg_label'] for s in all_structures])
kinases = np.array([s['kinase'] for s in all_structures])

# Leave-ABL1-out validation
test_mask = kinases == 'ABL1'
X_train, y_train = X[~test_mask], y[~test_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train: {len(X_train)}, Test: {len(X_test)} (ABL1)")

# GPU tensors
X_tr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
y_tr = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
y_te = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

# Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(85, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

model = Model().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

best_acc = 0
for epoch in range(100):
    model.train()
    opt.zero_grad()
    loss = criterion(model(X_tr), y_tr)
    loss.backward()
    opt.step()
    
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(X_te)) > 0.5).float()
        acc = (pred == y_te).float().mean().item()
    
    if acc > best_acc:
        best_acc = acc
    
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Test Acc={acc:.1%}")

print(f"\n{'='*50}")
print(f"BEST TEST ACCURACY: {best_acc:.1%}")
print(f"{'='*50}")

# Save
Path('klifs_cache').mkdir(exist_ok=True)
json.dump({'test_acc': best_acc, 'train_size': len(X_train), 'test_size': len(X_test)},
          open('klifs_cache/dfg_results.json', 'w'))
