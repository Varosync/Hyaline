#!/usr/bin/env python
"""Train V2-D production model."""
import sys
sys.path.insert(0, '/home/ec2-user/Jinja')

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from pathlib import Path

print('Loading dataset...')
from hyaline.data import load_dataset_with_motifs
from hyaline.model_v2 import HyalineV2, count_parameters

device = 'cuda'
data_list, labels, _, _ = load_dataset_with_motifs()
print(f'Loaded {len(data_list)} samples')

# V2-D config (BEST from optimization)
config = {'hidden_dim': 320, 'num_layers': 5, 'num_rbf': 96, 'dropout': 0.15, 'use_motif_bias': True}

loader = DataLoader(data_list, batch_size=4, shuffle=True)
model = HyalineV2(**config).to(device)
print(f'V2-D params: {count_parameters(model):,}')

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}: Loss={total_loss/len(loader):.4f}')

model.eval()
preds, truth = [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.extend(probs.flatten())
        truth.extend(batch.y.cpu().numpy())
train_auroc = roc_auc_score(truth, preds)
print(f'Training AuROC: {train_auroc:.4f}')

Path('checkpoints').mkdir(exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'train_auroc': train_auroc
}, 'checkpoints/hyaline.pt')

print('Done! Model saved to checkpoints/hyaline.pt')
