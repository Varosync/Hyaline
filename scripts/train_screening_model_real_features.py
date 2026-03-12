#!/usr/bin/env python3
"""
Train Type II Screening Model with REAL Pocket Features
========================================================

Uses real geometric features extracted from PDB structures.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyaline.screening.screening_model import Type2ScreeningModel


class KLIFSDataset(Dataset):
    """Dataset with real pocket features."""
    
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
        # Pocket features (REAL geometric features)
        self.pocket_features = torch.tensor(df[[
            'dfg_chelix_distance',
            'hinge_activation_angle', 
            'volume',
            'n_residues'
        ]].values, dtype=torch.float32)
        
        # Normalize
        self.pocket_features = (self.pocket_features - self.pocket_features.mean(dim=0)) / (self.pocket_features.std(dim=0) + 1e-8)
        
        # Compound features (mock - will be replaced with real descriptors later)
        self.compound_features = torch.randn(len(df), 8)
        
        # Labels
        self.type_labels = torch.tensor((df['type'] == 'Type II').astype(int).values, dtype=torch.long)
        self.dfg_labels = torch.tensor((df['dfg'] == 'out').astype(int).values, dtype=torch.long)
        self.pki_values = torch.tensor(df['pki'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'pocket': self.pocket_features[idx],
            'compound': self.compound_features[idx],
            'type_label': self.type_labels[idx],
            'dfg_label': self.dfg_labels[idx],
            'pki': self.pki_values[idx]
        }


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        pocket = batch['pocket'].to(device)
        compound = batch['compound'].to(device)
        type_label = batch['type_label'].to(device).float()
        dfg_label = batch['dfg_label'].to(device).float()
        pki = batch['pki'].to(device)
        
        optimizer.zero_grad()
        
        # Model outputs: [type2_score, dfg_out_prob, affinity]
        output = model(pocket, compound)
        
        # Multi-task loss
        type_loss = nn.BCELoss()(output[:, 0], type_label)
        dfg_loss = nn.BCELoss()(output[:, 1], dfg_label)
        pki_loss = nn.MSELoss()(output[:, 2], pki)
        
        loss = type_loss + dfg_loss + 0.1 * pki_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    
    all_type_preds = []
    all_type_labels = []
    all_dfg_preds = []
    all_dfg_labels = []
    all_pki_preds = []
    all_pki_true = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pocket = batch['pocket'].to(device)
            compound = batch['compound'].to(device)
            
            # Model outputs: [type2_score, dfg_out_prob, affinity]
            output = model(pocket, compound)
            
            all_type_preds.extend((output[:, 0] > 0.5).cpu().numpy())
            all_type_labels.extend(batch['type_label'].numpy())
            all_dfg_preds.extend((output[:, 1] > 0.5).cpu().numpy())
            all_dfg_labels.extend(batch['dfg_label'].numpy())
            all_pki_preds.extend(output[:, 2].cpu().numpy())
            all_pki_true.extend(batch['pki'].numpy())
    
    metrics = {
        'type_ii_accuracy': accuracy_score(all_type_labels, all_type_preds),
        'dfg_accuracy': accuracy_score(all_dfg_labels, all_dfg_preds),
        'pki_mae': mean_absolute_error(all_pki_true, all_pki_preds),
    }
    
    return metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  TRAINING TYPE II SCREENING MODEL (REAL FEATURES)")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    
    # Load dataset with real features
    df = pd.read_csv('data/klifs_pocket_features.csv')
    print(f"  Total structures: {len(df)}")
    
    # Split
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    print(f"  Train: {len(train_df)} structures")
    print(f"  Val: {len(val_df)} structures")
    
    # Datasets
    train_dataset = KLIFSDataset(train_df)
    val_dataset = KLIFSDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Model
    model = Type2ScreeningModel(
        pocket_dim=4,  # Real features: dfg_chelix_distance, hinge_angle, volume, n_residues
        compound_dim=8,
        hidden_dim=128
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(50):
        print(f"\n  Epoch {epoch+1}/50")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Type II Acc: {val_metrics['type_ii_accuracy']:.1%}")
        print(f"    Val DFG Acc: {val_metrics['dfg_accuracy']:.1%}")
        print(f"    Val pKi MAE: {val_metrics['pki_mae']:.2f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **val_metrics
        })
        
        # Save best model
        val_loss = train_loss  # Simplified
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/screening/best_model_real_features.pt')
            print(f"    ✓ Saved best model")
    
    # Save history
    with open('checkpoints/screening/training_history_real_features.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n  ✓ Training complete!")
    print(f"  Best model saved to: checkpoints/screening/best_model_real_features.pt")


if __name__ == "__main__":
    main()
