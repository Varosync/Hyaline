#!/usr/bin/env python
"""Train TFActivationModel with REAL TF-DNA structures."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import json

from hyaline.models.tf_activation_model import TFActivationModel, TFActivationConfig
from hyaline.loaders.pdb_loader import load_tf_structures, TF_PDB_MAPPING
from hyaline.loaders.tf_activation_data import SCENICContext, CELL_TYPES, TF_NAMES, VALIDATION_CASES, EXPECTED_LEVELS


class RealTFDataset(Dataset):
    """Dataset with real TF-DNA structures + SCENIC context."""
    
    def __init__(self, samples, node_dim=32, edge_dim=8):
        self.samples = samples
        self.node_dim = node_dim
        self.edge_dim = edge_dim
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'node_features': torch.from_numpy(s['node_features']).float(),
            'pos': torch.from_numpy(s['pos']).float(),
            'edge_index': torch.from_numpy(s['edge_index']).long(),
            'edge_attr': torch.from_numpy(s['edge_attr']).float(),
            'cell_type_idx': torch.tensor([s['cell_type_idx']]).long(),
            'tf_activity': torch.from_numpy(s['tf_activity']).float().unsqueeze(0),
            'chromatin_topics': torch.from_numpy(s['chromatin_topics']).float().unsqueeze(0),
            'coactivator_expr': torch.from_numpy(s['coactivator_expr']).float().unsqueeze(0),
            'label': torch.tensor([s['label']]).float(),
            'tf_name': s['tf_name'],
            'cell_type': s['cell_type'],
        }


def create_real_dataset(data_dir, cell_types, n_tfs=200, n_topics=30, n_coactivators=20):
    """Create dataset with real PDB structures."""
    structures = load_tf_structures(Path(data_dir))
    samples = []
    
    for tf_name, struct_list in structures.items():
        for struct in struct_list:
            for cell_type in cell_types:
                context = SCENICContext.from_cell_type(
                    cell_type, tf_name, n_tfs, n_topics, n_coactivators
                )
                
                # Generate label based on TF-lineage match
                from hyaline.loaders.tf_activation_data import generate_activation_label
                label = float(generate_activation_label(tf_name, cell_type) > 0.5)
                
                samples.append({
                    'node_features': struct.node_features,
                    'pos': struct.pos,
                    'edge_index': struct.edge_index,
                    'edge_attr': struct.edge_attr,
                    'cell_type_idx': context.cell_type_idx,
                    'tf_activity': context.tf_activity,
                    'chromatin_topics': context.chromatin_topics,
                    'coactivator_expr': context.coactivator_expr,
                    'label': label,
                    'tf_name': tf_name,
                    'cell_type': cell_type,
                })
    
    np.random.shuffle(samples)
    return samples


def collate_fn(batch):
    return batch[0] if len(batch) == 1 else batch


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds, labels = 0, [], []
    
    for batch in loader:
        nf = batch['node_features'].to(device)
        pos = batch['pos'].to(device)
        ei = batch['edge_index'].to(device)
        ea = batch['edge_attr'].to(device)
        ct = batch['cell_type_idx'].to(device)
        tf = batch['tf_activity'].to(device)
        ch = batch['chromatin_topics'].to(device)
        co = batch['coactivator_expr'].to(device)
        y = batch['label'].to(device)
        
        optimizer.zero_grad()
        out = model(nf, pos, ei, ea, ct, tf, ch, co)
        loss = criterion(out.activation_prob, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds.append(out.activation_prob.detach().cpu())
        labels.append(y.cpu())
    
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    try:
        auc = roc_auc_score(labels, preds)
    except:
        auc = 0.5
    return total_loss / len(loader), auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0, [], []
    
    for batch in loader:
        nf = batch['node_features'].to(device)
        pos = batch['pos'].to(device)
        ei = batch['edge_index'].to(device)
        ea = batch['edge_attr'].to(device)
        ct = batch['cell_type_idx'].to(device)
        tf = batch['tf_activity'].to(device)
        ch = batch['chromatin_topics'].to(device)
        co = batch['coactivator_expr'].to(device)
        y = batch['label'].to(device)
        
        out = model(nf, pos, ei, ea, ct, tf, ch, co)
        loss = criterion(out.activation_prob, y)
        total_loss += loss.item()
        preds.append(out.activation_prob.cpu())
        labels.append(y.cpu())
    
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    try:
        auc = roc_auc_score(labels, preds)
    except:
        auc = 0.5
    return total_loss / len(loader), auc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Config
    config = TFActivationConfig(
        node_input_dim=32, edge_input_dim=8,
        hidden_dim=128, num_egnn_layers=4,
        dropout=0.2, base_threshold=1.0,
    )
    
    # Create dataset with real structures
    print("\nLoading real TF-DNA structures...")
    cell_types = ['melanocyte', 'hepatocyte', 'cardiomyocyte', 'fibroblast', 
                  'neuron', 't_cell', 'epithelial', 'esc']
    samples = create_real_dataset('data/tf_dna_structures', cell_types)
    print(f"Created {len(samples)} samples")
    
    # Split
    n_train = int(0.8 * len(samples))
    train_samples, val_samples = samples[:n_train], samples[n_train:]
    
    train_dataset = RealTFDataset(train_samples)
    val_dataset = RealTFDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = TFActivationModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training
    save_dir = Path('checkpoints/tf_real')
    save_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0
    
    print("\nTraining...")
    print("-" * 60)
    
    for epoch in range(1, 51):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'model': model.state_dict(), 'config': config.__dict__}, 
                      save_dir / 'best.pt')
            print(f"  -> Saved best (AUC: {best_auc:.4f})")
    
    print(f"\nBest Val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
