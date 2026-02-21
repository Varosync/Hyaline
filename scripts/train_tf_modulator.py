#!/usr/bin/env python
"""
Train TF-Modulator on TF-DNA and CryptoSite Data
=================================================

Unified training script for the Hyaline TF-Modulator architecture.

Uses EXISTING data:
- data/cryptosite/ (27 PDB files)
- data/tf_dna/ (6 TF-DNA PDB files from Nectar research)

Usage:
    source .venv312/bin/activate
    python scripts/train_tf_modulator.py --epochs 100

Architecture (from system spec):
- 6 EGNN layers
- Two heads: pocket prediction + druggability scoring
- Node features: element, hybridization, residue, charge, dynamics
- Edge features: distance, bond type, interface flag, dynamics
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyaline.models.tf_modulator import TFModulator, TFModulatorConfig
from hyaline.features.geometric import GeometricFeatureExtractor, extract_from_pdb_file


# Focal loss for imbalanced data
class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class TFDNADataset(Dataset):
    """
    Dataset for TF-DNA and cryptic pocket structures.
    
    Loads PDB files and extracts geometric features.
    """
    
    def __init__(
        self,
        pdb_files: List[Path],
        cache_dir: Optional[Path] = None,
        include_dynamics: bool = False,
        max_nodes: int = 2000,  # Limit nodes to prevent OOM
    ):
        self.pdb_files = pdb_files
        self.cache_dir = cache_dir
        self.include_dynamics = include_dynamics
        self.max_nodes = max_nodes
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = GeometricFeatureExtractor()
        self.data_cache = []
        
        self._load_all()
    
    def _load_all(self):
        """Load and process all PDB files."""
        for pdb_path in tqdm(self.pdb_files, desc="Loading structures"):
            cache_path = None
            if self.cache_dir:
                cache_path = self.cache_dir / f"{pdb_path.stem}.npz"
                if cache_path.exists():
                    data = np.load(cache_path, allow_pickle=True)
                    self.data_cache.append({
                        'node_features': data['node_features'],
                        'edge_features': data['edge_features'],
                        'edge_index': data['edge_index'],
                        'pos': data['pos'],
                        'is_dna': data['is_dna'],
                        'pdb_id': str(data['pdb_id']),
                    })
                    continue
            
            try:
                features = extract_from_pdb_file(str(pdb_path))
                
                # Subsample large structures to prevent OOM
                if len(features.pos) > self.max_nodes:
                    indices = np.random.choice(
                        len(features.pos), self.max_nodes, replace=False
                    )
                    indices = np.sort(indices)
                    features = self._subsample_features(features, indices)
                
                # Create synthetic pocket labels based on interface proximity
                # In real training, these come from fpocket or known binding sites
                labels = self._generate_pocket_labels(features)
                
                data = {
                    'node_features': features.node_features,
                    'edge_features': features.edge_features,
                    'edge_index': features.edge_index,
                    'pos': features.pos,
                    'is_dna': features.is_dna,
                    'labels': labels,
                    'pdb_id': pdb_path.stem,
                }
                
                if cache_path:
                    np.savez(cache_path, **data)
                
                self.data_cache.append(data)
                
            except Exception as e:
                print(f"Warning: Failed to load {pdb_path.name}: {e}")
    
    def _generate_pocket_labels(self, features) -> np.ndarray:
        """
        Generate pocket labels.
        
        For TF-DNA complexes: residues near DNA interface are potential pockets.
        For cryptic pockets: would use fpocket output.
        
        This is a placeholder - real labels should come from:
        1. fpocket predictions
        2. Known binding site annotations
        3. Ligand proximity in holo structures
        """
        n_nodes = len(features.pos)
        labels = np.zeros(n_nodes, dtype=np.float32)
        
        # Heuristic: interface residues near DNA are potential pockets
        if features.is_dna.any():
            dna_coords = features.pos[features.is_dna]
            protein_coords = features.pos[~features.is_dna]
            
            if len(dna_coords) > 0 and len(protein_coords) > 0:
                # Find protein atoms within 6A of DNA
                for i, coord in enumerate(features.pos):
                    if features.is_dna[i]:
                        continue
                    dist_to_dna = np.min(np.linalg.norm(dna_coords - coord, axis=1))
                    if dist_to_dna < 6.0:
                        labels[i] = 1.0
        else:
            # For non-DNA structures, mark ~10% random as pocket (placeholder)
            pocket_idx = np.random.choice(n_nodes, size=max(1, n_nodes // 10), replace=False)
            labels[pocket_idx] = 1.0
        
        return labels
    
    def _subsample_features(self, features, indices):
        """Subsample a GeometricFeatures object to reduce size."""
        from hyaline.features.geometric import GeometricFeatures
        
        # Create index mapping for edge reindexing
        idx_map = {old: new for new, old in enumerate(indices)}
        N = len(indices)
        
        # Filter edges that have both endpoints in subsample
        valid_edges = []
        for i in range(features.edge_index.shape[1]):
            src, dst = features.edge_index[0, i], features.edge_index[1, i]
            if src in idx_map and dst in idx_map:
                valid_edges.append(i)
        
        # Subsample edges
        if len(valid_edges) > 0:
            edge_index = features.edge_index[:, valid_edges]
            edge_features = features.edge_features[valid_edges]
            # Remap indices
            edge_index = np.array([
                [idx_map[e] for e in edge_index[0]],
                [idx_map[e] for e in edge_index[1]],
            ])
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, features.edge_features.shape[1]), dtype=np.float32)
        
        return GeometricFeatures(
            node_features=features.node_features[indices],
            edge_features=edge_features,
            edge_index=edge_index,
            pos=features.pos[indices],
            is_dna=features.is_dna[indices],
            is_ca=features.is_ca[indices],
            residue_ids=[features.residue_ids[i] for i in indices],
            chain_ids=[features.chain_ids[i] for i in indices],
        )
    
    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        data = self.data_cache[idx]
        
        return {
            'node_features': torch.from_numpy(data['node_features']).float(),
            'edge_features': torch.from_numpy(data['edge_features']).float(),
            'edge_index': torch.from_numpy(data['edge_index']).long(),
            'pos': torch.from_numpy(data['pos']).float(),
            'labels': torch.from_numpy(data.get('labels', np.zeros(len(data['pos'])))).float(),
            'pdb_id': data['pdb_id'],
        }


def collate_fn(batch):
    """Custom collate for variable-size graphs."""
    return batch[0] if len(batch) == 1 else batch


def compute_metrics(pocket_pred, drug_pred, labels):
    """Compute evaluation metrics."""
    pocket_np = pocket_pred.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    mask = ~np.isnan(pocket_np) & ~np.isnan(labels_np)
    pocket_np = pocket_np[mask]
    labels_np = labels_np[mask]
    
    if len(np.unique(labels_np)) < 2:
        return {'roc_auc': 0.5, 'pr_auc': 0.0}
    
    try:
        roc_auc = roc_auc_score(labels_np, pocket_np)
        precision, recall, _ = precision_recall_curve(labels_np, pocket_np)
        pr_auc = auc(recall, precision)
    except Exception:
        roc_auc, pr_auc = 0.5, 0.0
    
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_pocket_preds = []
    all_drug_preds = []
    all_labels = []
    
    for batch in dataloader:
        node_features = batch['node_features'].to(device)
        edge_features = batch['edge_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        pos = batch['pos'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(node_features, pos, edge_index, edge_features)
        
        # Combined loss on both heads
        pocket_loss = criterion(outputs['pocket_prob'], labels)
        # Druggability supervised by pocket labels (correlation)
        drug_loss = criterion(outputs['druggability'], labels)
        loss = pocket_loss + 0.5 * drug_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_pocket_preds.append(outputs['pocket_prob'])
        all_drug_preds.append(outputs['druggability'])
        all_labels.append(labels)
    
    all_pocket_preds = torch.cat(all_pocket_preds)
    all_drug_preds = torch.cat(all_drug_preds)
    all_labels = torch.cat(all_labels)
    
    metrics = compute_metrics(all_pocket_preds, all_drug_preds, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_pocket_preds = []
    all_labels = []
    per_protein = {}
    
    for batch in dataloader:
        node_features = batch['node_features'].to(device)
        edge_features = batch['edge_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        pos = batch['pos'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(node_features, pos, edge_index, edge_features)
        
        pocket_loss = criterion(outputs['pocket_prob'], labels)
        drug_loss = criterion(outputs['druggability'], labels)
        loss = pocket_loss + 0.5 * drug_loss
        
        total_loss += loss.item()
        all_pocket_preds.append(outputs['pocket_prob'])
        all_labels.append(labels)
        
        pdb_id = batch['pdb_id']
        per_protein[pdb_id] = compute_metrics(
            outputs['pocket_prob'], outputs['druggability'], labels
        )
    
    all_pocket_preds = torch.cat(all_pocket_preds)
    all_labels = torch.cat(all_labels)
    
    metrics = compute_metrics(all_pocket_preds, None, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    metrics['per_protein'] = per_protein
    
    return metrics


def discover_pdb_files(data_dirs: List[str]) -> List[Path]:
    """Find all PDB files in given directories."""
    pdb_files = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            pdb_files.extend(data_path.glob("*.pdb"))
    return sorted(set(pdb_files))


def main():
    parser = argparse.ArgumentParser(description='Train TF-Modulator')
    
    # Data - use existing directories
    parser.add_argument('--data_dirs', type=str, nargs='+',
                        default=['data/tf_dna', 'data/cryptosite'],
                        help='Directories containing PDB files')
    parser.add_argument('--cache_dir', type=str, default='data/tf_cache',
                        help='Feature cache directory')
    
    # Model (per system spec)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)  # Per spec
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Infrastructure
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints/tf_modulator')
    parser.add_argument('--wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Discover PDB files
    print("\nDiscovering PDB files...")
    pdb_files = discover_pdb_files(args.data_dirs)
    print(f"Found {len(pdb_files)} PDB files:")
    for data_dir in args.data_dirs:
        count = len([p for p in pdb_files if data_dir in str(p)])
        print(f"  {data_dir}: {count} files")
    
    if len(pdb_files) == 0:
        print("ERROR: No PDB files found!")
        return
    
    # Split data
    n_train = int(0.8 * len(pdb_files))
    train_files = pdb_files[:n_train]
    val_files = pdb_files[n_train:]
    
    print(f"\nTrain: {len(train_files)}, Val: {len(val_files)}")
    
    # Create datasets
    train_dataset = TFDNADataset(
        train_files,
        cache_dir=Path(args.cache_dir),
    )
    
    val_dataset = TFDNADataset(
        val_files,
        cache_dir=Path(args.cache_dir),
    )
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Create model with spec configuration
    config = TFModulatorConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    model = TFModulator(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TFModulator")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_auc = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train AUC: {train_metrics['roc_auc']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f}")
        
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': vars(args),
            }, save_dir / 'best_model.pt')
            print(f"  → Saved best model (AUC: {best_val_auc:.4f})")
    
    # Final results
    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Best Val ROC-AUC: {best_val_auc:.4f}")
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
