#!/usr/bin/env python
"""
Train Hybrid EGNN on CryptoSite Benchmark
==========================================

This script trains the hybrid EGNN (EGNN + classical MD features) 
on the CryptoSite benchmark for cryptic pocket detection.

Usage:
    source .venv312/bin/activate
    python scripts/train_hybrid.py --epochs 100 --lr 1e-3

The hybrid approach uses:
- Classical MD features (DCC, MI, RMSF, PCA) as inputs
- EGNN message passing for geometric learning
- Per-residue binary classification for pocket prediction

This is the scientifically-grounded alternative to the SNN approach.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyaline.features.classical import (
    ClassicalFeatureExtractor, 
    NormalModeGenerator,
    TrajectoryFeatures
)
from hyaline.models.hybrid_egnn import (
    HybridEGNN, 
    build_graph_from_coords
)
from benchmarks.cryptosite_loader import CryptoSiteDataset, CRYPTOSITE_PROTEINS


class CryptoSiteHybridDataset(Dataset):
    """
    Dataset that combines CryptoSite structures with classical MD features
    generated from normal mode perturbations.
    """
    
    def __init__(
        self,
        structures: list,
        n_nm_frames: int = 50,
        nm_amplitude: float = 2.0,
        contact_cutoff: float = 10.0,
        cache_dir: str = None
    ):
        """
        Args:
            structures: List of CryptoSite structure dictionaries
            n_nm_frames: Number of normal mode frames to generate
            nm_amplitude: Amplitude for normal mode perturbations
            contact_cutoff: Distance cutoff for graph edges
            cache_dir: Directory to cache extracted features
        """
        self.structures = structures
        self.n_nm_frames = n_nm_frames
        self.nm_amplitude = nm_amplitude
        self.contact_cutoff = contact_cutoff
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        self.nm_generator = NormalModeGenerator(
            n_modes=10,
            amplitude=nm_amplitude,
            n_frames=n_nm_frames
        )
        self.feature_extractor = ClassicalFeatureExtractor()
        
        # Precompute features if caching
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._precompute_all()
    
    def _precompute_all(self):
        """Precompute features for all structures."""
        self.data_cache = []
        
        for i, struct in enumerate(tqdm(self.structures, desc="Preparing data")):
            cache_path = None
            if self.cache_dir:
                cache_path = self.cache_dir / f"{struct['pdb_id']}.npz"
                if cache_path.exists():
                    data = np.load(cache_path, allow_pickle=True)
                    self.data_cache.append({
                        'node_features': data['node_features'],
                        'edge_features': data['edge_features'],
                        'pos': data['pos'],
                        'labels': data['labels'],
                        'pdb_id': str(data['pdb_id'])
                    })
                    continue
            
            # Get coordinates
            coords = struct['coords']  # [N, 3] Cα coords
            labels = struct['labels']  # [N] binary pocket labels
            
            # Generate normal mode trajectory
            try:
                trajectory = self.nm_generator.generate(coords)
            except Exception as e:
                print(f"Warning: NM failed for {struct['pdb_id']}: {e}")
                # Fallback: add random noise
                trajectory = coords + np.random.randn(self.n_nm_frames, len(coords), 3) * 0.5
            
            # Extract classical features
            features = self.feature_extractor.extract_all(trajectory, fast=True)
            
            # Prepare data dict
            data = {
                'node_features': features.node_features,
                'edge_features': features.edge_features,
                'pos': coords,
                'labels': labels,
                'pdb_id': struct['pdb_id']
            }
            
            # Cache
            if cache_path:
                np.savez(cache_path, **data)
            
            self.data_cache.append(data)
    
    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        data = self.data_cache[idx]
        
        # Convert to tensors
        pos = torch.from_numpy(data['pos']).float()
        node_features = torch.from_numpy(data['node_features']).float()
        labels = torch.from_numpy(data['labels']).float()
        
        # Build graph
        edge_index = build_graph_from_coords(pos, cutoff=self.contact_cutoff)
        
        # Get edge features from the [N, N, 3] matrix
        row, col = edge_index
        edge_features_matrix = data['edge_features']
        edge_features = torch.from_numpy(
            edge_features_matrix[row.numpy(), col.numpy()]
        ).float()
        
        return {
            'node_features': node_features,
            'pos': pos,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'labels': labels,
            'pdb_id': data['pdb_id']
        }


def collate_fn(batch):
    """Custom collate for variable-size graphs."""
    # For now, process one graph at a time
    # TODO: Use torch_geometric Batch for proper batching
    return batch[0] if len(batch) == 1 else batch


def compute_metrics(preds, labels):
    """Compute evaluation metrics."""
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # Filter out any NaNs
    mask = ~np.isnan(preds_np) & ~np.isnan(labels_np)
    preds_np = preds_np[mask]
    labels_np = labels_np[mask]
    
    if len(np.unique(labels_np)) < 2:
        return {'roc_auc': 0.5, 'pr_auc': 0.0}
    
    try:
        roc_auc = roc_auc_score(labels_np, preds_np)
        precision, recall, _ = precision_recall_curve(labels_np, preds_np)
        pr_auc = auc(recall, precision)
    except Exception:
        roc_auc, pr_auc = 0.5, 0.0
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        # Move to device
        node_features = batch['node_features'].to(device)
        pos = batch['pos'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(node_features, pos, edge_index, edge_features)
        
        # Loss
        loss = criterion(outputs['pocket_prob'], labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(outputs['pocket_prob'])
        all_labels.append(labels)
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    per_protein = {}
    
    for batch in dataloader:
        node_features = batch['node_features'].to(device)
        pos = batch['pos'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(node_features, pos, edge_index, edge_features)
        loss = criterion(outputs['pocket_prob'], labels)
        
        total_loss += loss.item()
        all_preds.append(outputs['pocket_prob'])
        all_labels.append(labels)
        
        # Per-protein metrics
        pdb_id = batch['pdb_id']
        per_protein[pdb_id] = compute_metrics(outputs['pocket_prob'], labels)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    metrics['per_protein'] = per_protein
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid EGNN on CryptoSite')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/cryptosite',
                        help='CryptoSite data directory')
    parser.add_argument('--cache_dir', type=str, default='data/feature_cache',
                        help='Feature cache directory')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of EGNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Normal modes
    parser.add_argument('--nm_frames', type=int, default=50,
                        help='Number of normal mode frames')
    parser.add_argument('--nm_amplitude', type=float, default=2.0,
                        help='Normal mode amplitude (Angstroms)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--pos_weight', type=float, default=5.0,
                        help='Positive class weight for imbalanced data')
    
    # Infrastructure
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints/hybrid_egnn',
                        help='Save directory')
    parser.add_argument('--wandb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='hyaline-hybrid',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Set seed
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
    
    # Initialize W&B
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"hybrid_egnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except Exception as e:
            print(f"W&B init failed: {e}")
            args.wandb = False
    
    # Load CryptoSite data
    print("Loading CryptoSite dataset...")
    
    try:
        # Try to use CryptoSiteDataset
        cryptosite = CryptoSiteDataset(data_dir=args.data_dir, split='test')
        structures = []
        for i in range(len(cryptosite)):
            data = cryptosite[i]
            structures.append({
                'pdb_id': data.pdb_id if hasattr(data, 'pdb_id') else f'protein_{i}',
                'coords': data.pos.numpy() if hasattr(data.pos, 'numpy') else np.array(data.pos),
                'labels': data.y.numpy() if hasattr(data.y, 'numpy') else np.array(data.y)
            })
    except Exception as e:
        print(f"CryptoSiteDataset loading failed: {e}")
        print("Creating synthetic data for testing architecture...")
        # Create synthetic data for testing the pipeline
        structures = []
        for i, (pdb_apo, chain, pdb_holo, cryptic_residues) in enumerate(CRYPTOSITE_PROTEINS):
            n_residues = np.random.randint(80, 200)
            coords = np.cumsum(np.random.randn(n_residues, 3) * 3.8, axis=0)
            labels = np.zeros(n_residues)
            # Use actual cryptic residue indices if valid
            for res in cryptic_residues:
                if res < n_residues:
                    labels[res] = 1
            # If no valid residues, create a synthetic pocket
            if labels.sum() == 0:
                pocket_start = np.random.randint(10, n_residues - 20)
                labels[pocket_start:pocket_start + 8] = 1
            structures.append({
                'pdb_id': pdb_apo,
                'coords': coords.astype(np.float32),
                'labels': labels.astype(np.float32)
            })
    
    print(f"Loaded {len(structures)} structures")
    
    # Split data (leave-one-out cross-validation is standard for CryptoSite)
    # For simplicity, use 80/20 split here
    n_train = int(0.8 * len(structures))
    train_structures = structures[:n_train]
    val_structures = structures[n_train:]
    
    print(f"Train: {len(train_structures)}, Val: {len(val_structures)}")
    
    # Create datasets
    train_dataset = CryptoSiteHybridDataset(
        train_structures,
        n_nm_frames=args.nm_frames,
        nm_amplitude=args.nm_amplitude,
        cache_dir=args.cache_dir
    )
    
    val_dataset = CryptoSiteHybridDataset(
        val_structures,
        n_nm_frames=args.nm_frames,
        nm_amplitude=args.nm_amplitude,
        cache_dir=args.cache_dir
    )
    
    # Create dataloaders (batch_size=1 for variable graph sizes)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = HybridEGNN(
        node_input_dim=2,      # RMSF, PCA contribution
        edge_input_dim=3,      # DCC, MI, contact_freq
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Loss function with class weighting
    criterion = nn.BCELoss()
    # Alternative: weighted BCE for imbalanced data
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_auc = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train AUC: {train_metrics['roc_auc']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f} | "
              f"Val PR-AUC: {val_metrics['pr_auc']:.4f}")
        
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/roc_auc': train_metrics['roc_auc'],
                'val/loss': val_metrics['loss'],
                'val/roc_auc': val_metrics['roc_auc'],
                'val/pr_auc': val_metrics['pr_auc'],
                'lr': scheduler.get_last_lr()[0]
            })
        
        # Save best model
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': vars(args)
            }, save_dir / 'best_model.pt')
            print(f"  → Saved best model (AUC: {best_val_auc:.4f})")
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    
    # Load best model
    checkpoint = torch.load(save_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_metrics = evaluate(model, val_loader, criterion, device)
    print(f"Best Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"Best Val PR-AUC: {val_metrics['pr_auc']:.4f}")
    
    print("\nPer-protein results:")
    for pdb_id, metrics in val_metrics['per_protein'].items():
        print(f"  {pdb_id}: ROC-AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")
    
    # Save final results
    results = {
        'best_val_roc_auc': val_metrics['roc_auc'],
        'best_val_pr_auc': val_metrics['pr_auc'],
        'per_protein': val_metrics['per_protein'],
        'config': vars(args)
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if args.wandb:
        wandb.finish()
    
    print(f"\nResults saved to {save_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
