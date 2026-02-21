#!/usr/bin/env python
"""
Train TFActivationModel with SCENIC+ Context
=============================================

Training script for context-dependent TF activation prediction.
Uses spiking dynamics + SCENIC+ cellular context.

The model predicts: P(TF activates | structure, cellular context)

Architecture:
- ContextEncoder: SCENIC+ features → threshold modulation
- SpikingEGNN: Equivariant message passing with LIF dynamics
- ActivationHead: Synchronization → activation probability

Usage:
    source .venv312/bin/activate
    python scripts/train_tf_activation.py --epochs 100

    # With synthetic data for testing:
    python scripts/train_tf_activation.py --synthetic --epochs 10
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyaline.models.tf_activation_model import TFActivationModel, TFActivationConfig
from hyaline.loaders import (
    create_tf_activation_dataset,
    create_validation_dataset,
    CELL_TYPES,
    TF_NAMES,
)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class TFActivationLoss(nn.Module):
    """
    Combined loss for TF activation prediction.
    
    Components:
    1. Activation BCE: Binary classification loss
    2. Sync regularization: Encourage meaningful sync scores
    3. Spike activity: Prevent spike collapse (all zeros) or explosion
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        sync_weight: float = 0.1,
        activity_weight: float = 0.01,
        target_spike_rate: float = 0.2,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.sync_weight = sync_weight
        self.activity_weight = activity_weight
        self.target_spike_rate = target_spike_rate
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        output,  # TFActivationOutput
        labels: torch.Tensor,  # [batch] binary
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        losses = {}
        
        # Activation BCE
        losses['bce'] = self.bce(output.activation_prob, labels)
        
        # Sync regularization: activated samples should have higher sync
        # sync_score should correlate with activation
        sync_target = labels * 0.7 + (1 - labels) * 0.3  # High for active, low for inactive
        losses['sync'] = ((output.sync_score.flatten() - sync_target) ** 2).mean()
        
        # Spike rate regularization
        spike_rate = output.spike_rate.mean()
        losses['activity'] = (spike_rate - self.target_spike_rate) ** 2
        
        # Total
        losses['total'] = (
            self.bce_weight * losses['bce'] +
            self.sync_weight * losses['sync'] +
            self.activity_weight * losses['activity']
        )
        
        return losses


class TFActivationDataset(Dataset):
    """
    Dataset for TF activation prediction.
    
    Each sample contains:
    - Graph: node_features, pos, edge_index, edge_attr
    - Context: cell_type_idx, tf_activity, chromatin_topics, coactivator_expr
    - Label: activation (0 or 1)
    """
    
    def __init__(
        self,
        data_list: List[Dict],
        config: TFActivationConfig,
    ):
        """
        Args:
            data_list: List of data dicts with keys:
                - node_features: [N, node_dim]
                - pos: [N, 3]
                - edge_index: [2, E]
                - edge_attr: [E, edge_dim]
                - cell_type_idx: int
                - tf_activity: [n_tfs]
                - chromatin_topics: [n_topics]
                - coactivator_expr: [n_coactivators]
                - label: float (0 or 1)
            config: Model config for dimension validation
        """
        self.data_list = data_list
        self.config = config
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        return {
            'node_features': torch.from_numpy(data['node_features']).float(),
            'pos': torch.from_numpy(data['pos']).float(),
            'edge_index': torch.from_numpy(data['edge_index']).long(),
            'edge_attr': torch.from_numpy(data['edge_attr']).float(),
            'cell_type_idx': torch.tensor([data['cell_type_idx']]).long(),
            'tf_activity': torch.from_numpy(data['tf_activity']).float().unsqueeze(0),
            'chromatin_topics': torch.from_numpy(data['chromatin_topics']).float().unsqueeze(0),
            'coactivator_expr': torch.from_numpy(data['coactivator_expr']).float().unsqueeze(0),
            'label': torch.tensor([data['label']]).float(),
        }


def generate_synthetic_data(
    n_samples: int,
    config: TFActivationConfig,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate synthetic training data for testing.
    
    Creates random graphs with SCENIC+ context and synthetic labels.
    Labels are correlated with context features for meaningful training.
    """
    np.random.seed(seed)
    data_list = []
    
    for i in range(n_samples):
        # Random graph size
        n_nodes = np.random.randint(20, 100)
        n_edges = np.random.randint(n_nodes * 2, n_nodes * 5)
        
        # Node features and positions
        node_features = np.random.randn(n_nodes, config.node_input_dim).astype(np.float32)
        pos = np.random.randn(n_nodes, 3).astype(np.float32) * 10
        
        # Edges (random connectivity)
        edge_index = np.random.randint(0, n_nodes, (2, n_edges)).astype(np.int64)
        edge_attr = np.random.randn(n_edges, config.edge_input_dim).astype(np.float32)
        
        # SCENIC+ context
        cell_type_idx = np.random.randint(0, config.n_cell_types)
        tf_activity = np.random.rand(config.n_tfs).astype(np.float32)
        chromatin_topics = np.random.rand(config.n_topics).astype(np.float32)
        coactivator_expr = np.random.rand(config.n_coactivators).astype(np.float32)
        
        # Synthetic label: correlated with context
        # High TF activity + high chromatin accessibility → higher activation probability
        activation_score = (
            0.3 * tf_activity.mean() +
            0.3 * chromatin_topics.mean() +
            0.2 * coactivator_expr.mean() +
            0.2 * np.random.rand()  # Some noise
        )
        label = float(activation_score > 0.5)
        
        data_list.append({
            'node_features': node_features,
            'pos': pos,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'cell_type_idx': cell_type_idx,
            'tf_activity': tf_activity,
            'chromatin_topics': chromatin_topics,
            'coactivator_expr': coactivator_expr,
            'label': label,
        })
    
    return data_list


def collate_fn(batch):
    """Custom collate - process one sample at a time."""
    return batch[0] if len(batch) == 1 else batch


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_sync = []
    all_spike_rate = []
    
    for batch in dataloader:
        # Move to device
        node_features = batch['node_features'].to(device)
        pos = batch['pos'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr = batch['edge_attr'].to(device)
        cell_type_idx = batch['cell_type_idx'].to(device)
        tf_activity = batch['tf_activity'].to(device)
        chromatin_topics = batch['chromatin_topics'].to(device)
        coactivator_expr = batch['coactivator_expr'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        output = model(
            node_features, pos, edge_index, edge_attr,
            cell_type_idx, tf_activity, chromatin_topics, coactivator_expr
        )
        
        # Loss
        losses = criterion(output, labels)
        loss = losses['total']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track
        total_loss += loss.item()
        all_preds.append(output.activation_prob.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_sync.append(output.sync_score.mean().item())
        all_spike_rate.append(output.spike_rate.mean().item())
    
    # Metrics
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    try:
        roc_auc = roc_auc_score(labels, preds)
    except ValueError:
        roc_auc = 0.5
    
    accuracy = accuracy_score(labels, (preds > 0.5).astype(float))
    
    return {
        'loss': total_loss / len(dataloader),
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'sync_score': np.mean(all_sync),
        'spike_rate': np.mean(all_spike_rate),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        node_features = batch['node_features'].to(device)
        pos = batch['pos'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr = batch['edge_attr'].to(device)
        cell_type_idx = batch['cell_type_idx'].to(device)
        tf_activity = batch['tf_activity'].to(device)
        chromatin_topics = batch['chromatin_topics'].to(device)
        coactivator_expr = batch['coactivator_expr'].to(device)
        labels = batch['label'].to(device)
        
        output = model(
            node_features, pos, edge_index, edge_attr,
            cell_type_idx, tf_activity, chromatin_topics, coactivator_expr
        )
        
        losses = criterion(output, labels)
        total_loss += losses['total'].item()
        
        all_preds.append(output.activation_prob.cpu())
        all_labels.append(labels.cpu())
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    try:
        roc_auc = roc_auc_score(labels, preds)
    except ValueError:
        roc_auc = 0.5
    
    accuracy = accuracy_score(labels, (preds > 0.5).astype(float))
    
    return {
        'loss': total_loss / len(dataloader),
        'roc_auc': roc_auc,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description='Train TFActivationModel')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/tf_activation',
                        help='Directory with training data')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--n_synthetic', type=int, default=200,
                        help='Number of synthetic samples')
    parser.add_argument('--use_bio_data', action='store_true',
                        help='Use biologically-informed synthetic data')
    parser.add_argument('--validate_cases', action='store_true',
                        help='Evaluate on validation cases after training')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_cell_types', type=int, default=50)
    parser.add_argument('--n_tfs', type=int, default=200)
    parser.add_argument('--n_topics', type=int, default=30)
    parser.add_argument('--n_coactivators', type=int, default=20)
    
    # Spiking parameters
    parser.add_argument('--beta', type=float, default=0.9,
                        help='LIF leak factor')
    parser.add_argument('--base_threshold', type=float, default=0.5,
                        help='Base spike threshold (lower = more spikes)')
    parser.add_argument('--surrogate_slope', type=float, default=25.0,
                        help='Surrogate gradient slope')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (1 for variable graph sizes)')
    
    # Loss weights
    parser.add_argument('--bce_weight', type=float, default=1.0)
    parser.add_argument('--sync_weight', type=float, default=0.1)
    parser.add_argument('--activity_weight', type=float, default=0.01)
    
    # Infrastructure
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints/tf_activation')
    
    args = parser.parse_args()
    
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Model config
    config = TFActivationConfig(
        node_input_dim=32,
        edge_input_dim=8,
        n_cell_types=args.n_cell_types,
        n_tfs=args.n_tfs,
        n_topics=args.n_topics,
        n_coactivators=args.n_coactivators,
        hidden_dim=args.hidden_dim,
        num_egnn_layers=args.num_layers,
        dropout=args.dropout,
        beta=args.beta,
        base_threshold=args.base_threshold,
        surrogate_slope=args.surrogate_slope,
    )
    
    # Create model
    model = TFActivationModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TFActivationModel")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  EGNN layers: {config.num_egnn_layers}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Spike threshold: {config.base_threshold}")
    print(f"  Beta (leak): {config.beta}")
    
    # Data
    if args.use_bio_data:
        print("\nCreating biologically-informed dataset...")
        # Use lineage-specific TFs for meaningful training
        full_dataset = create_tf_activation_dataset(
            pdb_dir=Path(args.data_dir) if Path(args.data_dir).exists() else None,
            cell_types=CELL_TYPES[:15],  # 15 cell types
            tf_names=list(TF_NAMES[:20]),  # 20 TFs
            samples_per_combination=2,
            n_tfs=args.n_tfs,
            n_topics=args.n_topics,
            n_coactivators=args.n_coactivators,
            node_dim=config.node_input_dim,
            edge_dim=config.edge_input_dim,
        )
        
        # Split
        n_train = int(0.8 * len(full_dataset))
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, len(full_dataset)))
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
    elif args.synthetic:
        print(f"\nGenerating {args.n_synthetic} synthetic samples...")
        data_list = generate_synthetic_data(args.n_synthetic, config)
        
        # Split data
        n_train = int(0.8 * len(data_list))
        train_data = data_list[:n_train]
        val_data = data_list[n_train:]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Datasets
        train_dataset = TFActivationDataset(train_data, config)
        val_dataset = TFActivationDataset(val_data, config)
    else:
        # Load real data
        data_path = Path(args.data_dir)
        if not data_path.exists():
            print(f"ERROR: Data directory {data_path} not found!")
            print("Use --synthetic or --use_bio_data for testing")
            return
        # TODO: Implement real data loading from files
        raise NotImplementedError("Real data loading not yet implemented. Use --synthetic or --use_bio_data")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    
    # Loss and optimizer
    criterion = TFActivationLoss(
        bce_weight=args.bce_weight,
        sync_weight=args.sync_weight,
        activity_weight=args.activity_weight,
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    best_val_auc = 0
    print("\nStarting training...")
    print("-" * 70)
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Print progress
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train AUC: {train_metrics['roc_auc']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Sync: {train_metrics['sync_score']:.3f} | "
            f"Spike: {train_metrics['spike_rate']:.3f}"
        )
        
        # Save best model
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': vars(args),
                'model_config': config.__dict__,
            }, save_dir / 'best_model.pt')
            print(f"  → Saved best model (AUC: {best_val_auc:.4f})")
    
    # Final results
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best Val ROC-AUC: {best_val_auc:.4f}")
    print(f"Results saved to: {save_dir}")
    
    # Validation cases evaluation
    if args.validate_cases or args.use_bio_data:
        print("\n" + "=" * 70)
        print("Evaluating on Biological Validation Cases")
        print("=" * 70)
        
        # Load best model
        checkpoint = torch.load(save_dir / 'best_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create validation dataset
        val_cases_dataset = create_validation_dataset(
            n_tfs=args.n_tfs,
            n_topics=args.n_topics,
            n_coactivators=args.n_coactivators,
            node_dim=config.node_input_dim,
            edge_dim=config.edge_input_dim,
        )
        
        val_cases_loader = DataLoader(
            val_cases_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
        )
        
        # Evaluate each case
        print(f"\n{'TF':<10} {'Cell Type':<15} {'Pred':<8} {'Label':<8} {'Correct':<8}")
        print("-" * 55)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_cases_loader):
                node_features = batch['node_features'].to(device)
                pos = batch['pos'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr = batch['edge_attr'].to(device)
                cell_type_idx = batch['cell_type_idx'].to(device)
                tf_activity = batch['tf_activity'].to(device)
                chromatin_topics = batch['chromatin_topics'].to(device)
                coactivator_expr = batch['coactivator_expr'].to(device)
                labels = batch['label'].to(device)
                
                output = model(
                    node_features, pos, edge_index, edge_attr,
                    cell_type_idx, tf_activity, chromatin_topics, coactivator_expr
                )
                
                pred = output.activation_prob.item()
                label = labels.item()
                pred_class = 1 if pred > 0.5 else 0
                is_correct = pred_class == int(label)
                correct += int(is_correct)
                total += 1
                
                # Get case info
                case = val_cases_dataset.samples[i]
                tf_name = case.structure.tf_name
                cell_type = case.context.cell_type_name
                
                print(f"{tf_name:<10} {cell_type:<15} {pred:.4f}   {int(label)}        {'✓' if is_correct else '✗'}")
        
        print("-" * 55)
        print(f"Validation Case Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")


if __name__ == '__main__':
    main()
