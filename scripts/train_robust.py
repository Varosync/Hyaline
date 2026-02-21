#!/usr/bin/env python
"""
Robust TF Activation Training
=============================
Balanced data, proper loss, early stopping, reproducible.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import random

from hyaline.models.tf_activation_model import TFActivationModel, TFActivationConfig
from hyaline.loaders.pdb_loader import load_tf_structures
from hyaline.loaders.tf_activation_data import SCENICContext, VALIDATION_CASES, EXPECTED_LEVELS, TF_LINEAGE


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BalancedTFDataset(Dataset):
    """Dataset with balanced positive/negative samples."""
    
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'nf': torch.from_numpy(s['nf']).float(),
            'pos': torch.from_numpy(s['pos']).float(),
            'ei': torch.from_numpy(s['ei']).long(),
            'ea': torch.from_numpy(s['ea']).float(),
            'ctx': s['ctx'],
            'label': s['label'],
            'tf': s['tf'],
            'ct': s['ct'],
        }


def create_balanced_dataset(structures, cell_types, samples_per_class=50):
    """Create balanced positive and negative samples."""
    positive_samples = []
    negative_samples = []
    
    for tf_name, struct_list in structures.items():
        tf_lineage = TF_LINEAGE.get(tf_name, None)
        
        for struct in struct_list:
            for cell_type in cell_types:
                # Determine if this is a positive (matching) or negative case
                is_positive = tf_lineage and tf_lineage in cell_type.lower()
                
                ctx = SCENICContext.from_cell_type(cell_type, tf_name, noise_scale=0.1)
                
                sample = {
                    'nf': struct.node_features.copy(),
                    'pos': struct.pos.copy(),
                    'ei': struct.edge_index.copy(),
                    'ea': struct.edge_attr.copy(),
                    'ctx': ctx,
                    'label': 1.0 if is_positive else 0.0,
                    'tf': tf_name,
                    'ct': cell_type,
                }
                
                if is_positive:
                    positive_samples.append(sample)
                else:
                    negative_samples.append(sample)
    
    # Balance classes
    n_pos = min(len(positive_samples), samples_per_class)
    n_neg = min(len(negative_samples), samples_per_class)
    n_samples = min(n_pos, n_neg)
    
    if n_samples == 0:
        # If no natural positives, create synthetic ones
        for tf_name, struct_list in structures.items():
            tf_lineage = TF_LINEAGE.get(tf_name, None)
            if not tf_lineage:
                continue
            for struct in struct_list:
                # Create matching context
                ctx = SCENICContext.from_cell_type(tf_lineage, tf_name, noise_scale=0.1)
                ctx.tf_activity[0] = 0.9  # Boost the TF's activity
                sample = {
                    'nf': struct.node_features.copy(),
                    'pos': struct.pos.copy(),
                    'ei': struct.edge_index.copy(),
                    'ea': struct.edge_attr.copy(),
                    'ctx': ctx,
                    'label': 1.0,
                    'tf': tf_name,
                    'ct': tf_lineage,
                }
                positive_samples.append(sample)
    
    np.random.shuffle(positive_samples)
    np.random.shuffle(negative_samples)
    
    balanced = positive_samples[:samples_per_class] + negative_samples[:samples_per_class]
    np.random.shuffle(balanced)
    
    print(f"Created {len(balanced)} samples ({samples_per_class} pos, {samples_per_class} neg)")
    return balanced


def collate_fn(batch):
    """Custom collate for single samples."""
    return batch[0]


class FocalBCELoss(nn.Module):
    """Focal loss for imbalanced classification."""
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = pred.clamp(1e-7, 1 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


def train_step(model, batch, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    ctx = batch['ctx']
    out = model(
        batch['nf'].to(device),
        batch['pos'].to(device),
        batch['ei'].to(device),
        batch['ea'].to(device),
        torch.tensor([ctx.cell_type_idx]).long().to(device),
        torch.from_numpy(ctx.tf_activity).float().unsqueeze(0).to(device),
        torch.from_numpy(ctx.chromatin_topics).float().unsqueeze(0).to(device),
        torch.from_numpy(ctx.coactivator_expr).float().unsqueeze(0).to(device),
    )
    
    label = torch.tensor([batch['label']]).float().to(device)
    loss = criterion(out.activation_prob, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item(), out.activation_prob.item(), batch['label']


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, labels = [], [], []
    
    for batch in loader:
        ctx = batch['ctx']
        out = model(
            batch['nf'].to(device),
            batch['pos'].to(device),
            batch['ei'].to(device),
            batch['ea'].to(device),
            torch.tensor([ctx.cell_type_idx]).long().to(device),
            torch.from_numpy(ctx.tf_activity).float().unsqueeze(0).to(device),
            torch.from_numpy(ctx.chromatin_topics).float().unsqueeze(0).to(device),
            torch.from_numpy(ctx.coactivator_expr).float().unsqueeze(0).to(device),
        )
        
        label = torch.tensor([batch['label']]).float().to(device)
        loss = criterion(out.activation_prob, label)
        
        losses.append(loss.item())
        preds.append(out.activation_prob.item())
        labels.append(batch['label'])
    
    try:
        auc = roc_auc_score(labels, preds)
    except:
        auc = 0.5
    
    acc = accuracy_score(labels, [1 if p > 0.5 else 0 for p in preds])
    return np.mean(losses), auc, acc


@torch.no_grad()
def evaluate_validation_cases(model, structures, device):
    """Evaluate on biological validation cases."""
    model.eval()
    correct = 0
    results = []
    
    for tf_name, cell_type, expected in VALIDATION_CASES:
        if tf_name not in structures:
            continue
        
        struct = structures[tf_name][0]
        ctx = SCENICContext.from_cell_type(cell_type, tf_name, noise_scale=0.0)
        
        out = model(
            torch.from_numpy(struct.node_features).float().to(device),
            torch.from_numpy(struct.pos).float().to(device),
            torch.from_numpy(struct.edge_index).long().to(device),
            torch.from_numpy(struct.edge_attr).float().to(device),
            torch.tensor([ctx.cell_type_idx]).long().to(device),
            torch.from_numpy(ctx.tf_activity).float().unsqueeze(0).to(device),
            torch.from_numpy(ctx.chromatin_topics).float().unsqueeze(0).to(device),
            torch.from_numpy(ctx.coactivator_expr).float().unsqueeze(0).to(device),
        )
        
        pred = out.activation_prob.item()
        expected_prob = EXPECTED_LEVELS.get(expected, 0.5)
        is_correct = (pred > 0.5) == (expected_prob > 0.5)
        correct += int(is_correct)
        
        results.append({
            'tf': tf_name,
            'cell': cell_type,
            'expected': expected,
            'pred': pred,
            'correct': is_correct,
        })
    
    return correct / len(results) if results else 0, results


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load structures
    print("\nLoading TF-DNA structures...")
    structures = load_tf_structures(Path('data/tf_dna_structures'))
    
    # Cell types including lineage-specific ones
    cell_types = [
        'melanocyte', 'hepatocyte', 'cardiomyocyte',  # Lineage matches
        'fibroblast', 'neuron', 't_cell', 'epithelial', 'esc',  # Non-matches
    ]
    
    # Create balanced dataset
    all_samples = create_balanced_dataset(structures, cell_types, samples_per_class=100)
    
    # Split
    n_train = int(0.8 * len(all_samples))
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]
    
    train_dataset = BalancedTFDataset(train_samples)
    val_dataset = BalancedTFDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Model
    config = TFActivationConfig(
        hidden_dim=96,
        num_egnn_layers=3,
        dropout=0.15,
        base_threshold=1.0,
    )
    model = TFActivationModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = FocalBCELoss(alpha=0.5, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop with early stopping
    save_dir = Path('checkpoints/tf_robust')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0
    best_bio_acc = 0
    patience = 7
    no_improve = 0
    
    print("\nTraining...")
    print("-" * 70)
    
    for epoch in range(1, 51):
        # Train
        train_losses, train_preds, train_labels = [], [], []
        for batch in train_loader:
            loss, pred, label = train_step(model, batch, optimizer, criterion, device)
            train_losses.append(loss)
            train_preds.append(pred)
            train_labels.append(label)
        
        try:
            train_auc = roc_auc_score(train_labels, train_preds)
        except:
            train_auc = 0.5
        
        # Validate
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        # Biological validation
        bio_acc, bio_results = evaluate_validation_cases(model, structures, device)
        
        print(f"Epoch {epoch:2d} | Train AUC: {train_auc:.4f} | "
              f"Val AUC: {val_auc:.4f} Acc: {val_acc:.4f} | Bio Acc: {bio_acc:.1%}")
        
        # Save best model (prioritize biological accuracy)
        if bio_acc > best_bio_acc or (bio_acc == best_bio_acc and val_acc > best_val_acc):
            best_bio_acc = bio_acc
            best_val_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'config': config.__dict__,
                'epoch': epoch,
                'bio_acc': bio_acc,
                'val_acc': val_acc,
            }, save_dir / 'best.pt')
            no_improve = 0
            print(f"  -> Saved (Bio: {bio_acc:.1%}, Val: {val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Final evaluation
    ckpt = torch.load(save_dir / 'best.pt')
    model.load_state_dict(ckpt['model'])
    
    print("\n" + "=" * 70)
    print("FINAL BIOLOGICAL VALIDATION")
    print("=" * 70)
    print(f"{'TF':<10} {'Cell Type':<15} {'Expected':<12} {'Pred':<8} {'Result'}")
    print("-" * 70)
    
    bio_acc, results = evaluate_validation_cases(model, structures, device)
    for r in results:
        symbol = "✓" if r['correct'] else "✗"
        print(f"{r['tf']:<10} {r['cell']:<15} {r['expected']:<12} {r['pred']:.4f}   {symbol}")
    
    print("-" * 70)
    print(f"Biological Accuracy: {bio_acc:.1%}")
    print(f"Best Epoch: {ckpt['epoch']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
