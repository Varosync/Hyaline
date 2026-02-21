#!/usr/bin/env python
"""
Train TF-Modulator on TF-DNA and CryptoSite Data (GPU Optimized)
=================================================================

GPU-optimized training with:
- Mixed precision (AMP) for memory efficiency
- Gradient accumulation for effective larger batches
- Aggressive node subsampling for large structures
- Gradient checkpointing in model

Usage:
    python scripts/train_tf_modulator_gpu.py --epochs 50
"""

import os
import sys
import argparse
import json
import gc
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from hyaline.models.tf_modulator import TFModulator, TFModulatorConfig
from hyaline.features.geometric import GeometricFeatureExtractor, GeometricFeatures


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced classification.
    
    Fixed: Uses BCE directly with probabilities (model outputs sigmoid).
    Avoids double-sigmoid bug with BCE_with_logits.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp predictions for numerical stability
        pred = pred.float().clamp(1e-7, 1 - 1e-7)
        target = target.float()
        
        # Standard BCE (model already applies sigmoid)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # Focal weighting
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        return (focal_weight * bce).mean()


class OptimizedTFDataset(Dataset):
    """
    Memory-optimized dataset for TF-DNA structures.
    
    Key optimizations:
    - Cα-only representation (1 node per residue vs ~8 per residue)
    - Maximum node limit with intelligent subsampling
    - Lazy loading with caching
    """
    
    def __init__(
        self,
        pdb_files: List[Path],
        max_nodes: int = 500,  # Aggressive limit for GPU
        use_ca_only: bool = True,  # Use Cα only for massive reduction
        seed: int = 42,  # Fixed seed for deterministic labels
    ):
        self.pdb_files = pdb_files
        self.max_nodes = max_nodes
        self.use_ca_only = use_ca_only
        self.seed = seed
        self.extractor = GeometricFeatureExtractor()
        self.data_cache = []
        
        # Set seed for deterministic label generation
        np.random.seed(seed)
        self._load_all()
    
    def _load_all(self):
        """Load and process all PDB files."""
        for pdb_path in tqdm(self.pdb_files, desc="Loading"):
            try:
                data = self._load_single(pdb_path)
                if data is not None:
                    self.data_cache.append(data)
            except Exception as e:
                print(f"Skip {pdb_path.name}: {e}")
    
    def _load_single(self, pdb_path: Path) -> Optional[Dict]:
        """Load a single PDB with optimizations."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
        
        # Extract only Cα atoms for memory efficiency
        coords = []
        elements = []
        residue_types = []
        is_dna = []
        
        DNA_BASES = ['DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C']
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname().strip()
                    is_dna_res = res_name in DNA_BASES
                    
                    if self.use_ca_only:
                        # For proteins, use Cα; for DNA, use C1' or first C
                        if 'CA' in residue:
                            atom = residue['CA']
                        elif "C1'" in residue:
                            atom = residue["C1'"]
                        else:
                            # Fallback to first carbon
                            c_atoms = [a for a in residue if a.element == 'C']
                            if c_atoms:
                                atom = c_atoms[0]
                            else:
                                continue
                        
                        coords.append(atom.coord)
                        elements.append('C')
                        residue_types.append(res_name)
                        is_dna.append(is_dna_res)
                    else:
                        # All heavy atoms
                        for atom in residue:
                            if atom.element in ['C', 'N', 'O', 'S', 'P']:
                                coords.append(atom.coord)
                                elements.append(atom.element)
                                residue_types.append(res_name)
                                is_dna.append(is_dna_res)
        
        if len(coords) < 10:
            return None
        
        coords = np.array(coords, dtype=np.float32)
        is_dna = np.array(is_dna)
        
        # Subsample if needed
        if len(coords) > self.max_nodes:
            indices = np.random.choice(len(coords), self.max_nodes, replace=False)
            indices = np.sort(indices)
            coords = coords[indices]
            elements = [elements[i] for i in indices]
            residue_types = [residue_types[i] for i in indices]
            is_dna = is_dna[indices]
        
        # Extract features using simplified method
        features = self.extractor.extract_from_coords(
            coords, elements, residue_types, is_dna
        )
        
        # Generate labels (interface-based for TF-DNA, random for others)
        labels = self._generate_labels(features)
        
        return {
            'node_features': features.node_features.astype(np.float32),
            'edge_features': features.edge_features.astype(np.float32),
            'edge_index': features.edge_index.astype(np.int64),
            'pos': features.pos.astype(np.float32),
            'labels': labels.astype(np.float32),
            'pdb_id': pdb_path.stem,
            'n_nodes': len(features.pos),
        }
    
    def _generate_labels(self, features) -> np.ndarray:
        """Generate pocket labels."""
        n = len(features.pos)
        labels = np.zeros(n, dtype=np.float32)
        
        if features.is_dna.any():
            dna_coords = features.pos[features.is_dna]
            for i, coord in enumerate(features.pos):
                if not features.is_dna[i] and len(dna_coords) > 0:
                    dist = np.min(np.linalg.norm(dna_coords - coord, axis=1))
                    if dist < 8.0:  # Interface residues
                        labels[i] = 1.0
        else:
            # Random 10% for non-DNA structures
            pocket_idx = np.random.choice(n, max(1, n // 10), replace=False)
            labels[pocket_idx] = 1.0
        
        return labels
    
    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, idx):
        data = self.data_cache[idx]
        return {
            'node_features': torch.from_numpy(data['node_features']),
            'edge_features': torch.from_numpy(data['edge_features']),
            'edge_index': torch.from_numpy(data['edge_index']),
            'pos': torch.from_numpy(data['pos']),
            'labels': torch.from_numpy(data['labels']),
            'pdb_id': data['pdb_id'],
        }


def collate_fn(batch):
    return batch[0] if len(batch) == 1 else batch


def train_epoch(model, loader, optimizer, criterion, device, scaler, accum_steps=2):
    """Train with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(loader):
        node_f = batch['node_features'].to(device)
        edge_f = batch['edge_features'].to(device)
        edge_i = batch['edge_index'].to(device)
        pos = batch['pos'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            outputs = model(node_f, pos, edge_i, edge_f)
            pocket_loss = criterion(outputs['pocket_prob'], labels)
            drug_loss = criterion(outputs['druggability'], labels)
            loss = (pocket_loss + 0.5 * drug_loss) / accum_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        all_preds.append(outputs['pocket_prob'].detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        # Clear cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    # Handle final batch if not evenly divisible
    if (len(loader)) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except:
        roc_auc = 0.5
    
    return {'loss': total_loss / len(loader), 'roc_auc': roc_auc}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate with mixed precision."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        node_f = batch['node_features'].to(device)
        edge_f = batch['edge_features'].to(device)
        edge_i = batch['edge_index'].to(device)
        pos = batch['pos'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            outputs = model(node_f, pos, edge_i, edge_f)
            loss = criterion(outputs['pocket_prob'], labels)
        
        total_loss += loss.item()
        all_preds.append(outputs['pocket_prob'].cpu())
        all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except:
        roc_auc = 0.5
    
    return {'loss': total_loss / len(loader), 'roc_auc': roc_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', default=['data/tf_dna', 'data/cryptosite'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)  # Reduced for memory
    parser.add_argument('--num_layers', type=int, default=4)    # Reduced for memory
    parser.add_argument('--max_nodes', type=int, default=400)   # Aggressive limit
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints/tf_modulator_gpu')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    print(f"Using: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Discover PDB files
    pdb_files = []
    for d in args.data_dirs:
        p = Path(d)
        if p.exists():
            pdb_files.extend(p.glob("*.pdb"))
    pdb_files = sorted(set(pdb_files))
    print(f"\nFound {len(pdb_files)} PDB files")
    
    # Split
    n_train = int(0.8 * len(pdb_files))
    train_files, val_files = pdb_files[:n_train], pdb_files[n_train:]
    
    # Create datasets with Cα-only representation
    train_ds = OptimizedTFDataset(train_files, max_nodes=args.max_nodes, use_ca_only=True)
    val_ds = OptimizedTFDataset(val_files, max_nodes=args.max_nodes, use_ca_only=True)
    
    if len(train_ds) == 0:
        print("ERROR: No training data loaded!")
        return
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Model with reduced size for memory
    config = TFModulatorConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    )
    model = TFModulator(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params")
    
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_auc = 0
    print("\nTraining...")
    
    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:3d} | "
              f"Train: {train_m['loss']:.4f} / {train_m['roc_auc']:.3f} | "
              f"Val: {val_m['loss']:.4f} / {val_m['roc_auc']:.3f}")
        
        if val_m['roc_auc'] > best_auc:
            best_auc = val_m['roc_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': best_auc,
            }, save_dir / 'best_model.pt')
            print(f"  → Best: {best_auc:.4f}")
    
    print(f"\n✓ Training complete. Best AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
