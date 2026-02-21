#!/usr/bin/env python3
"""
Train Hyaline Neuromorphic TF-Modulator on CryptoSite benchmark.

Based on research recommendations:
- SuperSpike surrogate (β=50, learnable)
- Linear threshold annealing (1.0 → 0.3)
- Activity regularization (target 10% spikes)
- AdamW optimizer with cosine annealing
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import wandb
from pathlib import Path
import argparse

from models import HyalineModel
from training import (
    SurrogateType,
    SurrogateConfig,
    BCEWithSynchronizationLoss,
    NeuromorphicTrainer,
    ThresholdScheduler
)
from benchmarks import CryptoSiteDataset, evaluate_predictions
from data import TrajectoryDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hyaline on CryptoSite')
    
    # Data
    parser.add_argument('--data-root', type=str, default='data/cryptosite',
                        help='CryptoSite data directory')
    parser.add_argument('--cache-dir', type=str, default='data/cache',
                        help='Cache directory for processed data')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of spiking EGNN layers')
    parser.add_argument('--timesteps', type=int, default=10,
                        help='Number of spike timesteps')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (small for temporal graphs)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Surrogate gradient
    parser.add_argument('--surrogate', type=str, default='superspike',
                        choices=['fast_sigmoid', 'superspike', 'exponential'],
                        help='Surrogate gradient type')
    parser.add_argument('--beta', type=float, default=50.0,
                        help='Surrogate gradient steepness')
    parser.add_argument('--learnable-beta', action='store_true',
                        help='Make beta learnable')
    
    # Threshold annealing
    parser.add_argument('--initial-threshold', type=float, default=1.0,
                        help='Initial spike threshold')
    parser.add_argument('--final-threshold', type=float, default=0.3,
                        help='Final spike threshold')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Epochs before threshold annealing starts')
    
    # Regularization
    parser.add_argument('--activity-reg', type=float, default=0.01,
                        help='Activity regularization weight')
    parser.add_argument('--target-sparsity', type=float, default=0.1,
                        help='Target spike rate (10% recommended)')
    parser.add_argument('--sync-weight', type=float, default=0.1,
                        help='Synchronization loss weight')
    
    # Logging
    parser.add_argument('--wandb-project', type=str, default='hyaline-cryptosite',
                        help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Steps between logging')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='Epochs between evaluation')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Epochs between checkpoints')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=f"hyaline_{args.surrogate}_beta{args.beta}"
    )
    
    # Load CryptoSite dataset
    print("Loading CryptoSite dataset...")
    dataset = CryptoSiteDataset(
        root=args.data_root,
        cache_dir=args.cache_dir,
        download=True
    )
    
    # Split into train/val/test
    # CryptoSite has 14 test proteins - use 10 for train, 2 for val, 2 for test
    train_size = 10
    val_size = 2
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("Creating Hyaline model...")
    
    # Map surrogate type string to enum
    surrogate_map = {
        'fast_sigmoid': SurrogateType.FAST_SIGMOID,
        'superspike': SurrogateType.SUPERSPIKE,
        'exponential': SurrogateType.EXPONENTIAL
    }
    
    surrogate_config = SurrogateConfig(
        surrogate_type=surrogate_map[args.surrogate],
        beta=args.beta,
        learnable=args.learnable_beta
    )
    
    model = HyalineModel(
        node_dim=dataset[0].x.shape[1],  # Input features
        edge_dim=dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        timesteps=args.timesteps,
        surrogate_config=surrogate_config
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = BCEWithSynchronizationLoss(
        sync_weight=args.sync_weight,
        activity_weight=args.activity_reg,
        target_sparsity=args.target_sparsity
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler (cosine annealing)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Create threshold scheduler
    threshold_scheduler = ThresholdScheduler(
        initial_threshold=args.initial_threshold,
        final_threshold=args.final_threshold,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        schedule_type='linear'
    )
    
    # Create trainer
    trainer = NeuromorphicTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        threshold_scheduler=threshold_scheduler,
        lr_scheduler=lr_scheduler,
        checkpoint_dir=Path(args.checkpoint_dir),
        log_interval=args.log_interval,
        use_wandb=True
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    best_val_auc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        if (epoch + 1) % args.eval_interval == 0:
            val_metrics = trainer.validate(val_loader, epoch)
            
            # Track best model
            if val_metrics['auc_roc'] > best_val_auc:
                best_val_auc = val_metrics['auc_roc']
                trainer.save_checkpoint(
                    epoch,
                    val_metrics,
                    filename='best_model.pt'
                )
                print(f"✓ New best model! AUC-ROC: {best_val_auc:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            trainer.save_checkpoint(
                epoch,
                train_metrics,
                filename=f'checkpoint_epoch_{epoch+1}.pt'
            )
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final evaluation on test set...")
    
    # Load best model
    best_checkpoint = Path(args.checkpoint_dir) / 'best_model.pt'
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_metrics = trainer.validate(test_loader, epoch=-1, split='test')
    
    print("\nTest Results:")
    print(f"  AUC-ROC:  {test_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:   {test_metrics['auc_pr']:.4f}")
    print(f"  Recall@5: {test_metrics.get('recall@5', 0):.4f}")
    print(f"  MCC:      {test_metrics.get('mcc', 0):.4f}")
    
    wandb.log({
        'test/auc_roc': test_metrics['auc_roc'],
        'test/auc_pr': test_metrics['auc_pr'],
        'test/recall@5': test_metrics.get('recall@5', 0),
        'test/mcc': test_metrics.get('mcc', 0)
    })
    
    wandb.finish()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
