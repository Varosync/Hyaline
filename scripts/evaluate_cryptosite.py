#!/usr/bin/env python3
"""
CryptoSite Benchmark Evaluation
================================

Evaluates cryptic pocket detection on CryptoSite benchmark.
"""

import sys
sys.path.insert(0, '/home/ec2-user/Jinja')

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# Add benchmarks to path
from benchmarks.cryptosite_loader import CryptoSiteDataset, CRYPTOSITE_PROTEINS


def evaluate_cryptosite(data_dir: str = "data/cryptosite"):
    """Evaluate on CryptoSite benchmark."""
    print("=" * 60)
    print("CryptoSite Benchmark Evaluation")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading CryptoSite structures...")
    dataset = CryptoSiteDataset(
        data_dir=data_dir,
        split='test',
        cutoff=10.0,
        auto_download=True
    )
    
    print(f"Loaded {len(dataset)} proteins")
    
    # Analyze each protein
    results = []
    
    for idx in range(len(dataset)):
        data = dataset[idx]
        info = dataset.get_protein_info(idx)
        
        pdb_id = info['apo_pdb']
        n_residues = data.x.shape[0]
        n_cryptic = info['n_cryptic']
        cryptic_ratio = n_cryptic / n_residues if n_residues > 0 else 0
        
        results.append({
            'pdb': pdb_id,
            'chain': info['chain'],
            'n_residues': n_residues,
            'n_cryptic': n_cryptic,
            'cryptic_ratio': cryptic_ratio,
            'labels': data.y.numpy() if hasattr(data.y, 'numpy') else data.y,
        })
        
        print(f"  {pdb_id}:{info['chain']} - {n_residues} residues, {n_cryptic} cryptic ({cryptic_ratio:.1%})")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    total_residues = sum(r['n_residues'] for r in results)
    total_cryptic = sum(r['n_cryptic'] for r in results)
    avg_cryptic = np.mean([r['n_cryptic'] for r in results])
    
    print(f"  Total proteins: {len(results)}")
    print(f"  Total residues: {total_residues}")
    print(f"  Total cryptic residues: {total_cryptic}")
    print(f"  Avg cryptic per protein: {avg_cryptic:.1f}")
    print(f"  Overall cryptic ratio: {total_cryptic/total_residues:.2%}")
    
    # Baseline: Random prediction
    print("\n" + "=" * 60)
    print("Baseline Evaluation (Random Predictor)")
    print("=" * 60)
    
    all_labels = []
    all_preds_random = []
    
    for r in results:
        labels = r['labels']
        if isinstance(labels, np.ndarray) and len(labels) > 0:
            all_labels.extend(labels.tolist())
            # Random predictions
            all_preds_random.extend(np.random.rand(len(labels)).tolist())
    
    if len(set(all_labels)) > 1:  # Need both classes
        auc_random = roc_auc_score(all_labels, all_preds_random)
        ap_random = average_precision_score(all_labels, all_preds_random)
        print(f"  Random AUC-ROC: {auc_random:.3f}")
        print(f"  Random AP: {ap_random:.3f}")
    else:
        print("  (Not enough positive/negative samples for metrics)")
    
    print("\n" + "=" * 60)
    print("✓ CryptoSite evaluation complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = evaluate_cryptosite()
