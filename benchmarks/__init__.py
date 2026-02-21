"""
Benchmark Module for Hyaline Neuromorphic
=========================================

Provides benchmarking infrastructure for cryptic pocket prediction.
"""

from .cryptosite_loader import (
    CryptoSiteDataset,
    CryptoBenchDataset,
    CRYPTOSITE_PROTEINS,
)

from .evaluation import (
    compute_all_metrics,
    evaluate_dataset,
    residue_auc_roc,
    residue_auc_pr,
    pocket_recall_at_k,
    precision_at_k,
    pocket_iou,
    matthews_correlation,
    print_metrics,
)

__all__ = [
    # Datasets
    'CryptoSiteDataset',
    'CryptoBenchDataset',
    'CRYPTOSITE_PROTEINS',
    
    # Evaluation
    'compute_all_metrics',
    'evaluate_dataset',
    'residue_auc_roc',
    'residue_auc_pr',
    'pocket_recall_at_k',
    'precision_at_k',
    'pocket_iou',
    'matthews_correlation',
    'print_metrics',
]
