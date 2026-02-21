"""
Evaluation Metrics for Cryptic Pocket Prediction
=================================================

Standard metrics for benchmarking:
- Residue-level AUC-ROC
- Pocket-level recall@k
- Intersection over Union (IoU)
- Matthews Correlation Coefficient (MCC)
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)


def residue_auc_roc(
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute residue-level AUC-ROC.
    
    Args:
        predictions: Predicted pocket probabilities [N]
        labels: Binary labels [N]
        
    Returns:
        AUC-ROC score
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # Undefined if only one class
    
    return roc_auc_score(labels, predictions)


def residue_auc_pr(
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute residue-level AUC-PR (Average Precision).
    
    More informative for imbalanced datasets (rare pockets).
    
    Args:
        predictions: Predicted pocket probabilities [N]
        labels: Binary labels [N]
        
    Returns:
        Average precision score
    """
    if labels.sum() == 0:
        return 0.0
    
    return average_precision_score(labels, predictions)


def pocket_recall_at_k(
    predictions: np.ndarray,
    labels: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute pocket recall at top-k predictions.
    
    What fraction of true pocket residues are in top-k predictions?
    
    Args:
        predictions: Predicted scores [N]
        labels: Binary labels [N]
        k: Number of top predictions to consider
        
    Returns:
        Recall@k score
    """
    n_pocket = int(labels.sum())
    if n_pocket == 0:
        return 0.0
    
    # Get top-k predicted indices
    top_k_idx = np.argsort(predictions)[-k:]
    
    # Count how many are true positives
    true_positives = labels[top_k_idx].sum()
    
    return true_positives / n_pocket


def precision_at_k(
    predictions: np.ndarray,
    labels: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute precision at top-k predictions.
    
    What fraction of top-k predictions are true pocket residues?
    
    Args:
        predictions: Predicted scores [N]
        labels: Binary labels [N]
        k: Number of top predictions
        
    Returns:
        Precision@k score
    """
    top_k_idx = np.argsort(predictions)[-k:]
    true_positives = labels[top_k_idx].sum()
    
    return true_positives / k


def pocket_iou(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute Intersection over Union for pocket predictions.
    
    IoU = |pred ∩ true| / |pred ∪ true|
    
    Args:
        predictions: Predicted probabilities [N]
        labels: Binary labels [N]
        threshold: Threshold for binarizing predictions
        
    Returns:
        IoU score
    """
    pred_binary = (predictions > threshold).astype(float)
    
    intersection = (pred_binary * labels).sum()
    union = ((pred_binary + labels) > 0).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def matthews_correlation(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute Matthews Correlation Coefficient.
    
    MCC is useful for imbalanced binary classification.
    Range: [-1, 1] where 1 is perfect prediction.
    
    Args:
        predictions: Predicted probabilities [N]
        labels: Binary labels [N]
        threshold: Threshold for binarizing
        
    Returns:
        MCC score
    """
    pred_binary = (predictions > threshold).astype(int)
    labels_int = labels.astype(int)
    
    return matthews_corrcoef(labels_int, pred_binary)


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted probabilities [N]
        labels: Binary labels [N]
        threshold: Binarization threshold
        k_values: K values for recall@k
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # AUC scores
    metrics['auc_roc'] = residue_auc_roc(predictions, labels)
    metrics['auc_pr'] = residue_auc_pr(predictions, labels)
    
    # Recall and precision at k
    for k in k_values:
        metrics[f'recall@{k}'] = pocket_recall_at_k(predictions, labels, k)
        metrics[f'precision@{k}'] = precision_at_k(predictions, labels, k)
    
    # IoU and MCC
    metrics['iou'] = pocket_iou(predictions, labels, threshold)
    metrics['mcc'] = matthews_correlation(predictions, labels, threshold)
    
    # Binary metrics at threshold
    pred_binary = (predictions > threshold).astype(int)
    labels_int = labels.astype(int)
    
    if len(np.unique(pred_binary)) == 2 and len(np.unique(labels_int)) == 2:
        tn, fp, fn, tp = confusion_matrix(labels_int, pred_binary).ravel()
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    else:
        metrics['sensitivity'] = 0.0
        metrics['specificity'] = 0.0
        metrics['precision'] = 0.0
        metrics['f1'] = 0.0
    
    return metrics


def evaluate_dataset(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on entire dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation set
        device: torch device
        threshold: Binarization threshold
        
    Returns:
        Aggregated metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            output = model(batch)
            preds = output['pocket_probs'].cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Concatenate all
    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    return compute_all_metrics(predictions, labels, threshold)


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print metrics."""
    print(f"\n{title}")
    print("=" * 50)
    
    # Group metrics
    auc_metrics = {k: v for k, v in metrics.items() if 'auc' in k}
    at_k_metrics = {k: v for k, v in metrics.items() if '@' in k}
    other_metrics = {k: v for k, v in metrics.items() if 'auc' not in k and '@' not in k}
    
    print("\nAUC Metrics:")
    for k, v in auc_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    
    print("\nRanking Metrics:")
    for k, v in sorted(at_k_metrics.items()):
        print(f"  {k:15s}: {v:.4f}")
    
    print("\nBinary Metrics:")
    for k, v in other_metrics.items():
        print(f"  {k:15s}: {v:.4f}")


if __name__ == "__main__":
    print("Testing Evaluation Metrics")
    print("=" * 50)
    
    # Generate test predictions
    N = 100
    np.random.seed(42)
    
    # Simulate predictions (higher for true positives)
    labels = (np.random.rand(N) > 0.9).astype(float)  # 10% positive
    predictions = np.random.rand(N) * 0.5  # Base predictions
    predictions[labels > 0] += 0.4  # Boost true positives
    predictions = np.clip(predictions, 0, 1)
    
    print(f"\nDataset: {N} residues, {int(labels.sum())} pocket")
    
    # Compute all metrics
    metrics = compute_all_metrics(predictions, labels)
    print_metrics(metrics, "Test Metrics")
    
    print("\n" + "=" * 50)
    print("✓ All evaluation metrics working!")
