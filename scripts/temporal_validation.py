#!/usr/bin/env python3
"""
Phase 1: Temporal Split and Baseline Comparisons
=================================================

This script:
1. Fetches PDB release dates from RCSB API
2. Creates temporal train/test split (train <2023, test >=2023)
3. Trains HyalineV2 on pre-2023 data
4. Evaluates on post-2023 data (truly blind test)
5. Runs baseline comparisons
"""
import sys
sys.path.insert(0, '/home/ec2-user/Jinja')

import torch
import torch.nn as nn
import numpy as np
import json
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
import h5py

from hyaline import HyalineV2, load_dataset_with_motifs

# Production config (from checkpoint)
V2D_CONFIG = {
    'node_input_dim': 1536,
    'edge_input_dim': 3,
    'hidden_dim': 320,
    'num_layers': 5,
    'num_heads': 4,
    'num_rbf': 96,
    'cutoff': 10.0,
    'dropout': 0.15,
    'update_coords': True,
    'use_motif_bias': True,
    'use_multiscale': False
}


def fetch_pdb_release_dates(pdb_ids: list) -> dict:
    """Fetch release dates from RCSB PDB API."""
    print(f"Fetching release dates for {len(pdb_ids)} structures...")
    
    dates = {}
    batch_size = 50  # API limit
    
    for i in tqdm(range(0, len(pdb_ids), batch_size)):
        batch = pdb_ids[i:i+batch_size]
        
        # RCSB GraphQL API
        query = """
        query ($ids: [String!]!) {
            entries(entry_ids: $ids) {
                rcsb_id
                rcsb_accession_info {
                    initial_release_date
                }
            }
        }
        """
        
        try:
            response = requests.post(
                "https://data.rcsb.org/graphql",
                json={"query": query, "variables": {"ids": batch}},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']['entries']:
                    for entry in data['data']['entries']:
                        if entry and entry['rcsb_accession_info']:
                            pdb_id = entry['rcsb_id']
                            date_str = entry['rcsb_accession_info']['initial_release_date']
                            dates[pdb_id] = date_str[:10]  # YYYY-MM-DD
        except Exception as e:
            print(f"  Error fetching batch: {e}")
            continue
    
    print(f"  Retrieved dates for {len(dates)} structures")
    return dates


def create_temporal_split(data_list, labels, pdb_ids, release_dates, cutoff_date="2023-01-01"):
    """Split data by PDB release date."""
    train_idx = []
    test_idx = []
    
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
    missing = 0
    
    for i, pdb_id in enumerate(pdb_ids):
        if pdb_id in release_dates:
            try:
                release = datetime.strptime(release_dates[pdb_id], "%Y-%m-%d")
                if release < cutoff:
                    train_idx.append(i)
                else:
                    test_idx.append(i)
            except:
                train_idx.append(i)  # Default to train if date parsing fails
        else:
            missing += 1
            train_idx.append(i)  # Default to train if no date
    
    print(f"Temporal split (cutoff={cutoff_date}):")
    print(f"  Train: {len(train_idx)} (pre-{cutoff_date})")
    print(f"  Test:  {len(test_idx)} (post-{cutoff_date})")
    print(f"  Missing dates: {missing}")
    
    return train_idx, test_idx


def train_hyaline_v2(train_data, val_data, config, device='cuda', epochs=30):
    """Train HyalineV2 model."""
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    
    model = HyalineV2(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auroc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs.flatten())
                truth.extend(batch.y.cpu().numpy())
        
        auroc = roc_auc_score(truth, preds) if len(set(truth)) > 1 else 0
        
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: AuROC = {auroc:.4f}")
    
    model.load_state_dict(best_state)
    return model, best_auroc


def evaluate_model(model, test_data, device='cuda'):
    """Evaluate model on test set."""
    loader = DataLoader(test_data, batch_size=4, shuffle=False)
    model.eval()
    
    preds, truth = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.flatten())
            truth.extend(batch.y.cpu().numpy())
    
    preds = np.array(preds)
    truth = np.array(truth)
    
    metrics = {
        'auroc': roc_auc_score(truth, preds) if len(set(truth)) > 1 else 0,
        'accuracy': accuracy_score(truth, preds > 0.5),
        'precision': precision_score(truth, preds > 0.5, zero_division=0),
        'recall': recall_score(truth, preds > 0.5, zero_division=0),
        'f1': f1_score(truth, preds > 0.5, zero_division=0),
        'n_samples': len(truth),
        'n_active': int(truth.sum()),
        'n_inactive': int(len(truth) - truth.sum())
    }
    
    return metrics, preds, truth


# ============ BASELINE MODELS ============

def baseline_random(train_labels, test_labels):
    """Random baseline (predict majority class)."""
    majority = 1 if train_labels.sum() > len(train_labels) / 2 else 0
    preds = np.full(len(test_labels), majority, dtype=float)
    
    # Add small noise for ROC calculation
    preds = preds + np.random.normal(0, 0.01, len(preds))
    preds = np.clip(preds, 0, 1)
    
    return roc_auc_score(test_labels, preds) if len(set(test_labels)) > 1 else 0.5


def baseline_esm_logistic(train_embeddings, train_labels, test_embeddings, test_labels):
    """ESM3 embeddings + Logistic Regression (sequence only, no structure)."""
    clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict_proba(test_embeddings)[:, 1]
    
    return roc_auc_score(test_labels, preds) if len(set(test_labels)) > 1 else 0.5


def baseline_esm_mlp(train_embeddings, train_labels, test_embeddings, test_labels, device='cuda'):
    """ESM3 embeddings + 2-layer MLP (sequence only, no structure)."""
    
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=1536):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    # Convert to tensors
    X_train = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
    X_test = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
    
    model = SimpleMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_test)).cpu().numpy()
    
    return roc_auc_score(test_labels, preds) if len(set(test_labels)) > 1 else 0.5


def run_phase1():
    """Run complete Phase 1 analysis."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("PHASE 1: TEMPORAL SPLIT + BASELINE COMPARISONS")
    print("=" * 70)
    print(f"Device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    data_list, labels, families, sequences = load_dataset_with_motifs()
    labels = np.array(labels)
    
    # Get PDB IDs from H5 file
    with h5py.File('data/gpcrdb_all/esm3_receptor_only.h5', 'r') as f:
        pdb_ids = [p.decode() if isinstance(p, bytes) else p for p in f['pdb_ids'][:]]
        embeddings = f['embeddings'][:]
    
    # Fetch release dates
    cache_path = Path('data/gpcrdb_all/pdb_release_dates.json')
    if cache_path.exists():
        print("Loading cached release dates...")
        with open(cache_path) as f:
            release_dates = json.load(f)
    else:
        release_dates = fetch_pdb_release_dates(pdb_ids)
        with open(cache_path, 'w') as f:
            json.dump(release_dates, f, indent=2)
    
    # Create temporal split
    train_idx, test_idx = create_temporal_split(
        data_list, labels, pdb_ids, release_dates, cutoff_date="2023-01-01"
    )
    
    if len(test_idx) < 10:
        print("\nWARNING: Very few test samples. Trying 2022-01-01 cutoff...")
        train_idx, test_idx = create_temporal_split(
            data_list, labels, pdb_ids, release_dates, cutoff_date="2022-01-01"
        )
    
    # Prepare data
    train_data = [data_list[i] for i in train_idx]
    test_data = [data_list[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    # Mean-pooled embeddings for baselines
    train_embeddings = np.array([embeddings[i].mean(axis=0) for i in train_idx])
    test_embeddings = np.array([embeddings[i].mean(axis=0) for i in test_idx])
    
    print(f"\nTrain: {len(train_data)} samples ({train_labels.sum():.0f} active)")
    print(f"Test:  {len(test_data)} samples ({test_labels.sum():.0f} active)")
    
    # ============ RUN BASELINES ============
    print("\n" + "=" * 70)
    print("BASELINE COMPARISONS")
    print("=" * 70)
    
    results = {}
    
    # 1. Random baseline
    print("\n1. Random (majority class)...")
    results['Random'] = baseline_random(train_labels, test_labels)
    print(f"   AuROC: {results['Random']:.4f}")
    
    # 2. ESM3 + Logistic Regression
    print("\n2. ESM3 + Logistic Regression (sequence only)...")
    results['ESM3+LogReg'] = baseline_esm_logistic(
        train_embeddings, train_labels, test_embeddings, test_labels
    )
    print(f"   AuROC: {results['ESM3+LogReg']:.4f}")
    
    # 3. ESM3 + MLP
    print("\n3. ESM3 + MLP (sequence only)...")
    results['ESM3+MLP'] = baseline_esm_mlp(
        train_embeddings, train_labels, test_embeddings, test_labels, device
    )
    print(f"   AuROC: {results['ESM3+MLP']:.4f}")
    
    # ============ TRAIN HYALINE V2 ============
    print("\n" + "=" * 70)
    print("HYALINE V2-D (TEMPORAL TRAINING)")
    print("=" * 70)
    
    # Use some train data for validation
    n_val = min(100, len(train_data) // 5)
    val_data = train_data[-n_val:]
    train_data_final = train_data[:-n_val]
    
    print(f"\nTraining on {len(train_data_final)} samples...")
    model, val_auroc = train_hyaline_v2(train_data_final, val_data, V2D_CONFIG, device, epochs=30)
    print(f"Validation AuROC: {val_auroc:.4f}")
    
    print("\nEvaluating on held-out temporal test set...")
    metrics, preds, truth = evaluate_model(model, test_data, device)
    results['HyalineV2'] = metrics['auroc']
    
    print(f"\n{'='*50}")
    print("TEMPORAL TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  AuROC:     {metrics['auroc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Samples:   {metrics['n_samples']} ({metrics['n_active']} active)")
    
    # ============ COMPARISON TABLE ============
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'AuROC':<10} {'vs Hyaline':<15}")
    print("-" * 50)
    
    hyaline_auroc = results['HyalineV2']
    for model_name, auroc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        diff = auroc - hyaline_auroc if model_name != 'HyalineV2' else 0
        diff_str = f"{diff:+.4f}" if model_name != 'HyalineV2' else "---"
        marker = "★" if model_name == 'HyalineV2' else ""
        print(f"{model_name:<25} {auroc:.4f}     {diff_str:<15} {marker}")
    
    # Save results
    output = {
        'temporal_split': {
            'cutoff_date': '2023-01-01',
            'n_train': len(train_data),
            'n_test': len(test_data)
        },
        'baseline_results': results,
        'hyaline_metrics': metrics
    }
    
    output_path = Path('results/temporal_validation.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\nResults saved to: {output_path}")
    
    return results, metrics


if __name__ == '__main__':
    run_phase1()
