#!/usr/bin/env python3
"""
Train Conformational Prior
===========================

Trains the sequence-only model that predicts P(DFG-out) from the
85-residue KLIFS pocket sequence. Used as a Bayesian prior for kinases
without crystal structures in both DFG conformations.

Kinase-identity holdout: no individual kinase appears in both train and val.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyaline.models.conformational_prior import (
    ConformationalPrior, ConformationalPriorConfig, encode_pocket
)
from hyaline.models.kinase_binding import KLIFSLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PocketSequenceDataset(Dataset):
    """Dataset of pocket sequences with DFG labels."""

    def __init__(self, structure_ids, sequences, dfg_labels):
        self.structure_ids = structure_ids
        self.sequences = sequences
        # Encode all pocket sequences
        self.encoded = torch.stack([
            torch.tensor(encode_pocket(seq), dtype=torch.long)
            for seq in sequences
        ])
        # DFG-out = 1, DFG-in = 0
        self.labels = torch.tensor(dfg_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]


def kinase_identity_split(df, seed=42):
    """Split by kinase identity: no kinase appears in both train and val."""
    rng = np.random.RandomState(seed)

    # Group structures by kinase_name
    kinase_groups = defaultdict(list)
    for idx, row in df.iterrows():
        kinase_groups[row["kinase_name"]].append(idx)

    kinases = list(kinase_groups.keys())
    rng.shuffle(kinases)

    # 80/20 split by kinase
    n_val = max(1, int(0.2 * len(kinases)))
    val_kinases = set(kinases[:n_val])
    train_kinases = set(kinases[n_val:])

    train_idx = [i for k in train_kinases for i in kinase_groups[k]]
    val_idx = [i for k in val_kinases for i in kinase_groups[k]]

    logger.info(f"Kinase-identity split: {len(train_kinases)} train kinases, {len(val_kinases)} val kinases")
    logger.info(f"  Train structures: {len(train_idx)}, Val structures: {len(val_idx)}")
    logger.info(f"  Val kinases: {sorted(val_kinases)[:10]}...")

    return df.loc[train_idx], df.loc[val_idx]


def main():
    parser = argparse.ArgumentParser(description="Train Conformational Prior")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data", type=str, default="data/klifs_with_bioactivity.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"\n{'='*70}")
    print(f"  CONFORMATIONAL PRIOR TRAINING")
    print(f"{'='*70}")
    print(f"  Device: {device}")

    # Load dataset
    df = pd.read_csv(args.data)

    # Binary DFG label: out/out-like = 1, in = 0
    df = df[df["dfg"].isin(["in", "out", "out-like"])].copy()
    df["dfg_out"] = df["dfg"].isin(["out", "out-like"]).astype(int)
    logger.info(f"Dataset: {len(df)} structures ({df['dfg_out'].sum()} DFG-out, {(~df['dfg_out'].astype(bool)).sum()} DFG-in)")

    # Need pocket sequences from MOL2 files
    loader = KLIFSLoader()
    sequences = []
    valid_mask = []

    logger.info("Loading pocket sequences from MOL2 cache...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading sequences"):
        sid = row["structure_id"]
        seq = loader.get_pocket_sequence_from_mol2(sid)
        if seq is not None and len(seq) == 85:
            sequences.append(seq)
            valid_mask.append(True)
        else:
            # Use a gap-filled placeholder (will still train, just less informative)
            sequences.append("-" * 85)
            valid_mask.append(True)

    df["pocket_sequence"] = sequences

    # Kinase-identity split
    train_df, val_df = kinase_identity_split(df, seed=args.seed)

    # Class distribution
    train_pos = train_df["dfg_out"].sum()
    val_pos = val_df["dfg_out"].sum()
    print(f"  Train: {len(train_df)} ({train_pos} DFG-out, {len(train_df)-train_pos} DFG-in)")
    print(f"  Val:   {len(val_df)} ({val_pos} DFG-out, {len(val_df)-val_pos} DFG-in)")

    # Datasets
    train_dataset = PocketSequenceDataset(
        train_df["structure_id"].tolist(),
        train_df["pocket_sequence"].tolist(),
        train_df["dfg_out"].tolist(),
    )
    val_dataset = PocketSequenceDataset(
        val_df["structure_id"].tolist(),
        val_df["pocket_sequence"].tolist(),
        val_df["dfg_out"].tolist(),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    config = ConformationalPriorConfig()
    model = ConformationalPrior(config).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class-weighted loss
    pos_weight = torch.tensor(
        [(len(train_df) - train_pos) / max(train_pos, 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_auroc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(device)
                logits = model(seqs)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs > 0.5).astype(int)

        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.5
        bal_acc = balanced_accuracy_score(all_labels, preds)

        record = {"epoch": epoch, "train_loss": train_loss, "auroc": auroc, "balanced_acc": bal_acc}
        history.append(record)

        if epoch % 10 == 0 or epoch <= 3:
            print(f"  Epoch {epoch}: loss={train_loss:.4f}, auroc={auroc:.3f}, bal_acc={bal_acc:.3f}")

        if auroc > best_auroc:
            best_auroc = auroc
            model.save("checkpoints/conf_prior.pt")
            if epoch % 10 == 0 or epoch <= 3:
                print(f"    ✓ New best (auroc={auroc:.3f})")

    # Save history
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/conf_prior_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Gold standard validation ──
    print(f"\n  Gold Standard Validation:")
    best_model = ConformationalPrior.from_pretrained("checkpoints/conf_prior.pt").to(device)

    gs_path = Path("gold-standard-inhibitor-curation/data/known_inhibitors_curated.csv")
    if gs_path.exists():
        gs_df = pd.read_csv(gs_path)

        dfg_in_scores = []
        dfg_out_scores = []

        for _, row in gs_df.iterrows():
            # Try to get pocket sequence
            # Gold standard uses PDB codes, need to find matching KLIFS structure
            dfg = row.get("DFG", "").lower()
            # Use a simple lookup from our main dataset
            name = row.get("NAME", "")
            pdb = row.get("PDB", "")

            match = df[(df["pdb"] == pdb)].head(1)
            if len(match) > 0:
                seq = match.iloc[0]["pocket_sequence"]
            else:
                seq = "-" * 85

            score = best_model.predict(seq)

            if "out" in dfg:
                dfg_out_scores.append(score)
            elif "in" in dfg:
                dfg_in_scores.append(score)

        if dfg_out_scores:
            mean_out = np.mean(dfg_out_scores)
            print(f"    DFG-out structures: mean P(DFG-out) = {mean_out:.3f} (target > 0.65)")
        if dfg_in_scores:
            mean_in = np.mean(dfg_in_scores)
            print(f"    DFG-in structures:  mean P(DFG-out) = {mean_in:.3f} (target < 0.35)")
    else:
        print("    Gold standard file not found, skipping")

    # Upload to S3
    try:
        import subprocess
        subprocess.run(["aws", "s3", "cp", "checkpoints/conf_prior.pt",
                       "s3://hyaline-kinase-data/models/conf_prior/conf_prior.pt"],
                      check=True, capture_output=True)
        print(f"  ✓ Uploaded to S3")
    except Exception as e:
        logger.warning(f"S3 upload failed: {e}")

    print(f"\n  Best AuROC: {best_auroc:.3f}")
    print(f"  Saved to: checkpoints/conf_prior.pt")


if __name__ == "__main__":
    main()
