#!/usr/bin/env python3
"""
Train Type II Screening Model v2 — Real Features, Class Balancing
==================================================================

Fixes from v1:
- Uses REAL compound descriptors (MW, LogP, HBD, HBA, TPSA, etc.) instead of random noise
- Uses all available pocket geometry features
- Class-frequency-inverse weighted BCE for imbalanced Type II / DFG-out labels
- Cosine annealing LR schedule
- Checkpoints on VALIDATION loss (not train loss)
- Tracks balanced accuracy and per-class recall
- Deterministic seeding
"""

import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    roc_auc_score, mean_absolute_error
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyaline.screening.screening_model import Type2ScreeningModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Feature columns ──────────────────────────────────────────────────────────

POCKET_FEATURES = [
    "dfg_chelix_distance",
    "hinge_activation_angle",
    "volume",
    "n_residues",
    "resolution",
]

COMPOUND_FEATURES = [
    "compound_mw",
    "compound_logp",
    "compound_hbd",
    "compound_hba",
    "compound_tpsa",
    "compound_rotatable_bonds",
    "compound_aromatic_rings",
    "compound_heavy_atoms",
]


# ── Dataset ──────────────────────────────────────────────────────────────────

class KLIFSScreeningDataset(Dataset):
    """Dataset with real pocket geometry + real compound descriptors."""

    def __init__(self, df: pd.DataFrame, pocket_stats=None, compound_stats=None):
        self.df = df.reset_index(drop=True)

        # Pocket features
        pocket_raw = torch.tensor(
            df[POCKET_FEATURES].fillna(0).values, dtype=torch.float32
        )
        if pocket_stats is None:
            self.pocket_mean = pocket_raw.mean(dim=0)
            self.pocket_std = pocket_raw.std(dim=0).clamp(min=1e-8)
        else:
            self.pocket_mean, self.pocket_std = pocket_stats
        self.pocket_features = (pocket_raw - self.pocket_mean) / self.pocket_std

        # Real compound features
        compound_raw = torch.tensor(
            df[COMPOUND_FEATURES].fillna(0).values, dtype=torch.float32
        )
        if compound_stats is None:
            self.compound_mean = compound_raw.mean(dim=0)
            self.compound_std = compound_raw.std(dim=0).clamp(min=1e-8)
        else:
            self.compound_mean, self.compound_std = compound_stats
        self.compound_features = (compound_raw - self.compound_mean) / self.compound_std

        # Labels
        self.type_labels = torch.tensor(
            (df["type"] == "Type II").astype(int).values, dtype=torch.float32
        )
        # DFG: treat "out" and "out-like" as positive
        self.dfg_labels = torch.tensor(
            df["dfg"].isin(["out", "out-like"]).astype(int).values, dtype=torch.float32
        )
        self.pki_values = torch.tensor(df["pki"].values, dtype=torch.float32)

    def get_normalization_stats(self):
        return (
            (self.pocket_mean, self.pocket_std),
            (self.compound_mean, self.compound_std),
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "pocket": self.pocket_features[idx],
            "compound": self.compound_features[idx],
            "type_label": self.type_labels[idx],
            "dfg_label": self.dfg_labels[idx],
            "pki": self.pki_values[idx],
        }


# ── Class weights ────────────────────────────────────────────────────────────

def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """Class-frequency-inverse weight for BCEWithLogitsLoss."""
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, type_criterion, dfg_criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        pocket = batch["pocket"].to(device)
        compound = batch["compound"].to(device)
        type_label = batch["type_label"].to(device)
        dfg_label = batch["dfg_label"].to(device)
        pki = batch["pki"].to(device)

        optimizer.zero_grad()

        output = model(pocket, compound)
        # output[:, 0] = type2 score (already sigmoided in model)
        # output[:, 1] = dfg_out_prob (already sigmoided)
        # output[:, 2] = pKi (raw regression)

        # We need logits for BCEWithLogitsLoss, but model applies sigmoid.
        # Use plain BCE with pos_weight approximation via weighted BCE.
        type_loss = type_criterion(output[:, 0], type_label)
        dfg_loss = dfg_criterion(output[:, 1], dfg_label)
        pki_loss = nn.MSELoss()(output[:, 2], pki)

        loss = type_loss + dfg_loss + 0.1 * pki_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device):
    model.eval()
    all_type_scores, all_type_labels = [], []
    all_dfg_scores, all_dfg_labels = [], []
    all_pki_preds, all_pki_true = [], []

    with torch.no_grad():
        for batch in loader:
            pocket = batch["pocket"].to(device)
            compound = batch["compound"].to(device)

            output = model(pocket, compound)

            all_type_scores.extend(output[:, 0].cpu().numpy())
            all_type_labels.extend(batch["type_label"].numpy())
            all_dfg_scores.extend(output[:, 1].cpu().numpy())
            all_dfg_labels.extend(batch["dfg_label"].numpy())
            all_pki_preds.extend(output[:, 2].cpu().numpy())
            all_pki_true.extend(batch["pki"].numpy())

    type_scores = np.array(all_type_scores)
    type_labels = np.array(all_type_labels)
    type_preds = (type_scores > 0.5).astype(int)

    dfg_scores = np.array(all_dfg_scores)
    dfg_labels = np.array(all_dfg_labels)
    dfg_preds = (dfg_scores > 0.5).astype(int)

    # Type II metrics
    type_balanced_acc = balanced_accuracy_score(type_labels, type_preds)
    type_recall_minority = recall_score(type_labels, type_preds, pos_label=1, zero_division=0)
    type_recall_majority = recall_score(type_labels, type_preds, pos_label=0, zero_division=0)
    try:
        type_auroc = roc_auc_score(type_labels, type_scores)
    except ValueError:
        type_auroc = 0.5

    # DFG metrics
    dfg_balanced_acc = balanced_accuracy_score(dfg_labels, dfg_preds)
    dfg_recall_minority = recall_score(dfg_labels, dfg_preds, pos_label=1, zero_division=0)
    try:
        dfg_auroc = roc_auc_score(dfg_labels, dfg_scores)
    except ValueError:
        dfg_auroc = 0.5

    # pKi
    pki_mae = mean_absolute_error(all_pki_true, all_pki_preds)

    return {
        "type_ii_balanced_acc": float(type_balanced_acc),
        "type_ii_recall_minority": float(type_recall_minority),
        "type_ii_recall_majority": float(type_recall_majority),
        "type_ii_auroc": float(type_auroc),
        "dfg_balanced_acc": float(dfg_balanced_acc),
        "dfg_recall_minority": float(dfg_recall_minority),
        "dfg_auroc": float(dfg_auroc),
        "pki_mae": float(pki_mae),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Type II Screening Model v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data", type=str, default="data/klifs_with_bioactivity.csv")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"\n{'='*70}")
    print(f"  SCREENING MODEL v2 — Real Features + Class Balancing")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")

    # ── Dataset integrity check ──
    manifest_path = Path("data/dataset_manifest.json")
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        with open(args.data, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        if actual_hash != manifest["sha256_hash"]:
            logger.warning(f"Dataset hash mismatch! Expected {manifest['sha256_hash'][:16]}..., got {actual_hash[:16]}...")
        else:
            logger.info("Dataset integrity verified (SHA-256 match)")

    # ── Load data ──
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} structures")

    # Check for required columns
    missing_pocket = [c for c in POCKET_FEATURES if c not in df.columns]
    missing_compound = [c for c in COMPOUND_FEATURES if c not in df.columns]
    if missing_pocket:
        raise ValueError(f"Missing pocket features: {missing_pocket}")
    if missing_compound:
        raise ValueError(f"Missing compound features: {missing_compound}")

    # Use existing split if available, otherwise create one
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
    else:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.17, random_state=args.seed, stratify=df["type"])

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Class distribution
    type_dist = train_df["type"].value_counts()
    dfg_dist = train_df["dfg"].value_counts()
    print(f"  Type distribution: {dict(type_dist)}")
    print(f"  DFG distribution: {dict(dfg_dist)}")

    # ── Datasets ──
    train_dataset = KLIFSScreeningDataset(train_df)
    pocket_stats, compound_stats = train_dataset.get_normalization_stats()
    val_dataset = KLIFSScreeningDataset(val_df, pocket_stats, compound_stats)

    # ── Class-balanced sampling ──
    type_weights = compute_pos_weight(train_dataset.type_labels)
    dfg_weights = compute_pos_weight(train_dataset.dfg_labels)
    print(f"  Type II pos_weight: {type_weights.item():.2f}")
    print(f"  DFG-out pos_weight: {dfg_weights.item():.2f}")

    # Weighted BCE loss
    type_criterion = nn.BCELoss(
        weight=None,
        reduction="none",
    )
    dfg_criterion = nn.BCELoss(
        weight=None,
        reduction="none",
    )

    # Custom weighted loss wrapper
    class WeightedBCE(nn.Module):
        def __init__(self, pos_weight):
            super().__init__()
            self.pos_weight = pos_weight

        def forward(self, pred, target):
            # Apply class-frequency-inverse weighting
            weight = torch.where(target == 1, self.pos_weight, torch.tensor(1.0, device=pred.device))
            loss = nn.functional.binary_cross_entropy(pred, target, reduction="none")
            return (loss * weight).mean()

    type_criterion = WeightedBCE(type_weights.to(device))
    dfg_criterion = WeightedBCE(dfg_weights.to(device))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model ──
    model = Type2ScreeningModel(
        pocket_dim=len(POCKET_FEATURES),
        compound_dim=len(COMPOUND_FEATURES),
        hidden_dim=128,
        dropout=0.2,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs // 2, eta_min=args.lr_min
    )

    # ── Training loop ──
    best_val_loss = float("inf")
    best_val_balanced_acc = 0.0
    history = []
    ckpt_dir = Path("checkpoints/screening")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, type_criterion, dfg_criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Composite val metric for checkpointing
        val_composite = (val_metrics["type_ii_balanced_acc"] + val_metrics["dfg_balanced_acc"]) / 2

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "lr": float(lr_now),
            **val_metrics,
        }
        history.append(record)

        if epoch % 10 == 0 or epoch <= 3:
            print(f"\n  Epoch {epoch}/{args.epochs} (lr={lr_now:.6f})")
            print(f"    Train Loss: {train_loss:.4f}")
            print(f"    Type II: balanced_acc={val_metrics['type_ii_balanced_acc']:.3f}, "
                  f"recall_minority={val_metrics['type_ii_recall_minority']:.3f}, "
                  f"auroc={val_metrics['type_ii_auroc']:.3f}")
            print(f"    DFG:     balanced_acc={val_metrics['dfg_balanced_acc']:.3f}, "
                  f"recall_minority={val_metrics['dfg_recall_minority']:.3f}, "
                  f"auroc={val_metrics['dfg_auroc']:.3f}")
            print(f"    pKi MAE: {val_metrics['pki_mae']:.3f}")

        # Save best model (on balanced accuracy, not train loss)
        if val_composite > best_val_balanced_acc:
            best_val_balanced_acc = val_composite
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "pocket_stats": pocket_stats,
                "compound_stats": compound_stats,
                "pocket_features": POCKET_FEATURES,
                "compound_features": COMPOUND_FEATURES,
                "seed": args.seed,
            }, ckpt_dir / "best_model.pt")
            if epoch % 10 == 0 or epoch <= 3:
                print(f"    ✓ New best (composite balanced_acc={val_composite:.3f})")

    # ── Save history ──
    with open(ckpt_dir / "training_history_v2.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Verify learning happened ──
    epoch1_acc = history[0]["type_ii_balanced_acc"]
    epoch50_acc = history[min(49, len(history) - 1)]["type_ii_balanced_acc"]
    delta = abs(epoch50_acc - epoch1_acc)
    print(f"\n  Type II balanced_acc change (epoch 1→50): {delta:.4f}")
    if delta < 0.01:
        logger.warning("Classification head may not be learning (delta < 0.01)")
    else:
        print(f"  ✓ Classification head is learning")

    # ── Final summary ──
    best = max(history, key=lambda r: (r["type_ii_balanced_acc"] + r["dfg_balanced_acc"]) / 2)
    print(f"\n{'='*70}")
    print(f"  BEST EPOCH: {best['epoch']}")
    print(f"  Type II balanced_acc: {best['type_ii_balanced_acc']:.3f}")
    print(f"  Type II minority recall: {best['type_ii_recall_minority']:.3f}")
    print(f"  DFG balanced_acc: {best['dfg_balanced_acc']:.3f}")
    print(f"  pKi MAE: {best['pki_mae']:.3f}")
    print(f"  Saved to: {ckpt_dir / 'best_model.pt'}")
    print(f"{'='*70}")

    # Upload to S3
    try:
        import subprocess
        subprocess.run([
            "aws", "s3", "cp",
            str(ckpt_dir / "best_model.pt"),
            "s3://hyaline-kinase-data/models/screening_v2/best_model.pt",
        ], check=True, capture_output=True)
        subprocess.run([
            "aws", "s3", "cp",
            str(ckpt_dir / "training_history_v2.json"),
            "s3://hyaline-kinase-data/models/screening_v2/training_history.json",
        ], check=True, capture_output=True)
        print(f"  ✓ Uploaded to S3")
    except Exception as e:
        logger.warning(f"S3 upload failed: {e}")


if __name__ == "__main__":
    main()
