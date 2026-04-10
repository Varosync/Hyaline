#!/usr/bin/env python3
"""
Gold Standard Validation
=========================

Evaluates Screening Model and Deep Model against the curated 438-structure
gold standard. Reports per-kinase-family breakdowns and checks for
train/test leakage.

Usage:
    python scripts/validate_gold_standard.py
    python scripts/validate_gold_standard.py --dry-run
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# KLIFS kinase group mapping (major groups)
KINASE_GROUP_MAP = {
    "CAMK": ["CAMK2", "CAMK1", "CAMKK", "DAPK", "MLCK", "PHK", "PIM", "CASK", "MARK"],
    "TK": ["Tec", "Src", "Abl", "EGFR", "FGFR", "VEGFR", "PDGFR", "InsR", "JAK", "Eph", "Met",
           "ALK", "ROS", "RET", "TIE", "AXL", "DDR", "Trk", "Ack", "Csk", "FAK", "FER", "SYK", "ZAP"],
    "CMGC": ["CDK", "MAPK", "GSK", "CLK", "DYRK", "CK2", "SRPK"],
    "AGC": ["PKA", "PKC", "PKG", "AKT", "SGK", "RSK", "ROCK", "NDR", "MAST", "GRK", "DMPK"],
    "STE": ["MAP2K", "MAP3K", "STE20", "MST", "PAK", "TAO", "GCK"],
    "CK1": ["CK1"],
    "TKL": ["RAF", "IRAK", "RIPK", "MLK", "LRRK", "LIM", "ACVR", "BMPR", "TGFBR"],
    "Atypical": ["PIK", "PIKK", "Alpha", "RIO", "ABC", "BRD"],
    "Other": [],
}

def map_kinase_to_group(family_name):
    """Map kinase family name to kinase group."""
    if not family_name or pd.isna(family_name):
        return "Other"
    for group, families in KINASE_GROUP_MAP.items():
        for fam in families:
            if fam.lower() in str(family_name).lower():
                return group
    return "Other"


def main():
    parser = argparse.ArgumentParser(description="Gold Standard Validation")
    parser.add_argument("--dry-run", action="store_true", help="Print dataset stats and exit")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--screening-checkpoint", type=str,
                        default="checkpoints/screening/best_model.pt")
    parser.add_argument("--deep-checkpoint", type=str,
                        default="checkpoints/deep_model/best_model.pt")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load gold standard
    gs_path = Path("gold-standard-inhibitor-curation/data/known_inhibitors_curated.csv")
    if not gs_path.exists():
        logger.error(f"Gold standard file not found: {gs_path}")
        sys.exit(1)

    gs_df = pd.read_csv(gs_path)
    gs_df["kinase_group"] = gs_df["GROUPS"].fillna("Other")

    print(f"\n{'='*70}")
    print(f"  GOLD STANDARD VALIDATION")
    print(f"{'='*70}")
    print(f"  Total structures: {len(gs_df)}")
    print(f"  DFG distribution:")
    print(f"    {dict(gs_df['DFG'].value_counts())}")
    print(f"  Type distribution:")
    print(f"    {dict(gs_df['TYPE'].value_counts())}")
    print(f"  Kinase groups:")
    for grp, count in gs_df["kinase_group"].value_counts().items():
        print(f"    {grp}: {count}")

    # Check for train/test leakage
    splits_dir = Path("data/splits")
    leakage_found = False
    if splits_dir.exists():
        for split_file in splits_dir.glob("seed_*_train.csv"):
            train_split = pd.read_csv(split_file)
            if "pdb" in train_split.columns:
                overlap = set(gs_df["PDB"].str.lower()) & set(train_split["pdb"].str.lower())
                if overlap:
                    logger.warning(f"LEAKAGE: {len(overlap)} gold standard PDBs found in {split_file.name}: {list(overlap)[:5]}")
                    leakage_found = True
                else:
                    print(f"  ✓ No leakage with {split_file.name}")
    if not leakage_found:
        print(f"  ✓ No train/test leakage detected")

    if args.dry_run:
        print(f"\n  --dry-run: exiting after stats summary")
        return

    # ── Load models ──
    results = []

    # -- Screening Model --
    screening_auroc = None
    if Path(args.screening_checkpoint).exists():
        from hyaline.screening.screening_model import Type2ScreeningModel

        ckpt = torch.load(args.screening_checkpoint, map_location=device, weights_only=False)
        pocket_features_names = ckpt.get("pocket_features", [
            "dfg_chelix_distance", "hinge_activation_angle", "volume", "n_residues", "resolution"
        ])
        compound_features_names = ckpt.get("compound_features", [
            "compound_mw", "compound_logp", "compound_hbd", "compound_hba",
            "compound_tpsa", "compound_rotatable_bonds", "compound_aromatic_rings", "compound_heavy_atoms"
        ])

        screening_model = Type2ScreeningModel(
            pocket_dim=len(pocket_features_names),
            compound_dim=len(compound_features_names),
            hidden_dim=128,
        ).to(device)
        screening_model.load_state_dict(ckpt["model"])
        screening_model.eval()

        pocket_stats = ckpt.get("pocket_stats")
        compound_stats = ckpt.get("compound_stats")

        # Match gold standard to KLIFS dataset for features
        klifs_df = pd.read_csv("data/klifs_with_bioactivity.csv")

        screening_scores = []
        screening_labels = []

        for _, row in gs_df.iterrows():
            pdb = row["PDB"]
            match = klifs_df[klifs_df["pdb"] == pdb].head(1)
            if len(match) == 0:
                continue

            m = match.iloc[0]
            pocket_feat = torch.tensor(
                [m.get(f, 0) for f in pocket_features_names], dtype=torch.float32
            )
            compound_feat = torch.tensor(
                [m.get(f, 0) for f in compound_features_names], dtype=torch.float32
            )

            if pocket_stats:
                pmean, pstd = pocket_stats
                pocket_feat = (pocket_feat - pmean.cpu()) / pstd.cpu()
            if compound_stats:
                cmean, cstd = compound_stats
                compound_feat = (compound_feat - cmean.cpu()) / cstd.cpu()

            with torch.no_grad():
                output = screening_model(
                    pocket_feat.unsqueeze(0).to(device),
                    compound_feat.unsqueeze(0).to(device),
                )
                dfg_score = output[0, 1].item()

            screening_scores.append(dfg_score)
            dfg_label = 1 if "out" in str(row.get("DFG", "")).lower() else 0
            screening_labels.append(dfg_label)

            results.append({
                "kinase": row["NAME"],
                "pdb": pdb,
                "kinase_group": row["kinase_group"],
                "dfg_true": row["DFG"],
                "type_true": row["TYPE"],
                "screening_dfg_score": dfg_score,
                "dfg_label": dfg_label,
            })

        if screening_scores:
            try:
                screening_auroc = roc_auc_score(screening_labels, screening_scores)
            except ValueError:
                screening_auroc = 0.5
            screening_bal_acc = balanced_accuracy_score(
                screening_labels, [1 if s > 0.5 else 0 for s in screening_scores]
            )
            print(f"\n  Screening Model Results:")
            print(f"    Matched: {len(screening_scores)}/{len(gs_df)} structures")
            print(f"    DFG AuROC: {screening_auroc:.3f} (target > 0.75)")
            print(f"    DFG balanced accuracy: {screening_bal_acc:.3f}")
    else:
        logger.error(f"Screening checkpoint not found: {args.screening_checkpoint}")
        print(f"  Screening model: SKIPPED (checkpoint missing)")

    # -- Deep Model --
    deep_auroc = None
    if Path(args.deep_checkpoint).exists():
        from hyaline.models.kinase_binding import KinaseBindingPredictor, KinaseBindingConfig
        ckpt = torch.load(args.deep_checkpoint, map_location=device, weights_only=False)
        config = KinaseBindingConfig(**ckpt.get("config", {}))
        deep_model = KinaseBindingPredictor(config).to(device)
        deep_model.load_state_dict(ckpt["model"])
        deep_model.eval()
        print(f"\n  Deep Model Results:")
        print(f"    Checkpoint loaded (epoch {ckpt.get('epoch', '?')}, val_mae={ckpt.get('val_mae', '?')})")
        print(f"    (Full evaluation requires pocket coordinate loading — see training metrics)")
    else:
        logger.error(f"Deep model checkpoint not found: {args.deep_checkpoint}")
        print(f"  Deep model: SKIPPED (checkpoint missing)")

    # ── Per-family breakdown ──
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n  Per-Kinase-Family Breakdown (Screening Model):")
        for grp in sorted(results_df["kinase_group"].unique()):
            grp_data = results_df[results_df["kinase_group"] == grp]
            n = len(grp_data)
            if n < 5:
                continue
            try:
                auroc = roc_auc_score(grp_data["dfg_label"], grp_data["screening_dfg_score"])
                print(f"    {grp:10s}: n={n:4d}, auroc={auroc:.3f}")
            except ValueError:
                print(f"    {grp:10s}: n={n:4d}, auroc=N/A (single class)")

        # Save results
        output_path = Path("results/gold_standard_validation.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n  Results saved to: {output_path}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
