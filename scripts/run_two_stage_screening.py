#!/usr/bin/env python3
"""
Two-Stage Screening Pipeline
==============================

Stage 1: Screening Model scores all compounds (fast, MLP-based)
Stage 2: Deep Model refines top N candidates (3D structure-based)

Usage:
    python scripts/run_two_stage_screening.py \
        --compounds library.csv \
        --kinase ABL1 \
        --output results/abl1_screen.csv \
        --top-n 100
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Screening Pipeline")
    parser.add_argument("--compounds", type=str, required=True,
                        help="CSV with SMILES column")
    parser.add_argument("--kinase", type=str, required=True,
                        help="Target kinase name (e.g., ABL1)")
    parser.add_argument("--output", type=str, default="results/two_stage_results.csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--screening-checkpoint", type=str,
                        default="checkpoints/screening/best_model.pt")
    parser.add_argument("--deep-checkpoint", type=str,
                        default="checkpoints/deep_model/best_model.pt")
    parser.add_argument("--prior-checkpoint", type=str,
                        default="checkpoints/conf_prior.pt")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"\n{'='*70}")
    print(f"  TWO-STAGE SCREENING PIPELINE")
    print(f"{'='*70}")
    print(f"  Target kinase: {args.kinase}")
    print(f"  Device: {device}")

    # Load compound library
    compounds_df = pd.read_csv(args.compounds)
    n_total = len(compounds_df)
    print(f"  Compounds: {n_total}")

    # Set up logging
    log_path = Path("results/screening_pipeline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: Screening Model (fast filter)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  STAGE 1: Screening Model")
    t1_start = time.time()

    from hyaline.screening.screening_model import Type2ScreeningModel

    if not Path(args.screening_checkpoint).exists():
        logger.error(f"Screening checkpoint not found: {args.screening_checkpoint}")
        sys.exit(1)

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

    # Get pocket features for the target kinase
    klifs_df = pd.read_csv("data/klifs_with_bioactivity.csv")
    kinase_structures = klifs_df[klifs_df["kinase_name"].str.contains(args.kinase, case=False, na=False)]

    if len(kinase_structures) == 0:
        logger.error(f"No structures found for kinase: {args.kinase}")
        sys.exit(1)

    # Use the best-resolution structure
    best_struct = kinase_structures.sort_values("resolution").iloc[0]
    pocket_feat = torch.tensor(
        [best_struct.get(f, 0) for f in pocket_features_names], dtype=torch.float32
    )
    if pocket_stats:
        pmean, pstd = pocket_stats
        pocket_feat = (pocket_feat - pmean.cpu()) / pstd.cpu()

    # Check if kinase has DFG-out structures
    has_dfg_out = kinase_structures["dfg"].isin(["out", "out-like"]).any()

    # Score all compounds
    screening_scores = []
    batch_size = 1024

    # Prepare compound features (use available columns or defaults)
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_df = compounds_df.iloc[start:end]

        # Extract compound features
        cfeat_list = []
        for f in compound_features_names:
            if f in batch_df.columns:
                cfeat_list.append(torch.tensor(batch_df[f].fillna(0).values, dtype=torch.float32))
            else:
                cfeat_list.append(torch.zeros(len(batch_df)))
        compound_feat = torch.stack(cfeat_list, dim=1)

        if compound_stats:
            cmean, cstd = compound_stats
            compound_feat = (compound_feat - cmean.cpu()) / cstd.cpu()

        pocket_batch = pocket_feat.unsqueeze(0).expand(len(batch_df), -1).to(device)
        compound_batch = compound_feat.to(device)

        with torch.no_grad():
            output = screening_model(pocket_batch, compound_batch)
            scores = output[:, 0].cpu().numpy()  # Type II score
            screening_scores.extend(scores)

    compounds_df["screening_score"] = screening_scores
    t1_elapsed = time.time() - t1_start

    # Filter to top N
    top_n = compounds_df.nlargest(args.top_n, "screening_score")
    print(f"    Processed: {n_total} compounds in {t1_elapsed:.1f}s")
    print(f"    Top {args.top_n} score range: [{top_n['screening_score'].min():.3f}, {top_n['screening_score'].max():.3f}]")
    logger.info(f"Stage 1: {n_total} compounds, {t1_elapsed:.1f}s, top-{args.top_n} filtered")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Deep Model (structure-based refinement)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  STAGE 2: Deep Model")
    t2_start = time.time()

    results = top_n.copy()
    results["pki_dfg_in"] = np.nan
    results["pki_dfg_out"] = np.nan
    results["delta_pki"] = np.nan
    results["sync_score"] = np.nan
    results["prior_dfg_out_prob"] = np.nan
    results["used_prior"] = False

    if Path(args.deep_checkpoint).exists() and has_dfg_out:
        from hyaline.models.kinase_binding import KinaseBindingPredictor, KinaseBindingConfig, KLIFSLoader
        from hyaline.models.conformational_prior import encode_pocket

        deep_ckpt = torch.load(args.deep_checkpoint, map_location=device, weights_only=False)
        deep_config = KinaseBindingConfig(**deep_ckpt.get("config", {}))
        deep_model = KinaseBindingPredictor(deep_config).to(device)
        deep_model.load_state_dict(deep_ckpt["model"])
        deep_model.eval()

        # Load pocket coords for both DFG-in and DFG-out
        loader = KLIFSLoader()
        dfg_in_struct = kinase_structures[kinase_structures["dfg"] == "in"].sort_values("resolution").head(1)
        dfg_out_struct = kinase_structures[kinase_structures["dfg"].isin(["out", "out-like"])].sort_values("resolution").head(1)

        if len(dfg_in_struct) > 0 and len(dfg_out_struct) > 0:
            sid_in = dfg_in_struct.iloc[0]["structure_id"]
            sid_out = dfg_out_struct.iloc[0]["structure_id"]

            coords_in = loader.get_pocket_coordinates(sid_in)
            coords_out = loader.get_pocket_coordinates(sid_out)
            seq = loader.get_pocket_sequence_from_mol2(sid_in) or "-" * 85

            if coords_in is not None and coords_out is not None:
                pocket_seq = torch.tensor(encode_pocket(seq), dtype=torch.long).unsqueeze(0).to(device)
                coords_in_t = torch.tensor(coords_in, dtype=torch.float32).unsqueeze(0).to(device)
                coords_out_t = torch.tensor(coords_out, dtype=torch.float32).unsqueeze(0).to(device)

                # Build edges
                from scipy.spatial import cKDTree
                nonz = np.any(np.abs(coords_in) > 1e-6, axis=1)
                valid_c = coords_in[nonz]
                valid_i = np.where(nonz)[0]
                tree = cKDTree(valid_c)
                k = min(10, len(valid_c) - 1)
                _, nn_idx = tree.query(valid_c, k=k+1)
                r, cl = [], []
                for i in range(len(valid_c)):
                    for j in range(1, k+1):
                        r.append(valid_i[i]); cl.append(valid_i[nn_idx[i,j]])
                ei = torch.tensor([r, cl], dtype=torch.long, device=device)
                bi = torch.zeros(85, dtype=torch.long, device=device)

                # Score each top compound
                for idx in results.index:
                    drug_fp = torch.zeros(1, 2048, device=device)
                    # Fill with available compound features
                    for fi, f in enumerate(compound_features_names):
                        if f in results.columns:
                            drug_fp[0, fi] = results.loc[idx, f] if pd.notna(results.loc[idx, f]) else 0

                    with torch.no_grad():
                        out = deep_model.predict_conformational_difference(
                            pocket_seq, coords_in_t, coords_out_t, drug_fp, ei, bi
                        )
                        results.loc[idx, "pki_dfg_in"] = out["pki_dfg_in"].item()
                        results.loc[idx, "pki_dfg_out"] = out["pki_dfg_out"].item()
                        results.loc[idx, "delta_pki"] = out["delta_pki"].item()
                        results.loc[idx, "sync_score"] = out["sync_dfg_in"].item()

    elif not has_dfg_out:
        # Fallback to ConformationalPrior
        print(f"    No DFG-out structures for {args.kinase} — using ConformationalPrior")
        if Path(args.prior_checkpoint).exists():
            from hyaline.models.conformational_prior import ConformationalPrior
            prior = ConformationalPrior.from_pretrained(args.prior_checkpoint).to(device)

            # Get pocket sequence
            loader_seq = None
            from hyaline.models.kinase_binding import KLIFSLoader
            loader = KLIFSLoader()
            sid = best_struct["structure_id"]
            loader_seq = loader.get_pocket_sequence_from_mol2(sid) or "-" * 85

            prior_prob = prior.predict(loader_seq)
            results["prior_dfg_out_prob"] = prior_prob
            results["used_prior"] = True
            print(f"    ConformationalPrior P(DFG-out) = {prior_prob:.3f}")
        else:
            logger.warning("ConformationalPrior checkpoint not found")
    else:
        logger.warning("Deep model checkpoint not found — Stage 2 skipped")

    t2_elapsed = time.time() - t2_start
    print(f"    Processed: {len(results)} compounds in {t2_elapsed:.1f}s")
    logger.info(f"Stage 2: {len(results)} compounds, {t2_elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════
    output_cols = ["screening_score", "pki_dfg_in", "pki_dfg_out", "delta_pki",
                   "sync_score", "prior_dfg_out_prob", "used_prior"]
    if "smiles" in results.columns or "SMILES" in results.columns:
        smiles_col = "smiles" if "smiles" in results.columns else "SMILES"
        output_cols = [smiles_col] + output_cols

    avail_cols = [c for c in output_cols if c in results.columns]
    extra_cols = [c for c in results.columns if c not in avail_cols]
    results[avail_cols + extra_cols[:5]].to_csv(args.output, index=False)

    print(f"\n  Results saved to: {args.output}")
    print(f"  Total runtime: {t1_elapsed + t2_elapsed:.1f}s")
    logger.info(f"Total: {t1_elapsed + t2_elapsed:.1f}s, output: {args.output}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
