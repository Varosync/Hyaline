# Hyaline Kinase — Conformational Selectivity

**Branch:** `kinase-v1` (separate from `master`, which is GPCR-only)

## Project

Extension of Hyaline to **kinase conformational selectivity**: predicting how drug binding changes between DFG-in and DFG-out conformations. Same kinase, different 3D shape → different binding. Relevant for Type I vs Type II inhibitor design.

## Main Finding

Structure is necessary. Sequence-only models fail (R² ≈ 0.01). The main signal is the **interaction term** DFG_displacement × Drug_size (~56% of feature importance). Hand-crafted features outperform end-to-end GNNs on this dataset (~1.6K structures). The hybrid model (RF features + MLP) reaches R² ≈ 0.95.

## Codebase

| Path | Purpose |
|------|---------|
| `hyaline/loaders/klifs_loader.py` | KLIFS API client |
| `hyaline/loaders/klifs_pipeline.py` | Feature extraction pipeline |
| `hyaline/models/kinase_binding.py` | Spiking EGNN + kinase model |
| `scripts/hybrid_kinase_model.py` | Hybrid RF+MLP baseline |
| `scripts/kinase_ablation.py` | Structure vs sequence ablations |
| `scripts/klifs_validation.py` | Validation on known drugs |
| `scripts/train_real_klifs.py` | DFG classifier on real KLIFS |

## Data

- **KLIFS:** ~1,661 structures, 10 kinases, DFG/C-helix annotations
- **S3:** `in_notion`
- **ChEMBL:** 132 bioactivity records via KLIFS
