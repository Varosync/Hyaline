# Hyaline Kinase — Conformational Selectivity

**Branch:** `kinase-real-descriptors` (built on `kinase-v1`; `master` is GPCR-only)

Extension of Hyaline to **kinase conformational selectivity**: predicting the
DFG-in vs DFG-out state that decides whether a Type I or Type II inhibitor can
bind. Same kinase, same sequence, different 3D shape, different binding.

---

## Status (updated after reproducibility audit)

Every number below was regenerated on real hardware. Claims are tagged
**REAL** (real KLIFS data), **SYNTHETIC** (model run on invented data — a
mechanism check, not real-molecule performance), or **LEAKY** (inflated by an
evaluation that lets the same kinase appear in train and test).

> TL;DR: the impressive headline numbers are either **synthetic** or **leakage
> inflated**. The honest, leakage-free signal comes from a single interpretable
> **geometric descriptor**, which already beats the leakage-corrected sequence
> model. The underlying KLIFS data is sound (known-drug check passes 5/6).

### Claim vs reality

| Metric | Value (reproduced) | Tag | Command |
|---|---|---|---|
| Hybrid RF / MLP regression | R² = 0.964 / 0.947 | SYNTHETIC | `python scripts/hybrid_kinase_model.py` |
| DFG × drug-size importance | 0.56 (hybrid gen.) **but 0.155** (ablation gen.) | SYNTHETIC | `python scripts/hybrid_kinase_model.py` |
| Sequence-only regression | R² = 0.015 | SYNTHETIC | `python scripts/kinase_ablation.py` |
| Structure features (RF/GBM) | R² = 0.352 | SYNTHETIC | `python scripts/kinase_ablation.py` |
| Static EGNN / Spiking EGNN | R² = −0.064 / −0.089 (did not converge; spiking no better, p = 0.125) | SYNTHETIC | `python scripts/kinase_ablation.py` |
| DFG classifier, pocket sequence, **ungrouped** | acc 0.893, AUROC 0.946 | LEAKY | `python scripts/kinase_audit.py` |
| DFG classifier, pocket sequence, **grouped LOKO** | acc 0.711, **AUROC 0.62** | REAL | `python scripts/kinase_audit.py` |
| DFG separation, **geometric descriptor** (training-free) | **AUROC 0.796** | REAL | `python scripts/kinase_audit.py` |
| Known-drug Type I/II validation | 5 / 6 correct | REAL | `python scripts/klifs_validation.py` |

Majority-class baseline for DFG state is **0.792** — note the grouped sequence
classifier (0.711) is *below* it.

---

## The two findings that matter

### 1. Sequence features leak kinase identity
Under an ungrouped split the pocket-sequence classifier looks strong
(AUROC 0.946), but under **grouped leave-one-kinase-out** (no kinase in both
train and test) AUROC collapses to **0.62** — barely above random. The
85-residue pocket is essentially a kinase's ID badge, so ungrouped folds let the
model memorize identity rather than learn conformation. Always report the
grouped number.

### 2. A single real geometric descriptor beats the sequence model, without leakage
Parsing real Cα coordinates from KLIFS (`structure_get_pocket` mol2) and
computing the **DFG-to-αC-helix distance**:

| DFG state | distance |
|---|---|
| DFG-in  | 10.8 ± 1.6 Å |
| DFG-out | 12.9 ± 2.3 Å |

The raw distance, with **no training and no possible leakage**, classifies DFG
state at **AUROC 0.796** — better than the leakage-corrected sequence model
(0.62). (Note: the classes overlap; the README's old "16–22 Å for DFG-out"
claim does not hold for this centroid definition.) Adding the hinge angle and a
grouped classifier is the next step.

---

## Reproduce everything

```bash
pip install scikit-learn scipy requests numpy torch      # deps used by the audit
python scripts/kinase_audit.py        # REAL numbers (KLIFS; caches to klifs_cache/)
python scripts/klifs_validation.py    # REAL known-drug Type I/II check (5/6)
python scripts/hybrid_kinase_model.py # SYNTHETIC hybrid regression
python scripts/kinase_ablation.py     # SYNTHETIC seq/struct/EGNN ablation (slow, CPU)
```

Results are written to `checkpoints/kinase_audit.json` and cached under
`klifs_cache/` for deterministic re-runs.

---

## Codebase

| Path | Purpose |
|------|---------|
| `scripts/kinase_audit.py` | **Reproducibility audit** — regenerates the REAL numbers above |
| `hyaline/features/kinase_geometry.py` | Geometric descriptors (DFG–αC distance, hinge angle) |
| `hyaline/loaders/klifs_loader.py` | KLIFS API client |
| `hyaline/loaders/klifs_pipeline.py` | Feature-extraction pipeline |
| `hyaline/models/kinase_binding.py` | Kinase binding / Spiking EGNN model |
| `scripts/hybrid_kinase_model.py` | Hybrid RF+MLP baseline (synthetic) |
| `scripts/kinase_ablation.py` | Structure vs sequence vs EGNN ablation (synthetic) |
| `scripts/klifs_validation.py` | Known-drug Type I/II validation (real) |
| `scripts/train_real_klifs.py` | Original DFG classifier (hardcoded ABL1 test; superseded by `kinase_audit.py`) |

## Data

- **KLIFS:** conformation-annotated human kinase structures on a standardized
  85-residue pocket alignment. Audit set: **3,259 structures, 15 kinases**
  (2,580 DFG-in / 679 DFG-out). Counts grow as KLIFS is updated.
- Real pocket coordinates via KLIFS `structure_get_pocket` (mol2, 85 residues).

## Known limitations

- The regression results (hybrid, ablation) are **synthetic** — two independent,
  non-comparable data generators; they demonstrate mechanism, not real-molecule
  accuracy.
- The spiking-EGNN "synchronization" idea shows **no measurable benefit** over a
  static EGNN on the tested benchmark.
- `train_screening_model_real_features.py` still uses **mock compound features**
  (`torch.randn`) despite the "real features" name.
- Conformation-paired ΔpKi (same ligand, both DFG states) is scarce in real
  data; hence the synthetic stand-in.

## Roadmap

Cleanup toward a defensible artifact (Ayman's phase plan). Invariants across all
phases: record whether each input is **experimental or predicted**,
**determinism** (same input → same output), and **every README claim backed by a
command**.

- **Phase 0/1** — pin the real descriptor pipeline; `analyze` command returning
  DFG/αC state, the geometric fingerprint, and an inhibitor-class call, for both
  crystals and predicted (AlphaFold) models.
- **Phase 2** — one defensible number: grouped leave-one-kinase-out on UniProt,
  reproduced by `make benchmark`. (Prototyped in `kinase_audit.py`.)
- **Phase 3** — the atlas: browsable/downloadable map of every human kinase
  (DFG distance vs hinge angle, colored by state).
- **Phase 4/5** — Colab/UI + PyMOL export; clean-clone release with artifacts.
