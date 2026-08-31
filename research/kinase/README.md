# Hyaline Kinase — Conformational Selectivity

**Branch:** `kinase-real-descriptors` (built on `kinase-v1`; `master` is GPCR-only)

Predict the DFG-in vs DFG-out state that decides whether a Type I or Type II
inhibitor can bind a kinase — from interpretable geometric descriptors computed
on the real KLIFS pocket. Same kinase, same sequence, different 3D shape,
different binding.

The kinase tools (`analyze`, `atlas`, `benchmark`) are self-contained: no GPU, no
retraining, and none of the GPCR/torch dependencies of the main package.

## Install

```bash
git clone -b kinase-real-descriptors https://github.com/Varosync/Hyaline.git
cd Hyaline
pip install requests numpy scikit-learn pandas matplotlib pyarrow
```

## Quickstart — annotate a structure

```bash
python scripts/analyze_kinase.py 2hyy                                 # PDB code (KLIFS)
python scripts/analyze_kinase.py --klifs-id 1081                      # KLIFS structure_ID
python scripts/analyze_kinase.py --pdb-file model.pdb --kinase ABL1   # any PDB / AlphaFold
python scripts/analyze_kinase.py 2hyy --pymol out.pml                 # + annotated PyMOL session
```

Output (`2hyy`, imatinib-bound ABL1) — correctly called **DFG-out / Type II**:

```json
{
  "schema_version": "1.0",
  "identifier": "2hyy", "source": "pdb", "provenance": "experimental",
  "dfg_achelix_distance_A": 12.237, "hinge_activation_angle_deg": 59.668,
  "dfg_call": "DFG-out", "dfg_confidence": 0.674, "dfg_driver": "dfg_achelix_distance_A",
  "achelix_state": "in", "achelix_source": "klifs_annotation",
  "inhibitor_class": "Type II",
  "inhibitor_rationale": "DFG-out exposes the allosteric pocket engaged by Type II inhibitors"
}
```

For an arbitrary local PDB (crystal or AlphaFold model), the 85-residue pocket is
extracted by aligning to a KLIFS reference for the given kinase (validated: local
`2hyy.pdb` reproduces KLIFS descriptors to < 0.1 Å; AlphaFold ABL1/EGFR annotate
cleanly). `--pymol` writes a session that loads the structure, colours the DFG
motif and αC-helix, and labels the call.

## Colab

`research/kinase/colab/hyaline_kinase.ipynb` — open in Colab, set `INPUT` to a
**PDB ID** or **kinase name**, and Run all: one pass clones the repo, computes the
descriptors, prints the JSON call, and plots where the structure sits among real
kinase conformations.

## Atlas

A browsable, downloadable map of human kinases in KLIFS.

```bash
python scripts/build_kinase_atlas.py       # writes research/kinase/atlas/
```

- `index.html` — offline page (no external assets): searchable table + scatter
  (DFG-to-αC distance vs hinge angle, colored by DFG state). Open it in a browser.
- `kinase_atlas.parquet` / `.csv` — one row per kinase (loads in pandas).

Each row carries: accessible DFG states, structure counts, known Type I / Type II
inhibitors, a **Type-II-opportunity score** (heuristic: well-studied kinases that
reach DFG-out but have few known Type II inhibitors rank high — e.g. HGK, LRRK2),
and a geometric descriptor fingerprint where computed. Build: **318 kinases,
13,325 structures**; per-kinase counts reconcile against KLIFS by construction.

## Benchmark — the defensible number

```bash
python scripts/kinase_descriptors.py   # geometric descriptors + grouped LOKO + Figure 1
python scripts/kinase_audit.py         # sequence-classifier audit (leaky vs grouped)
python scripts/klifs_validation.py     # known-drug Type I/II check (5/6)
```

Evaluation is **grouped leave-one-kinase-out** (no kinase in both train and test),
so the number generalizes to unseen kinases and does not leak identity.

| Approach | Grouped LOKO AUROC |
|---|---|
| Pocket **sequence** classifier | 0.62 (leaks identity) |
| DFG-to-αC **distance** alone (training-free) | 0.844 |
| **Both geometric descriptors** | **0.834** (acc 0.80) |

![Figure 1](paper/figure1_descriptors.png)

*DFG-in (blue) separates from DFG-out (orange) along the DFG-to-αC distance; a
grouped classifier generalizes across kinases at 0.834 AUROC.*

## Release

Entry points via `make` (or run the `python` commands directly on Windows):

```bash
make install     # deps
make analyze ARGS="2hyy"
make benchmark   # grouped leave-one-kinase-out -> checkpoints/kinase_benchmark.json + splits.csv
make atlas       # offline atlas
make verify      # smoke test: analyze + benchmark + atlas must all succeed
```

Attached artifacts:

| Artifact | Path |
|---|---|
| Atlas (parquet / csv / html) | `research/kinase/atlas/` |
| Benchmark splits (LOKO folds + OOF preds) | `research/kinase/paper/splits.csv` |
| Figure 1 + descriptor table | `research/kinase/paper/` |
| DFG model checkpoint | `hyaline/kinase/dfg_model.json` |
| Abstract | `research/kinase/paper/ABSTRACT.md` |

A clean checkout runs install → analyze → benchmark → atlas with nothing failing
(`make verify`).

## Findings (honest, command-backed)

Every number is tagged **REAL** (real KLIFS data), **SYNTHETIC** (model on invented
data — a mechanism check, not real-molecule performance), or **LEAKY** (inflated by
an evaluation that lets the same kinase appear in train and test).

| Metric | Value | Tag | Command |
|---|---|---|---|
| Hybrid RF / MLP regression | R² 0.964 / 0.947 | SYNTHETIC | `python scripts/hybrid_kinase_model.py` |
| DFG × drug-size importance | 0.56 (hybrid gen.) **vs 0.155** (ablation gen.) | SYNTHETIC | `python scripts/hybrid_kinase_model.py` |
| Sequence-only regression | R² 0.015 | SYNTHETIC | `python scripts/kinase_ablation.py` |
| Static / Spiking EGNN | R² −0.064 / −0.089 (did not converge; spiking no better, p 0.125) | SYNTHETIC | `python scripts/kinase_ablation.py` |
| DFG classifier, sequence, **ungrouped** | acc 0.893, AUROC 0.946 | LEAKY | `python scripts/kinase_audit.py` |
| DFG classifier, sequence, **grouped LOKO** | acc 0.711, **AUROC 0.62** | REAL | `python scripts/kinase_audit.py` |
| DFG classifier, **2 geometric descriptors, grouped LOKO** | acc 0.80, **AUROC 0.834** | REAL | `python scripts/kinase_descriptors.py` |
| Known-drug Type I/II validation | 5 / 6 correct | REAL | `python scripts/klifs_validation.py` |

**Two lessons.** (1) *Sequence features leak kinase identity*: ungrouped AUROC 0.946
collapses to 0.62 under grouped LOKO — the 85-residue pocket is a kinase ID badge.
(2) *Interpretable geometry wins without leakage*: a training-free physical
descriptor (0.844) beats the leakage-corrected sequence model (0.62). Known-drug
validation confirms the data is sound (5/6). The regression/EGNN numbers are
synthetic (two non-comparable generators) and demonstrate mechanism only.

## Codebase

| Path | Purpose |
|------|---------|
| `hyaline/kinase/analyze.py` | **`analyze`** — annotate one structure (DFG/αC, descriptors, inhibitor class, provenance) |
| `hyaline/kinase/pocket_extract.py` | Extract the 85-residue pocket from an arbitrary PDB / AlphaFold model |
| `hyaline/kinase/pymol_export.py` | Write an annotated PyMOL session (`--pymol`) |
| `hyaline/kinase/dfg_model.json` | Dependency-light logistic DFG model (real descriptors, grouped AUROC 0.834) |
| `scripts/analyze_kinase.py` | CLI for `analyze` |
| `scripts/kinase_descriptors.py` | Geometric descriptors → grouped LOKO + Figure 1 |
| `scripts/kinase_audit.py` | Reproducibility audit (sequence classifier, leaky vs grouped) |
| `scripts/build_kinase_atlas.py` | Atlas builder (per-kinase table + offline HTML) |
| `scripts/klifs_validation.py` | Known-drug Type I/II validation |
| `research/kinase/colab/` | Colab notebook |
| `research/kinase/atlas/` | Atlas artifacts (parquet/csv/html) |
| `research/kinase/paper/` | Figure 1 + descriptor CSV |

## Limitations

- Regression results (hybrid, ablation) are **synthetic** — two non-comparable
  generators; mechanism, not real-molecule accuracy.
- The spiking-EGNN "synchronization" idea shows **no measurable benefit** over a
  static EGNN.
- `train_screening_model_real_features.py` still uses **mock compound features**.
- Conformation-paired ΔpKi (same ligand, both DFG states) is scarce in real data.
- Local-PDB pocket extraction requires the **kinase name** and a KLIFS reference
  for that kinase.

## Roadmap (Ayman's phase plan)

Invariants across all phases: record whether each input is **experimental or
predicted**, **determinism**, and **every claim backed by a command**.

- **Phase 0/1** — ✅ `analyze` (DFG/αC + fingerprint + inhibitor class, provenance),
  experimental and predicted, incl. arbitrary PDB / AlphaFold pocket extraction.
- **Phase 2** — ✅ defensible grouped leave-one-kinase-out number (AUROC 0.834),
  reproduced by command.
- **Phase 3** — ✅ atlas (318 kinases, offline HTML + parquet). *Next:* grow
  descriptor coverage beyond the current 12 kinases.
- **Phase 4** — ✅ `--pymol` export + Colab notebook + README rebuilt around
  install / quickstart / atlas / benchmark / Colab.
- **Phase 5** — ✅ release: `make verify` runs install → analyze → benchmark →
  atlas with nothing failing; artifacts attached (parquet/CSV/HTML/splits/model);
  the atlas, AlphaFold annotation, inhibitor-class output, and the
  leave-one-kinase-out number (0.834) are folded into
  `research/kinase/paper/ABSTRACT.md`.
