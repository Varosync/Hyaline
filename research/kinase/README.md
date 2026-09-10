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
pip install -e . --no-deps        # optional: enables the `hyaline analyze` command
```

## Quickstart — annotate a structure

Once installed, `analyze` is a first-class CLI subcommand (dependency-light — no
torch is imported). The `scripts/analyze_kinase.py` invocation below is exactly
equivalent and needs no install.

```bash
hyaline analyze 2hyy                                          # PDB code (KLIFS)
hyaline analyze --klifs-id 1081                               # KLIFS structure_ID
hyaline analyze --pdb-file model.pdb --kinase ABL1 --provenance predicted   # any PDB / AlphaFold
hyaline analyze 2hyy --pymol out.pml                         # + annotated PyMOL session

# equivalent without installing the package:
python scripts/analyze_kinase.py 2hyy
```

Output (`2hyy`, imatinib-bound ABL1) — correctly called **DFG-out / Type II**:

```json
{
  "schema_version": "1.1",
  "identifier": "2hyy", "source": "pdb", "provenance": "experimental",
  "dfg_achelix_distance_A": 12.237, "hinge_activation_angle_deg": 59.668,
  "dfg_call": "DFG-out", "dfg_confidence": 0.674, "dfg_driver": "dfg_achelix_distance_A",
  "achelix_ke_distance_A": 10.851, "achelix_state": "aC-in", "achelix_confidence": 0.977,
  "achelix_source": "klifs_annotation", "achelix_driver": "achelix_ke_distance_A",
  "inhibitor_class": "Type II",
  "inhibitor_rationale": "DFG-out exposes the allosteric pocket engaged by Type II inhibitors"
}
```

The **αC-helix state** (aC-in / aC-out) is computed geometrically from the
β3-Lys(17)–αC-Glu(24) Cα salt-bridge distance (`achelix_model.json`; grouped
leave-one-kinase-out **AUROC 0.901**), so it — with a confidence — is returned for
**any** input, including AlphaFold models that have no KLIFS annotation. When a
KLIFS αC annotation is present it stays authoritative (`achelix_source:
klifs_annotation`) and the independent geometric confidence is still reported; a
disagreement is flagged in `warnings`.

For an arbitrary local PDB (crystal or AlphaFold model), the 85-residue pocket is
extracted by aligning to a KLIFS reference for the given kinase. `--pymol` writes a
session that loads the structure, colours the DFG motif and αC-helix, and labels
the call.

### Demo — 5 experimental + 2 AlphaFold (committed evidence)

```bash
python scripts/demo_analyze_batch.py     # or: make demo
```

Annotates five experimental KLIFS structures (ABL1, EGFR, BRAF, KIT, SRC) and two
AlphaFold models (ABL1, EGFR), tags each with provenance, and validates every
result against the schema. Outputs are committed under
[`research/kinase/demo/`](demo/demo_summary.md). Note the honest contrast: the
imatinib-bound ABL1 crystal (`2hyy`) is **DFG-out / Type II**, while the AlphaFold
ABL1 model is **DFG-in / Type I** — AlphaFold predicts the active conformation, and
provenance makes that explicit rather than silent.

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

Each row carries: a `provenance` tag (all `experimental` here — KLIFS crystal
structures; recorded explicitly, never implicit), accessible DFG states, structure
counts, known Type I / Type II inhibitors, a **Type-II-opportunity score**
(heuristic: well-studied kinases that reach DFG-out but have few known Type II
inhibitors rank high — e.g. HGK, LRRK2), and a geometric descriptor fingerprint
where computed. Build: **318 kinases, 13,325 structures**; per-kinase counts
reconcile against KLIFS by construction.

## Benchmark — the defensible number

```bash
python scripts/kinase_descriptors.py   # geometric descriptors + grouped LOKO + Figure 1
python scripts/kinase_benchmark.py     # the defensible number -> checkpoints/ + splits.csv
python scripts/kinase_audit.py         # sequence-classifier audit (leaky vs grouped)
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
| DFG / αC model checkpoints | `hyaline/kinase/dfg_model.json`, `hyaline/kinase/achelix_model.json` |
| Output schema (v1.1) | `hyaline/kinase/analysis_schema.json` |
| Demo evidence (5 experimental + 2 AlphaFold) | `research/kinase/demo/` |
| Abstract | `research/kinase/paper/ABSTRACT.md` |

A clean checkout runs install → analyze → benchmark → atlas with nothing failing
(`make verify`).

## Findings — the real, command-backed result

Each number below is reproduced by a command in this branch. Numbers are tagged
**REAL** (real KLIFS data) or **LEAKY** (inflated by an evaluation that lets the same
kinase appear in train and test — shown to make the leakage explicit).

| Metric | Value | Tag | Command |
|---|---|---|---|
| DFG classifier, sequence, **ungrouped** | acc 0.893, AUROC 0.946 | LEAKY | `python scripts/kinase_audit.py` |
| DFG classifier, sequence, **grouped LOKO** | acc 0.711, **AUROC 0.62** | REAL | `python scripts/kinase_audit.py` |
| DFG classifier, **2 geometric descriptors, grouped LOKO** | acc 0.80, **AUROC 0.834** | REAL | `python scripts/kinase_descriptors.py` |
| DFG-to-αC **distance alone** (training-free) | **AUROC 0.844** | REAL | `python scripts/kinase_descriptors.py` |
| Benchmark (grouped LOKO, deterministic, offline) | **AUROC 0.834** | REAL | `python scripts/kinase_benchmark.py` |
| αC-in/out from β3-Lys–αC-Glu distance, **grouped LOKO** | acc 0.917, **AUROC 0.901** | REAL | `python scripts/calibrate_achelix.py` |

**Two lessons.** (1) *Sequence features leak kinase identity*: ungrouped AUROC 0.946
collapses to 0.62 under grouped LOKO — the 85-residue pocket is a kinase ID badge.
(2) *Interpretable geometry wins without leakage*: a training-free physical
descriptor (0.844) beats the leakage-corrected sequence model (0.62).

### Archived experiments (results only; generator scripts not in this branch)

Earlier exploratory runs are kept for the record but are **not reproducible from this
branch** — their generator scripts predate `kinase-real-descriptors` and are not
included. Raw outputs sit in `checkpoints/`. These are **SYNTHETIC** (models fit on
invented data): a mechanism check, never real-molecule performance.

- Hybrid RF / MLP regression R² 0.964 / 0.947, and sequence-only regression R² 0.015
  — `checkpoints/hybrid_results.json`, `checkpoints/kinase_ablation.json`.
- Static / Spiking EGNN R² −0.064 / −0.089 (did not converge; spiking no better)
  — `checkpoints/kinase_ablation.json`. Negative result kept deliberately.
- Per-kinase DFG-in/out ligand inventory (ABL1, EGFR, BRAF, SRC, KIT)
  — `checkpoints/klifs_validation.json`.

## Codebase

| Path | Purpose |
|------|---------|
| `hyaline/kinase/analyze.py` | **`analyze`** — annotate one structure (DFG/αC, descriptors, inhibitor class, provenance) |
| `hyaline/kinase/pocket_extract.py` | Extract the 85-residue pocket from an arbitrary PDB / AlphaFold model |
| `hyaline/kinase/pymol_export.py` | Write an annotated PyMOL session (`--pymol`) |
| `hyaline/kinase/dfg_model.json` | Dependency-light logistic DFG model (real descriptors, grouped AUROC 0.834) |
| `hyaline/kinase/achelix_model.json` | Logistic αC-in/out model (β3-Lys–αC-Glu distance, grouped AUROC 0.901) |
| `hyaline/kinase/analysis_schema.json` | Fixed output schema (v1.1); every `analyze` result is validated against it |
| `hyaline/cli.py` | `hyaline analyze` subcommand (dependency-light wrapper over `analyze`) |
| `scripts/analyze_kinase.py` | Standalone CLI for `analyze` (no install needed) |
| `scripts/kinase_descriptors.py` | Geometric descriptors → grouped LOKO + Figure 1 |
| `scripts/calibrate_achelix.py` | Calibrate the αC model (grouped LOKO) from cached pockets |
| `scripts/kinase_benchmark.py` | The defensible number (grouped LOKO) → `checkpoints/` + `splits.csv` |
| `scripts/demo_analyze_batch.py` | Demo: 5 experimental + 2 AlphaFold, schema-validated |
| `scripts/kinase_audit.py` | Reproducibility audit (sequence classifier, leaky vs grouped) |
| `scripts/build_kinase_atlas.py` | Atlas builder (per-kinase table + offline HTML) |
| `research/kinase/colab/` | Colab notebook |
| `research/kinase/atlas/` | Atlas artifacts (parquet/csv/html) |
| `research/kinase/paper/` | Figure 1 + descriptor CSV |

## Limitations

- The archived regression results (hybrid, ablation) are **synthetic** — two
  non-comparable generators; mechanism, not real-molecule accuracy. Not reproducible
  from this branch (see *Archived experiments*).
- The archived spiking-EGNN "synchronization" idea shows **no measurable benefit**
  over a static EGNN.
- Compound-level screening features remain **mock** — no real-molecule affinity model
  ships in this branch; the kinase tool scores conformational state, not binding.
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
- **Phase 4** — ✅ `hyaline analyze` CLI subcommand + `--pymol` export + Colab
  notebook + README (root and kinase) rebuilt around install / quickstart / atlas /
  benchmark / Colab.
- **Phase 5** — ✅ release: `make verify` runs install → analyze → benchmark →
  atlas with nothing failing; artifacts attached (parquet/CSV/HTML/splits/model);
  the atlas, AlphaFold annotation, inhibitor-class output, and the
  leave-one-kinase-out number (0.834) are folded into
  `research/kinase/paper/ABSTRACT.md`.
