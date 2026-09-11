# The Hyaline Kinase branch, explained end to end

A plain-language walkthrough of what this branch is, why each piece exists, and
the reasoning behind the design.

## Part 1 — The problem

Kinases are molecular on/off switches and the biggest family of cancer-drug
targets. The same kinase, with the same sequence, can fold into different 3D
shapes. The key switch is the **DFG** motif (Asp-Phe-Gly):

- **DFG-in** = active shape, flat pocket -> **Type I** inhibitors bind here.
- **DFG-out** = inactive shape, opens a back-pocket -> **Type II** inhibitors bind here.

The most useful thing for drug design is knowing *which shape* a structure is in,
because it decides which drug class can bind. Sequence alone can't tell you (same
sequence, different shape) — you must read the **geometry**. That is this branch's
purpose. The parent project (`master`) does the analogous thing for GPCRs; this
branch reuses that skeleton and builds the kinase capability on top.

## Part 2 — The core design decision

**Use interpretable physical measurements, not a black box.** Reasoning:

1. **Leakage.** Feeding the pocket *sequence* into a classifier scores ~0.95
   AUROC — but it cheats: the 85-residue pocket is a fingerprint of *which kinase*
   it is. Tested honestly (never the same kinase in train and test), that 0.95
   collapses to **0.62**. Sequence leaks identity.
2. **Geometry can't leak.** A distance/angle measured off the atoms is a real
   property of that structure. Two hand-chosen descriptors:
   - **DFG-to-alphaC distance** (opens up in DFG-out)
   - **hinge / activation-loop angle** at the catalytic lysine

   Distance alone (training-free) hits **0.844** — beating the leakage-corrected
   sequence model (0.62). Thesis: honest, simple geometry wins. No GPU, no
   retraining, every claim backed by a runnable command.

## Part 3 — The pieces

### Engine — `hyaline/kinase/`
- `analyze.py` — heart: computes descriptors, calls the models, returns DFG +
  alphaC states with confidence, geometric fingerprint, inhibitor class with the
  driving reason, and a provenance tag. Dependency-light (NumPy + requests).
- `dfg_model.json` — the "model": 4 numbers (logistic over 2 descriptors).
  AUROC 0.834, n=621.
- `achelix_model.json` — alphaC model: 1 weight over the Lys17-Glu24 salt-bridge
  distance. AUROC 0.901, n=664.
- `pocket_extract.py` — bridge to AlphaFold: hand-written PDB parser +
  hand-written Needleman-Wunsch alignment to map any structure onto KLIFS's
  85-residue pocket numbering (no Biopython).
- `pymol_export.py` — hand-writes a `.pml` script that opens the structure with
  DFG, alphaC, and the call labeled (no pymol dependency; it generates commands).
- `analysis_schema.json` — fixed output contract (v1.1) every result validates against.

### CLI / scripts
- `hyaline analyze ...` and `scripts/analyze_kinase.py` — run the analyzer.
- `scripts/kinase_descriptors.py` — descriptors from raw KLIFS coords.
- `scripts/kinase_benchmark.py` — reproduce the defensible number, offline.
- `scripts/kinase_audit.py` — leaky 0.95 vs honest 0.62, side by side.
- `scripts/calibrate_achelix.py` — fits the alphaC model.
- `scripts/demo_analyze_batch.py` — 5 experimental + 2 AlphaFold, committed evidence.

### Outputs — `research/kinase/`
- `atlas/` — offline, browsable map of 318 kinases / 13,325 structures
  (searchable table + DFG-distance vs hinge-angle scatter, colored by state,
  data baked into the HTML). Carries per-kinase states, known Type I/II drugs, and
  a **Type-II-opportunity score** (well-studied kinases that reach DFG-out but lack
  Type II drugs — i.e. drug-discovery openings).
- `data/` — descriptor CSV + benchmark splits (the CSVs the code reads).
- `demo/` — committed proof the tool runs on real + predicted structures.
- `colab/` — a notebook to run it in the cloud in one pass.

### The benchmark method — why it's trustworthy
**Grouped leave-one-kinase-out (LOKO):** hold out an entire kinase, train on the
rest, test on the held-out one, repeat for all. No kinase is ever in both train
and test — the only way to know it generalizes to unseen kinases. This is what
kills the leaky 0.95 and yields the honest 0.834.

## Part 4 — What we built/fixed (this branch, incl. the audit pass)

1. **`hyaline analyze` CLI** — wired the DFG logic into a real command; made the
   package importable without PyTorch so a kinase-only install works.
2. **Fixed schema + validation** — every output validated against
   `analysis_schema.json`; fails loudly on violation.
3. **Geometric alphaC state with confidence** — the alphaC in/out call is now
   computed from the Lys17-Glu24 salt-bridge distance on *any* input (incl.
   AlphaFold), calibrated at AUROC 0.901. KLIFS annotation stays authoritative
   when present; the independent geometric confidence is still reported.
4. **Atlas provenance column** — every row tags experimental vs predicted.
5. **README honesty** — removed dead-command references; archived the synthetic
   experiments in a clearly labeled section.
6. **Committed AlphaFold evidence** — the demo shows the honest contrast:
   imatinib-bound ABL1 crystal reads DFG-out, but the AlphaFold ABL1 model reads
   DFG-in (AlphaFold predicts the active shape) — provenance makes that visible.

## Part 5 — What is/ isn't from scratch

- **Our code, hand-written (often dependency-free):** the descriptor pipeline
  (parser, geometry, calibration, LOKO eval), the AlphaFold pocket extraction
  (incl. a hand-coded Needleman-Wunsch aligner), the PyMOL `.pml` generator, and
  the atlas (builder, opportunity score, interactive page with hand-drawn SVG
  scatter — no charting library).
- **Built on (not ours):** the structural data (KLIFS), AlphaFold models
  (AlphaFold DB), the underlying structural-biology concepts (DFG/alphaC states,
  the K-E salt bridge, KLIFS's 85-residue pocket numbering), the Needleman-Wunsch
  *algorithm*, PyMOL itself, and standard libraries (NumPy, scikit-learn, pandas).
- **The actual novelty:** turning known biology into precise computable numbers on
  the KLIFS framework, and the honest demonstration that leak-proof geometry
  (0.834) beats the leaky sequence model (0.62).

## Part 6 — The three invariants

1. **Provenance** — every output records experimental vs predicted.
2. **Reproducibility** — same input, same output; frozen JSON models, deterministic
   benchmark.
3. **Command-backed honesty** — every number has a command that regenerates it, and
   numbers are tagged REAL / LEAKY / (archived) SYNTHETIC.

**One line:** a small, honest, GPU-free tool that reads a kinase's shape from its
geometry, names the drug class that fits with a calibrated confidence, works on
real and AlphaFold structures, maps every human kinase, and backs every claim with
a reproducible number.
