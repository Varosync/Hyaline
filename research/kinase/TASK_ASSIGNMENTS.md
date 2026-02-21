# Hyaline Kinase Project — Task Assignments

**Based on team backgrounds (corrected):**
- **Luca**: Hardcore AI/MLOps — Master's in AI, Novartis-scale PyTorch pipelines, Vision Transformer optimization
- **Sasha**: Elite wet-lab biologist → CS — 2.5 years MINFLUX, PCR, immunostaining; junior coding (intro Python/Java)
- **Emily**: Freshman — Data cleaning, Pandas/Scikit-Learn baselines, literature, technical writing

---

## Team Structure & Ownership

| Person | Primary Strength | Owns |
|--------|------------------|------|
| **Luca** | AI/ML, distributed systems, production pipelines | Feature extraction, model pipeline, virtual screening, MLOps |
| **Sasha** | Wet-lab, molecular biology | Assay setup, biological curation, hit validation |
| **Emily** | Data prep, baselines, documentation | KLIFS cleaning, ablation scripts, literature, docs |

---

## LUCA — AI/MLOps Engineer

### 1. Cross-Kinase Feature Extraction Pipeline
**Goal:** Build kinase-agnostic, physics-based features so the model generalizes across kinases instead of memorizing sequences.

**Tasks:**
- Pull pocket structures from KLIFS API (`structure_get_pocket` for MOL2)
- Compute per-structure: pocket volume, electrostatic surface potential, hydrophobicity
- Use RDKit, MDAnalysis, or PyMOL scripting for 3D calculations
- Add geometric features: DFG–C-helix distance, hinge–activation loop angle (from KLIFS conformational API)
- Output: CSV/Parquet of structure_id → feature vector
- **Deliverable:** `hyaline/features/pocket_features.py` + feature matrix for all KLIFS structures

**Why Luca:** Needs 3D geometry, scripting, and integration with our stack. His MLOps background fits batch feature extraction.

---

### 2. Hybrid Model + Virtual Screening Pipeline
**Goal:** Connect the hybrid model to large compound libraries and produce a ranked hit list.

**Tasks:**
- Integrate Luca's new features with the existing hybrid model
- Implement virtual screening: ZINC / Enamine "In-Stock" API or download
- Pipeline: (kinase structure + features) × (compound SMILES → fingerprints) → model → score
- Filter: price ≤ $30/mg
- Output: Ranked list of predicted Type II (DFG-out) inhibitors
- Use batching and optional GPU for throughput
- **Deliverable:** `scripts/virtual_screen.py` + documentation

**Why Luca:** Pipeline design, scaling, integration. Similar to what he did for 5M images at Novartis.

---

### 3. Model & Evaluation Infrastructure
**Tasks:**
- Retrain hybrid model using new pocket features; compare to current baseline
- Run leave-one-kinase-out evaluation
- Log experiments (weights, metrics, configs)
- **Deliverable:** Reproducible training + evaluation scripts

**Why Luca:** Standard ML experiment management and evaluation.

---

## SASHA — Wet-Lab Biologist (Transitioning to CS)

### 1. Wet-Lab Assay Setup & Execution
**Goal:** Validate predicted hits experimentally.

**Tasks:**
- Choose assay: Thermal shift (DSF) if kinase is available; yeast rescue if not
- For DSF: find protocols (kinase + SYPRO Orange), list reagents and costs
- For yeast: design BRAF V600E rescue assay, identify strain and cloning strategy
- Run assays on top 5–10 compounds from Luca's pipeline
- Record: compound ID, concentration, result (bind / no bind)
- **Deliverable:** Assay protocol doc + results table (`data/wetlab_validation.csv`)

**Why Sasha:** Her wet-lab experience (PCR, immunostaining, MINFLUX) is a strong fit.

---

### 2. Gold-Standard Inhibitor Curation
**Goal:** Curated dataset of known Type I/II inhibitors for model validation.

**Tasks:**
- Search literature for Type I vs Type II inhibitors (PubMed, reviews)
- For each: drug name, kinase(s), type, preferred conformation, PDB if known
- Match to KLIFS: structure_ID, PDB code, DFG state
- Check KLIFS DFG annotations against literature
- Store in structured format for Emily/Luca
- **Deliverable:** `data/known_inhibitors_curated.csv` + short methods note

**Why Sasha:** Requires strong mol-bio interpretation and literature; minimal coding (spreadsheets).

---

### 3. Biological Validation of Features
**Goal:** Check that Luca's features match known biology.

**Tasks:**
- When features are ready, compare DFG-in vs DFG-out: pocket volume, hydrophobicity, etc.
- Confirm DFG-out has larger back pocket; known hotspots (gatekeeper, hinge) show sensible values
- Note any inconsistencies for follow-up
- **Deliverable:** Short report: “Feature validation — biological sanity check”

**Why Sasha:** Needs mol-bio judgment; minimal coding (can use Emily’s plots).

---

### 4. Hit Triage (When Pipeline Output Is Ready)
**Tasks:**
- Review top predicted hits for obvious issues (known PAINS, toxicity, promiscuity)
- Cross-check with PubChem/ChEMBL when possible
- Provide brief notes to prioritize compounds for wet-lab testing
- **Deliverable:** Triage notes for each candidate batch

**Why Sasha:** Biology-focused review; can be done in spreadsheets or simple forms.

---

## EMILY — Freshman (Data + Baselines + Documentation)

### 1. KLIFS Data Cleaning & Preprocessing
**Goal:** Clean, consistent datasets for training and validation.

**Tasks:**
- Download/refresh KLIFS structures list (kinase_ID, structure_ID, PDB, DFG, pocket, resolution, etc.)
- Normalize fields (missing values, types, duplicates)
- Link to bioactivity: match ligand PDB codes to ChEMBL where possible
- Create train/val/test splits (e.g. by kinase or by time)
- **Deliverable:** `data/klifs_cleaned.parquet` (or CSV) + short data dictionary

**Why Emily:** Data cleaning and preprocessing fit her skills.

---

### 2. Baseline Model Scripts
**Goal:** Reproducible baselines for benchmarking.

**Tasks:**
- RF on sequence-only features (one-hot pocket)
- RF on structure features (current + new from Luca)
- GradientBoosting on same features
- Run 5-fold CV and record R², accuracy
- **Deliverable:** `scripts/run_baselines.py` + `results/baseline_comparison.csv`

**Why Emily:** Scikit-Learn, Pandas, simple evaluation loops.

---

### 3. Literature Review & Technical Writing
**Tasks:**
- Short review: Type I vs Type II inhibitors, DFG-in vs DFG-out
- Compile KLIFS + ChEMBL data sources and usage
- Document data pipeline: KLIFS → features → model → screening
- Maintain README and user-facing docs
- **Deliverable:** `research/kinase/LITERATURE_REVIEW.md`, updated README

**Why Emily:** Technical writing and literature review; she can work from notes from Sasha/Luca.

---

### 4. Support for Sasha's Curation
**Tasks:**
- Convert Sasha’s curation into structured CSVs if needed
- Create templates for adding new compounds
- Help with simple validation checks (e.g. duplicates, missing fields)

**Why Emily:** Bridges Sasha’s curation and Luca’s data requirements.

---

## Handoffs & Dependencies

```
Emily (Data)  ──►  Luca (Features + Model)
     │                      │
     │                      ▼
     │              Luca (Virtual Screen)
     │                      │
     ▼                      ▼
Sasha (Curation)  ──►  Hit List  ──►  Sasha (Wet-Lab)
```

| From | To | Artifact |
|------|----|----------|
| Emily | Luca | Clean KLIFS + bioactivity data |
| Luca | Sasha | Ranked hit list (top N, price-filtered) |
| Sasha | Emily | Curation tables for validation |
| Sasha | Luca | Feature sanity-check feedback |
| Emily | All | Documentation, baselines |

---

## Sprint 1 (Weeks 1–2) — Foundation

| Person | Sprint 1 Focus |
|--------|----------------|
| **Luca** | KLIFS structure download + pocket feature extraction (volume, electrostatics, hydrophobicity) |
| **Sasha** | Literature search + gold-standard inhibitor curation (50+ compounds) |
| **Emily** | KLIFS data cleaning, baseline RF scripts, data dictionary |

---

## Sprint 2 (Weeks 3–4) — Integration

| Person | Sprint 2 Focus |
|--------|----------------|
| **Luca** | Integrate new features into hybrid model; ZINC/Enamine virtual screening pipeline |
| **Sasha** | Assay protocol research (DSF or yeast); biological validation of Luca's features |
| **Emily** | Baseline comparison report; literature review draft; documentation updates |

---

## Sprint 3 (Weeks 5–6) — Validation

| Person | Sprint 3 Focus |
|--------|----------------|
| **Luca** | Run full virtual screen; produce ranked hit list; leave-one-kinase-out evaluation |
| **Sasha** | Run wet-lab assay on top 5–10 hits; hit triage; final curation pass |
| **Emily** | Final documentation; baseline vs hybrid comparison; handoff materials |

---

## Success Metrics by Role

| Person | Success = |
|--------|-----------|
| **Luca** | (1) Feature pipeline runs on all KLIFS structures, (2) Virtual screen completes with price filter, (3) Leave-one-kinase-out R² improves vs current baseline |
| **Sasha** | (1) Gold-standard curation complete, (2) Assay run on ≥5 compounds, (3) Feature validation report done |
| **Emily** | (1) Clean data delivered, (2) Baselines reproduced, (3) Documentation updated and usable |

---

## What Each Person Should NOT Do

| Person | Avoid |
|--------|-------|
| **Luca** | Manual literature curation; wet-lab work; long manual data cleaning |
| **Sasha** | Heavy feature extraction; MLOps; ZINC/Enamine API integration |
| **Emily** | Complex model architecture; wet-lab; production pipeline design |

---

*Last updated: Jan 2026*
