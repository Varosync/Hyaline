# Hyaline: 6-Week Validation Sprints

**Task Database** — For Notion import or team reference  
**Last updated:** Jan 2026

---

## Environment: Branch & S3

### Git Branch: `kinase-v1`

This version of Hyaline is developed on **`kinase-v1`** (not `master`).

**What’s on `kinase-v1`:**
- Kinase conformational selectivity code (KLIFS, hybrid model, virtual screening)
- `hyaline/data/klifs_loader.py`, `hyaline/features/`, `hyaline/models/kinase_binding.py`
- `scripts/hybrid_kinase_model.py`, `scripts/kinase_ablation.py`, `scripts/klifs_validation.py`, etc.
- TF activation / CryptoSite / GPCR scripts (legacy)

**Important:** Check out `kinase-v1` before starting work:
```bash
git checkout kinase-v1
git pull origin kinase-v1  # when pushed
```

---

### S3 Bucket

Data is stored in S3, not in the repo.

**Bucket:** `s3://amzn-s3-proteinbucket`  
**Kinase prefix:** `hyaline/kinase/`  
**Full path:** `s3://amzn-s3-proteinbucket/hyaline/kinase/`

**Contents:**
| Folder | Contents | Size |
|--------|----------|------|
| `klifs_cache/` | KLIFS API responses, key_kinases.json, conformational data | ~12 MB |
| `data/klifs_cache/` | Processed KLIFS data | ~12 MB |
| `checkpoints/` | Model results (hybrid_results.json, etc.) | ~11 MB |

**Upload (after local changes):**
```bash
./scripts/upload_kinase_to_s3.sh
```

**Download (for fresh setup):**
```bash
aws s3 sync s3://amzn-s3-proteinbucket/hyaline/kinase/ ./ --region us-east-1
```

**Alternative bucket (if used):** `s3://hyaline-kinase-data/` — check `scripts/upload_to_s3.sh` for which is active.

---

---

# Kanban View (Grouped by Status)

## To Do
| Task | Assignee | Sprint |
|------|----------|--------|
| Extract 3D Physics Features | Luca | Sprint 1 |
| Gold-Standard Inhibitor Curation | Sasha | Sprint 1 |
| KLIFS Data Cleaning | Emily | Sprint 1 |
| Hybrid Model & Virtual Screen Pipeline | Luca | Sprint 2 |
| Assay Protocol Design & Feature Validation | Sasha | Sprint 2 |
| Baseline Model Scripts | Emily | Sprint 2 |
| Order Lab Supplies | PM | Sprint 2 |
| Full Screen & Leave-One-Out Eval | Luca | Sprint 3 |
| Hit Triage & Data Analysis | Sasha | Sprint 3 |
| Literature Review & Docs | Emily | Sprint 3 |
| Execute Wet-Lab Assay | PM | Sprint 3 |

## In Progress
*(Empty at sprint start — update as work begins)*

## Blocked
*(Empty — add tasks blocked by dependencies)*

## Done
*(Empty at sprint start — move here when complete)*

---

---

# Table View (Sorted by Sprint)

| Task Name | Assignee | Status | Sprint | Deliverable |
|-----------|----------|--------|--------|-------------|
| Extract 3D Physics Features | Luca | To Do | Sprint 1 | hyaline/features/pocket_features.py + feature matrix |
| Gold-Standard Inhibitor Curation | Sasha | To Do | Sprint 1 | data/known_inhibitors_curated.csv |
| KLIFS Data Cleaning | Emily | To Do | Sprint 1 | data/klifs_cleaned.parquet + data dictionary |
| Hybrid Model & Virtual Screen Pipeline | Luca | To Do | Sprint 2 | scripts/virtual_screen.py |
| Assay Protocol Design & Feature Validation | Sasha | To Do | Sprint 2 | Assay protocol doc + Feature biological sanity check |
| Baseline Model Scripts | Emily | To Do | Sprint 2 | scripts/run_baselines.py + baseline_comparison.csv |
| Order Lab Supplies | PM | To Do | Sprint 2 | Target kinase + SYPRO Orange secured at Frontier Tower |
| Full Screen & Leave-One-Out Eval | Luca | To Do | Sprint 3 | Reproducible training scripts + final ranked hit list |
| Hit Triage & Data Analysis | Sasha | To Do | Sprint 3 | Triage notes + final wet-lab analysis graphs |
| Literature Review & Docs | Emily | To Do | Sprint 3 | LITERATURE_REVIEW.md + README update |
| Execute Wet-Lab Assay | PM | To Do | Sprint 3 | Physical qPCR experiment completed |

---

---

# Task Details (Execution Instructions)

---

## 1. Extract 3D Physics Features

| Property | Value |
|----------|-------|
| **Task Name** | Extract 3D Physics Features |
| **Assignee** | Luca |
| **Status** | To Do |
| **Sprint** | Sprint 1 |
| **Deliverable** | `hyaline/features/pocket_features.py` + feature matrix |

**Execution instructions:**
- Pull pocket structures from KLIFS API (`structure_get_pocket` for MOL2).
- Compute pocket volume, electrostatic surface potential, and hydrophobicity using RDKit/PyMOL.
- Add geometric features (DFG–C-helix distance, hinge–activation loop angle) from KLIFS conformational API.
- Output CSV/Parquet: structure_id → feature vector.
- Run on all KLIFS structures; optionally upload results to S3 `hyaline/kinase/`.

---

## 2. Gold-Standard Inhibitor Curation

| Property | Value |
|----------|-------|
| **Task Name** | Gold-Standard Inhibitor Curation |
| **Assignee** | Sasha |
| **Status** | To Do |
| **Sprint** | Sprint 1 |
| **Deliverable** | `data/known_inhibitors_curated.csv` |

**Execution instructions:**
- Search literature for 50+ Type I/II inhibitors (PubMed, reviews).
- Match to KLIFS structure IDs (structure_ID, PDB code, DFG state).
- Check DFG annotations against literature.
- Store in structured format for Emily and Luca.

---

## 3. KLIFS Data Cleaning

| Property | Value |
|----------|-------|
| **Task Name** | KLIFS Data Cleaning |
| **Assignee** | Emily |
| **Status** | To Do |
| **Sprint** | Sprint 1 |
| **Deliverable** | `data/klifs_cleaned.parquet` + data dictionary |

**Execution instructions:**
- Download KLIFS structures via API or from S3 `hyaline/kinase/klifs_cache/`.
- Normalize missing values, types, and duplicates.
- Link ligand PDB codes to ChEMBL bioactivity where possible.
- Create train/val/test splits (e.g. by kinase).
- Document fields in a short data dictionary.

---

## 4. Hybrid Model & Virtual Screen Pipeline

| Property | Value |
|----------|-------|
| **Task Name** | Hybrid Model & Virtual Screen Pipeline |
| **Assignee** | Luca |
| **Status** | To Do |
| **Sprint** | Sprint 2 |
| **Deliverable** | `scripts/virtual_screen.py` |

**Execution instructions:**
- Integrate new pocket features with the hybrid model.
- Implement ZINC or Enamine API/database for compound screening.
- Filter out any compound costing >$30/mg.
- Output ranked list of predicted Type II (DFG-out) inhibitors.
- Use batching and GPUs for throughput.

---

## 5. Assay Protocol Design & Feature Validation

| Property | Value |
|----------|-------|
| **Task Name** | Assay Protocol Design & Feature Validation |
| **Assignee** | Sasha |
| **Status** | To Do |
| **Sprint** | Sprint 2 |
| **Deliverable** | Assay protocol doc + Feature biological sanity check |

**Execution instructions:**
- Design Thermal Shift (DSF) protocol: kinase + SYPRO Orange concentrations, buffers, plate layout.
- Compare Luca’s extracted features (DFG-in vs DFG-out) to ensure they match known biology (e.g. back pocket volume larger in DFG-out).
- Document protocol for PM; write short feature validation report.

---

## 6. Baseline Model Scripts

| Property | Value |
|----------|-------|
| **Task Name** | Baseline Model Scripts |
| **Assignee** | Emily |
| **Status** | To Do |
| **Sprint** | Sprint 2 |
| **Deliverable** | `scripts/run_baselines.py` + `baseline_comparison.csv` |

**Execution instructions:**
- Code Random Forest and Gradient Boosting baselines on sequence-only vs structure features.
- Run 5-fold CV and record R² and accuracy.
- Produce `baseline_comparison.csv` for Luca to beat.

---

## 7. Order Lab Supplies

| Property | Value |
|----------|-------|
| **Task Name** | Order Lab Supplies |
| **Assignee** | PM |
| **Status** | To Do |
| **Sprint** | Sprint 2 |
| **Deliverable** | Target kinase + SYPRO Orange secured at Frontier Tower |

**Execution instructions:**
- Purchase purified target kinase (and fluorescent dye if needed) based on Sasha’s protocol.
- Ensure reagents are available at Frontier Tower before Sprint 3.

---

## 8. Full Screen & Leave-One-Out Eval

| Property | Value |
|----------|-------|
| **Task Name** | Full Screen & Leave-One-Out Eval |
| **Assignee** | Luca |
| **Status** | To Do |
| **Sprint** | Sprint 3 |
| **Deliverable** | Reproducible training scripts + final ranked hit list |

**Execution instructions:**
- Retrain hybrid model using new pocket features.
- Run leave-one-kinase-out evaluation.
- Log experiments (configs, metrics, seeds).
- Output final Top 50 hits for Sasha to review.
- Ensure scripts are reproducible and documented.

---

## 9. Hit Triage & Data Analysis

| Property | Value |
|----------|-------|
| **Task Name** | Hit Triage & Data Analysis |
| **Assignee** | Sasha |
| **Status** | To Do |
| **Sprint** | Sprint 3 |
| **Deliverable** | Triage notes + final wet-lab analysis graphs |

**Execution instructions:**
- Review Luca’s Top 50 hits for PAINS, toxicity, promiscuity.
- Narrow to Top 5 for wet-lab assay.
- After PM runs the physical assay, process raw CSV data into Tm melting curves.
- Produce triage notes and analysis graphs.

---

## 10. Literature Review & Docs

| Property | Value |
|----------|-------|
| **Task Name** | Literature Review & Docs |
| **Assignee** | Emily |
| **Status** | To Do |
| **Sprint** | Sprint 3 |
| **Deliverable** | `LITERATURE_REVIEW.md` + README update |

**Execution instructions:**
- Draft review on Type I/II kinase inhibitors and DFG conformations.
- Document the KLIFS → Features → Model data pipeline.
- Update GitHub README for users and contributors.
- Ensure docs reflect branch `kinase-v1` and S3 data location.

---

## 11. Order Lab Supplies

| Property | Value |
|----------|-------|
| **Task Name** | Order Lab Supplies |
| **Assignee** | PM |
| **Status** | To Do |
| **Sprint** | Sprint 2 |
| **Deliverable** | Target kinase + SYPRO Orange secured at Frontier Tower |

**Execution instructions:**
- Purchase purified target kinase and fluorescent dye based on Sasha’s protocol.

---

## 12. Execute Wet-Lab Assay

| Property | Value |
|----------|-------|
| **Task Name** | Execute Wet-Lab Assay |
| **Assignee** | PM |
| **Status** | To Do |
| **Sprint** | Sprint 3 |
| **Deliverable** | Physical qPCR experiment completed |

**Execution instructions:**
- Order the Top 5 compounds Sasha approves.
- Mix reagents in lab according to Sasha’s protocol.
- Run the qPCR machine.
- Email raw CSV to Sasha for analysis.

---

# Quick Reference: S3 & Branch

| Item | Value |
|------|-------|
| **Branch** | `kinase-v1` |
| **S3 Bucket** | `s3://amzn-s3-proteinbucket` |
| **Kinase Path** | `hyaline/kinase/` |
| **Full S3 Path** | `s3://amzn-s3-proteinbucket/hyaline/kinase/` |
| **Upload Script** | `./scripts/upload_kinase_to_s3.sh` |
| **Download** | `aws s3 sync s3://amzn-s3-proteinbucket/hyaline/kinase/ ./ --region us-east-1` |

---

*Document prepared for Notion import or team handoff. Duplicate Kanban/Table sections if importing into separate Notion databases.*
