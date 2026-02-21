# Hyaline Drug Discovery AI — Product Brief

**Document type**: Product manager brief for stakeholder review  
**Purpose**: Define vision, current state, learnings, and roadmap for engineering allocation  
** audience**: Product managers, then engineers for task assignment

---

## 1. Vision: What We Are Building

We are building **AI technology that transforms drug discovery** by predicting how small molecules bind to protein targets across different conformational states. The core insight: **the same protein in different shapes binds different drugs**. Today’s methods largely treat proteins as static; we aim to model conformational dynamics and use them to predict binding.

**Long-term product vision**:
- A platform that predicts **conformational-selective binding** (which drugs prefer which protein conformations)
- Applicable across protein families: kinases today, GPCRs, other therapeutic targets later
- Actionable for medicinal chemists: “This compound will bind better when the kinase is DFG-out”
- Built on geometric deep learning (3D structure) plus domain knowledge, not sequence-only

**Why this matters**:
- Kinase inhibitors: ~$20B+ annual market (e.g. Imatinib, Gefitinib)
- Type I vs Type II selectivity drives efficacy, resistance, and safety
- Structure-based prediction is still underused; most tools are either empirical or fully black-box

---

## 2. What Hyaline Initially Was

**Original Hyaline** (published, bioRxiv 2026):

- **Problem**: GPCR activation state prediction (active vs inactive)
- **Input**: 3D structure of a GPCR
- **Output**: Probability of active vs inactive conformation
- **Architecture**: E(n)-equivariant GNN with ESM3 embeddings, RBF distance encoding, motif attention (DRY, NPxxY, CWxP)
- **Results**: 0.995 AuROC on cross-validation, 0.819 on temporal holdout, large gains over sequence-only baselines (especially Class C GPCRs)
- **Data**: 1,596 GPCR structures from PDB

So the **original product** is: *predict GPCR activation state from structure*. It is a focused, working model for one protein family and one task.

---

## 3. What We Are Aiming For (Current Target)

**Current target**: Extend Hyaline’s approach to **kinase conformational selectivity** — predicting how binding changes when a kinase switches between DFG-in and DFG-out.

**Concrete goals**:

| Goal | Success Metric | Status |
|------|----------------|--------|
| Prove structure is necessary | RF(structure) >> RF(sequence) | ✓ Achieved |
| Predict ΔpKi (binding change across conformations) | R² > 0.5 | Partially (synthetic only) |
| Classify Type I vs Type II from structure | Accuracy > 80% | ✓ Achieved on known drugs |
| Spiking/temporal dynamics add value | Spiking EGNN > static EGNN | ✗ Debunked for current setup |
| Generalize across kinases | Leave-one-kinase-out r > 0.4 | ✗ Not achieved |

**Product framing**: A tool that tells medicinal chemists which kinase conformations a compound will prefer, and by how much — to guide design and prioritization.

---

## 4. What We Have Achieved

### 4.1 Validated Scientific Insights
- **Structure is necessary**: Sequence-only models (R² ≈ 0.01) fail; structure-based models (R² ≈ 0.35–0.96) work
- **Interaction terms are critical**: DFG displacement × drug size drives ~56% of prediction; this is the main missing signal for black-box models
- **Domain knowledge beats raw learning on small data**: Hand-crafted features outperform end-to-end GNNs when data < ~2,000 samples

### 4.2 Technical Milestones
- **Hybrid model**: RF features (DFG, C-helix, drug size, DFG×size) + neural net → R² ≈ 0.95–0.96 on synthetic data
- **KLIFS integration**: API client, ~1,661 structures for 10 kinases, conformational annotations (mobitz_dihedral, dfg_d_rotation, etc.)
- **Bioactivity pipeline**: 132 kinase–inhibitor records from ChEMBL via KLIFS
- **Biological validation**: 100% correct classification of Imatinib and Nilotinib as Type II (DFG-out) from KLIFS structures

### 4.3 Within-Kinase Performance
- DFG classification from pocket sequence: 84.7% average accuracy vs 77.6% baseline
- KDR: 100% accuracy; BRAF: 86%; others 75–79%

### 4.4 Architecture and Code
- Ablation framework (RF sequence vs structure vs GNN)
- Hybrid model training pipeline
- KLIFS loader and pipeline
- Scripts for validation on known drugs

---

## 5. What We Have Struggled With

### 5.1 Graph Neural Networks
- **Static EGNN**: R² ≈ -0.11 (worse than predicting the mean)
- **Spiking EGNN**: R² ≈ -0.06 (still worse than mean)
- Both far worse than RF (R² ≈ 0.35)
- **Root cause**: ~100K+ parameters, ~1.5K samples, and no pre-training → cannot learn key interactions (e.g. DFG×drug_size) from raw coordinates

### 5.2 Cross-Kinase Generalization
- Leave-one-kinase-out (e.g. hold out ABL1): ~25% accuracy (below random)
- Pocket sequences are kinase-specific; model memorizes training kinase instead of generalizing
- Transfer across kinases is a core problem, not solved yet

### 5.3 Real Binding Affinity Data
- ChEMBL bioactivity is not 1:1 with structures; many records lack matched PDB structures
- ΔpKi requires same drug bound to same kinase in both DFG-in and DFG-out — rare in public data
- Current “real” experiments use semi-synthetic or proxy labels

### 5.4 Prior Pivot: TF Activation
- Earlier TF activation task failed: model learned lookup (TF + cell type → activity), not structure-driven biology
- Led to pivot to kinases, where structure is provably necessary

---

## 6. What We Have Debunked

### 6.1 “Spiking dynamics help for this task”
- Spiking EGNN is slightly better than static EGNN, but both have negative R²
- No meaningful value demonstration; cannot claim spiking helps until base model is useful

### 6.2 “End-to-end GNNs will learn structure–drug interactions”
- On small datasets, GNNs do not learn DFG×drug_size or equivalent interactions
- RF with hand-crafted features wins decisively

### 6.3 “Pocket sequence alone suffices”
- Sequence-only models fail (R² ≈ 0.01)
- Structure (or structure-derived features) is required

### 6.4 “100% on 5 known drugs = validated”
- Small sample size; many are textbook examples
- Need larger, harder validation sets

### 6.5 “Cross-kinase transfer from pocket sequence”
- Pocket sequences are kinase-specific; model does not generalize to unseen kinases
- Requires different features or pre-training

---

## 7. Problems We Are Solving (Value Proposition)

| Problem | Who Has It | Our Approach |
|---------|-----------|--------------|
| Predicting which conformation a drug prefers | Medicinal chemists, computational chemists | Structure-based model with DFG/C-helix features and interaction terms |
| Type I vs Type II classification from structure | Kinase drug discovery teams | Classifier using pocket + conformational features |
| Knowing if structure adds value vs sequence | Method developers | Ablations showing structure necessity |
| Using conformational data in design | Pharma R&D | Pipeline from KLIFS structures + ChEMBL affinities to predictions |

---

## 8. Problems the Technology Has (Gaps and Risks)

| Problem | Severity | Mitigation |
|---------|----------|------------|
| GNNs fail on small data | High | Stick with hybrid; pre-train GNNs on large molecular sets before fine-tuning |
| No cross-kinase transfer | High | Pre-training, shared representations, or kinase-family–specific models |
| Limited real ΔpKi data | Medium | Synthetic/proxy labels; prioritize kinases with rich data |
| ChEMBL–structure matching is partial | Medium | Manual curation; partner data; focus on well-covered kinases |
| Spiking hypothesis not validated | Low | Deprioritize until base model performs |

---

## 9. Approaches We Could Take

### 9.1 Hybrid-First (Recommended Near-Term)
- **Idea**: Use hand-crafted features (DFG, C-helix, drug size, DFG×size) and improve the neural head
- **Effort**: Medium | **Risk**: Low | **Data**: Current KLIFS + ChEMBL
- **Deliverable**: Production hybrid model on real KLIFS conformational features

### 9.2 Pre-Trained GNN
- **Idea**: Pre-train GNN on ZINC, PubChem, or large kinase sets; fine-tune on our task
- **Effort**: High | **Risk**: Medium | **Data**: Large external + our curated set
- **Deliverable**: GNN that can potentially generalize across kinases

### 9.3 Task Simplification: Classification Over Regression
- **Idea**: Predict Type I vs Type II instead of continuous ΔpKi
- **Effort**: Low | **Risk**: Low | **Data**: Current
- **Deliverable**: High-accuracy classifier for known drugs and new candidates

### 9.4 Real Conformational Features in Hybrid
- **Idea**: Replace synthetic DFG/C-helix features with KLIFS mobitz_dihedral, dfg_d_rotation, ploop metrics
- **Effort**: Low | **Risk**: Low | **Data**: KLIFS conformational API
- **Deliverable**: Hybrid model trained on real conformational data

### 9.5 Data Expansion
- **Idea**: Broaden kinases, improve structure–affinity matching, add private or partner data
- **Effort**: Medium–High | **Risk**: Medium | **Data**: KLIFS, ChEMBL, partners
- **Deliverable**: Larger, higher-quality training set

### 9.6 Multi-Target Platform
- **Idea**: Extend beyond kinases to GPCRs (original Hyaline) and other families
- **Effort**: High | **Risk**: High | **Data**: Family-specific
- **Deliverable**: Unified conformational prediction platform

---

## 10. Recommended Engineering Priorities

**Phase 1 — Quick Wins (1–2 sprints)**  
1. Replace synthetic features with real KLIFS conformational features in the hybrid model  
2. Train and validate hybrid on real structural data  
3. Implement Type I/II classifier as a product feature  
4. Document API and usage for internal/external users  

**Phase 2 — Data & Robustness (2–3 sprints)**  
5. Improve ChEMBL–KLIFS structure matching and curation  
6. Expand kinase coverage (more kinases, more conformations)  
7. Add proper train/validation/test splits and reporting  

**Phase 3 — Scale & Generalization (3+ sprints)**  
8. Explore GNN pre-training on large molecular datasets  
9. Investigate cross-kinase transfer (e.g. leave-one-kinase-out)  
10. Consider integration with design tools and workflows  

---

## 11. Executive Summary (for PM Discussion)

| Dimension | Summary |
|-----------|---------|
| **Vision** | AI for conformational-selective drug binding prediction |
| **Origin** | Hyaline = GPCR activation prediction; extended to kinases |
| **Target** | Kinase conformational selectivity (DFG-in vs DFG-out, Type I vs II) |
| **Achieved** | Structure necessity proven; hybrid R²≈0.95; KLIFS pipeline; 100% on known Type II drugs |
| **Struggled** | GNNs fail; no cross-kinase transfer; limited real ΔpKi data |
| **Debunked** | Spiking helps; end-to-end GNNs learn interactions; sequence suffices |
| **Value** | Predict drug–conformation preference for medicinal chemistry |
| **Top approach** | Hybrid with real KLIFS conformational features; classification as simpler product |
| **Top risk** | Data volume and cross-kinase generalization |

---

## 12. Appendix: Key Assets and Code Locations

```
scripts/
├── hybrid_kinase_model.py      # Hybrid model (R²≈0.95)
├── kinase_ablation.py          # Ablation study
├── train_real_klifs.py         # Real KLIFS training
├── klifs_validation.py         # Known drug validation

hyaline/
├── models/kinase_binding.py
├── data/klifs_loader.py
├── data/klifs_pipeline.py

klifs_cache/
├── key_kinases.json            # 10 kinases, 1,661 structures
├── bioactivity_kinase.json     # 132 bioactivity records

research/kinase/
├── SUMMARY.md                  # Technical summary
├── PROGRESS.md                 # Session handoff
└── PRODUCT_BRIEF.md            # This document
```

---

*Document prepared for product manager review and engineering assignment. Last updated: Jan 2026.*
