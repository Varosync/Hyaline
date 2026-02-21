# Hyaline Kinase Conformational Selectivity — Research & Implementation Guide

**For**: Graduate Student Researcher (Molecular Biology @ UIUC, Computer Science @ Northeastern)  
**Project**: Hyaline v1.0 — Kinase Conformational Selectivity Prediction  
**Branch**: `kinase-v1`  
**Last Updated**: February 2026

---

## Executive Summary

We are extending Hyaline (originally GPCR activation prediction) to predict kinase conformational selectivity. The goal is to predict **ΔpKi** (change in binding affinity) between DFG-in (active) and DFG-out (inactive) kinase conformations.

**Current Status**:
- Pure GNNs are failing (R² < 0) due to small sample sizes (~1,661 structures)
- Hybrid model (hand-crafted features + MLP) achieves R² ≈ 0.95 on synthetic data
- Key insight: The interaction term (DFG-displacement × Drug-Size) accounts for 56% of predictive power
- We have 1,661 KLIFS structures across 10 kinases with conformational annotations

**Your Mission**: Bridge the gap from synthetic to real data and implement a Feature-Injected GNN.

---

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Data Engineering](#2-data-engineering)
3. [Biological Validation](#3-biological-validation)
4. [CS Implementation](#4-cs-implementation)
5. [Deliverables](#5-deliverables)
6. [Resources](#6-resources)

---

## 1. Project Setup

### 1.1 Repository Access

**GitHub Repository**: `Varosync/Hyaline`  
**Working Branch**: `kinase-v1`

```bash
git clone https://github.com/Varosync/Hyaline.git
cd Hyaline
git checkout kinase-v1
```


### 1.2 Data Access

**IMPORTANT**: All data is stored in S3. It is NOT committed to the GitHub repository.

**S3 Bucket**: `s3://amzn-s3-proteinbucket/hyaline/kinase/`  
**Region**: `us-east-1`

The following datasets are stored in S3:

```bash
# Download all kinase data (35 MB total)
aws s3 sync s3://amzn-s3-proteinbucket/hyaline/kinase/ ./ --region us-east-1

# This will download:
# - klifs_cache/ (12 MB) → ./klifs_cache/
# - data/klifs_cache/ (12 MB) → ./data/klifs_cache/
# - checkpoints/ (11 MB) → ./checkpoints/
```

**What's included**:
- `klifs_cache/` (12 MB): 1,661 kinase structures with conformational annotations
- `klifs_cache/key_kinases.json`: Summary of 10 kinases (ABL1, EGFR, BRAF, SRC, KIT, MET, ALK, JAK2, FLT3, KDR)
- `klifs_cache/bioactivity_kinase.json`: 132 bioactivity records from ChEMBL
- `data/klifs_cache/` (12 MB): Additional cached KLIFS API responses
- `checkpoints/` (11 MB): Trained model checkpoints and results
- `checkpoints/hybrid_results.json`: Baseline hybrid model results (R² ≈ 0.95)

**AWS Credentials**: Contact PI for IAM access keys or use your institutional AWS account.

**Note**: The `.gitignore` excludes `data/`, `*.pt`, `*.pdb` files to keep the repository lightweight. All large files must be downloaded from S3.

### 1.3 Environment Setup

```bash
# Create conda environment
conda create -n hyaline python=3.10
conda activate hyaline

# Install dependencies
pip install -e .

# Verify installation
python -c "import hyaline; print(hyaline.__version__)"
```

**Required packages**:
- PyTorch 2.0+
- PyTorch Geometric
- BioPython (for PDB parsing)
- RDKit (for drug features)
- scikit-learn
- requests (for KLIFS API)

---

## 2. Data Engineering

### 2.1 Understanding KLIFS

**KLIFS** (Kinase-Ligand Interaction Fingerprints and Structures) is a database of 15,000+ kinase structures with:
- Standardized 85-residue binding pocket alignment
- DFG conformation annotations (in/out/in-like/out-like)
- αC-helix conformation annotations
- Structural features: `mobitz_dihedral`, `dfg_d_rotation`, `Grich_distance`, `Grich_angle`

**API**: https://klifs.net/api_v2  
**Documentation**: https://klifs.net/swagger/

### 2.2 Task 1: Extract Conformational Features

**Goal**: Extract `mobitz_dihedral` and `dfg_d_rotation` from cached KLIFS structures.

**File**: `hyaline/data/klifs_loader.py` (already implemented)

**What you need to do**:

```python
from hyaline.loaders.klifs_loader import KLIFSClient

client = KLIFSClient(cache_enabled=True)

# Load cached kinase data
with open('klifs_cache/key_kinases.json') as f:
    kinase_data = json.load(f)

# For each structure, extract features
for kinase_info in kinase_data['kinases']:
    structures = client.get_structures(kinase_id=kinase_info['id'])
    
    for struct in structures:
        # Extract from KLIFS API response
        features = {
            'structure_id': struct.structure_id,
            'pdb_id': struct.pdb_id,
            'dfg_conformation': struct.dfg.value,
            'mobitz_dihedral': struct.mobitz_dihedral,  # TODO: Add to API call
            'dfg_d_rotation': struct.dfg_d_rotation,    # TODO: Add to API call
            'Grich_distance': struct.Grich_distance,
            'Grich_angle': struct.Grich_angle
        }
```

**Challenge**: The current `klifs_loader.py` doesn't fetch `mobitz_dihedral` and `dfg_d_rotation`. You need to:

1. Check KLIFS API documentation for the correct endpoint
2. Modify `KLIFSClient.get_structures()` to include these fields
3. Update `KLIFSStructure` dataclass to store them

**Expected Output**: A CSV file with columns:
```
structure_id, pdb_id, kinase_name, dfg_conf, mobitz_dihedral, dfg_d_rotation, Grich_distance, Grich_angle
```


### 2.3 Task 2: Map to Bioactivity Data

**Goal**: Create matched structure-affinity pairs (same drug, multiple conformations).

**Data Sources**:
- KLIFS structures (1,661 with conformations)
- ChEMBL bioactivity (132 records in `klifs_cache/bioactivity_kinase.json`)

**What you need to do**:

```python
# Load bioactivity data
with open('klifs_cache/bioactivity_kinase.json') as f:
    bioactivity = json.load(f)

# Match structures to bioactivity
matched_pairs = []

for record in bioactivity:
    kinase = record['target']
    drug = record['compound_name']
    pki = record['pKi']  # -log10(Ki)
    
    # Find structures with this drug
    structures = [s for s in all_structures 
                  if s.kinase_name == kinase and s.ligand == drug]
    
    # Group by conformation
    dfg_in_structs = [s for s in structures if s.is_dfg_in]
    dfg_out_structs = [s for s in structures if s.is_dfg_out]
    
    # Create pairs
    for s_in in dfg_in_structs:
        for s_out in dfg_out_structs:
            matched_pairs.append({
                'kinase': kinase,
                'drug': drug,
                'pdb_in': s_in.pdb_id,
                'pdb_out': s_out.pdb_id,
                'pki_in': pki_in,  # Need to fetch separately
                'pki_out': pki_out,
                'delta_pki': pki_in - pki_out
            })
```

**Challenge**: Most bioactivity records don't specify which conformation was used. You'll need to:

1. Use known Type I/II inhibitor classifications (see Section 3)
2. Infer conformation from crystal structure (DFG annotation)
3. For ambiguous cases, use both and flag as uncertain

**Expected Output**: `matched_pairs.csv` with columns:
```
kinase, drug, pdb_in, pdb_out, pki_in, pki_out, delta_pki, confidence
```

**Realistic Expectation**: You'll likely get 20-50 high-confidence matched pairs, not hundreds.

### 2.4 Task 3: Extract Drug Features

**Goal**: Compute drug size and flexibility from SMILES/SDF.

**Tools**: RDKit

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def extract_drug_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    return {
        'mol_weight': Descriptors.MolWt(mol),
        'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
        'num_heavy_atoms': Lipinski.HeavyAtomCount(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_h_donors': Lipinski.NumHDonors(mol),
        'num_h_acceptors': Lipinski.NumHAcceptors(mol)
    }
```

**Data Source**: 
- KLIFS provides ligand PDB codes
- Use PubChem API to get SMILES: `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug}/property/CanonicalSMILES/JSON`

**Expected Output**: `drug_features.csv`

---

## 3. Biological Validation

### 3.1 Type I vs Type II Inhibitors

**Background**: Kinase inhibitors are classified by binding mode:

- **Type I**: Bind to DFG-in (active) conformation
  - Smaller, fit into ATP pocket
  - Examples: Gefitinib, Erlotinib, Dasatinib
  
- **Type II**: Bind to DFG-out (inactive) conformation
  - Larger, access hydrophobic back pocket
  - Examples: Imatinib, Sorafenib, Nilotinib

**Hypothesis**: Our hybrid model's DFG×Drug_Size interaction captures this biology.

### 3.2 Task 4: Validate Known Inhibitors

**Goal**: Verify that our features align with known pharmacology.

**Test Cases** (from `hyaline/data/klifs_loader.py`):

```python
KNOWN_KINASE_INHIBITORS = {
    'Imatinib': {'type': 'II', 'targets': ['ABL1'], 'dfg_preference': 'out'},
    'Nilotinib': {'type': 'II', 'targets': ['ABL1'], 'dfg_preference': 'out'},
    'Gefitinib': {'type': 'I', 'targets': ['EGFR'], 'dfg_preference': 'in'},
    'Erlotinib': {'type': 'I', 'targets': ['EGFR'], 'dfg_preference': 'in'},
    'Sorafenib': {'type': 'II', 'targets': ['BRAF'], 'dfg_preference': 'out'},
}
```

**What you need to do**:

1. For each drug, find all KLIFS structures
2. Check if DFG annotation matches expected preference
3. Compute drug size (MW, heavy atoms)
4. Verify: Type II drugs are larger than Type I

**Script**: `scripts/klifs_validation.py` (already exists, but needs updating)

**Expected Result**:
- Type II drugs: 100% in DFG-out structures ✓
- Type I drugs: >80% in DFG-in structures
- Type II drugs: MW > 450 Da
- Type I drugs: MW < 450 Da


### 3.3 Task 5: Literature Review

**Goal**: Understand the structural basis of conformational selectivity.

**Key Papers**:

1. **Möbitz (2015)**: "The ABC of protein kinase conformations"
   - Defines the Möbitz dihedral angle
   - Explains DFG-in/out transition mechanism

2. **Roskoski (2016)**: "Classification of small molecule protein kinase inhibitors"
   - Type I, II, III, IV classification
   - Structure-activity relationships

3. **Zhao et al. (2014)**: "Exploration of type II binding mode"
   - Structural determinants of Type II selectivity
   - Role of gatekeeper residue

**Questions to Answer**:
- What is the typical range of `mobitz_dihedral` for DFG-in vs DFG-out?
- How does the gatekeeper residue affect Type II binding?
- Are there kinases that strongly prefer one conformation?

**Deliverable**: 1-2 page summary with key insights for model design.

---

## 4. CS Implementation

### 4.1 Current Model Architecture

**Hybrid Model** (`scripts/hybrid_kinase_model.py`):

```
Input: 6 hand-crafted features
  - dfg_magnitude (Å)
  - chelix_shift (Å)
  - rmsd (Å)
  - drug_size (normalized)
  - drug_flex (normalized)
  - dfg_magnitude × drug_size (interaction term)

Architecture:
  - MLP: [6] → [64] → [32] → [16] → [8] → [1]
  - LayerNorm + GELU + Dropout(0.2)
  - AdamW optimizer, lr=1e-3

Output: Predicted ΔpKi

Performance: R² ≈ 0.95 (synthetic data)
```

**Problem**: This is on synthetic data. Real data performance is unknown.

### 4.2 Task 6: Feature-Injected GNN

**Goal**: Instead of expecting the GNN to learn DFG displacement from raw coordinates, inject it as a global graph attribute.

**Architecture**:

```python
class FeatureInjectedGNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Node features: ESM3 embeddings (1536-dim)
        self.node_encoder = nn.Linear(1536, 128)
        
        # Edge features: RBF distance (96-dim)
        self.edge_encoder = nn.Linear(96, 64)
        
        # Global features: Biological dihedrals
        self.global_encoder = nn.Linear(4, 32)  # mobitz, dfg_d, Grich_dist, Grich_angle
        
        # EGNN layers
        self.egnn1 = EGNNLayer(128, 64)
        self.egnn2 = EGNNLayer(128, 64)
        self.egnn3 = EGNNLayer(128, 64)
        
        # Readout with global features
        self.readout = nn.Sequential(
            nn.Linear(128 + 32, 64),  # Concatenate graph + global
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, batch):
        # Encode nodes
        h = self.node_encoder(batch.x)
        
        # Encode edges
        edge_attr = self.edge_encoder(batch.edge_attr)
        
        # Encode global features (biological dihedrals)
        global_feat = self.global_encoder(batch.global_features)
        
        # EGNN layers
        h = self.egnn1(h, batch.edge_index, edge_attr, batch.pos)
        h = self.egnn2(h, batch.edge_index, edge_attr, batch.pos)
        h = self.egnn3(h, batch.edge_index, edge_attr, batch.pos)
        
        # Global pooling
        h_graph = global_mean_pool(h, batch.batch)
        
        # Concatenate with global features
        h_combined = torch.cat([h_graph, global_feat], dim=-1)
        
        # Predict ΔpKi
        return self.readout(h_combined)
```

**Key Idea**: The GNN learns spatial patterns, but we give it the hard-to-learn conformational features directly.


### 4.3 Task 7: Implementation Steps

**Step 1**: Modify data loader to include global features

```python
# In hyaline/data/klifs_pipeline.py

def create_graph_with_global_features(structure, conformational_features):
    # Standard graph construction
    data = Data(
        x=node_features,  # ESM3 embeddings
        edge_index=edge_index,
        edge_attr=edge_attr,  # RBF distances
        pos=coords
    )
    
    # Add global features
    data.global_features = torch.tensor([
        conformational_features['mobitz_dihedral'],
        conformational_features['dfg_d_rotation'],
        conformational_features['Grich_distance'],
        conformational_features['Grich_angle']
    ], dtype=torch.float32)
    
    return data
```

**Step 2**: Implement the Feature-Injected GNN

Create `hyaline/models/feature_injected_gnn.py` with the architecture above.

**Step 3**: Training script

Create `scripts/train_feature_injected_gnn.py`:

```python
# Pseudocode
model = FeatureInjectedGNN()
optimizer = AdamW(model.parameters(), lr=1e-4)

for epoch in range(300):
    for batch in train_loader:
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()
```

**Step 4**: Ablation study

Compare:
1. Pure GNN (no global features) — baseline
2. Feature-Injected GNN (with dihedrals)
3. Hybrid MLP (hand-crafted features only)

**Expected Result**: Feature-Injected GNN should outperform pure GNN, but may not beat Hybrid MLP.

### 4.4 Task 8: Hyperparameter Tuning

**Search Space**:
- Learning rate: [1e-5, 1e-4, 1e-3]
- Hidden dims: [64, 128, 256]
- Number of EGNN layers: [2, 3, 4]
- Dropout: [0.1, 0.2, 0.3]
- Global feature encoding: [16, 32, 64]

**Tool**: Optuna or Ray Tune

**Metric**: 5-fold cross-validation R²

**Budget**: ~50 trials (should take 2-3 hours on GPU)

---

## 5. Deliverables

### 5.1 Data Deliverables

**Due**: Week 2

1. **`conformational_features.csv`**: All 1,661 structures with extracted dihedrals
   - Columns: `structure_id, pdb_id, kinase, dfg_conf, mobitz_dihedral, dfg_d_rotation, Grich_distance, Grich_angle`

2. **`matched_pairs.csv`**: Paired structures with bioactivity
   - Columns: `kinase, drug, pdb_in, pdb_out, pki_in, pki_out, delta_pki, confidence`
   - Target: 20-50 high-confidence pairs

3. **`drug_features.csv`**: Drug properties
   - Columns: `drug_name, smiles, mol_weight, num_rotatable_bonds, logp, tpsa`

### 5.2 Validation Deliverables

**Due**: Week 3

4. **`validation_report.pdf`**: Biological validation results
   - Type I/II classification accuracy
   - Drug size vs. conformation preference
   - Literature review summary (1-2 pages)

### 5.3 Model Deliverables

**Due**: Week 5

5. **`feature_injected_gnn.py`**: Implemented model

6. **`performance_report.pdf`**: Comparison of models
   - Table: Model | R² | Pearson r | MAE | RMSE
   - Rows: Pure GNN, Feature-Injected GNN, Hybrid MLP
   - Include learning curves and ablation results

7. **`trained_model.pt`**: Best checkpoint

### 5.4 Final Deliverable

**Due**: Week 6

8. **`FINAL_REPORT.pdf`**: 5-10 page report covering:
   - Introduction and motivation
   - Data engineering process and challenges
   - Biological validation results
   - Model architecture and training
   - Results and discussion
   - Limitations and future work

---

## 6. Resources

### 6.1 Code Structure

```
Hyaline/
├── hyaline/
│   ├── data/
│   │   ├── klifs_loader.py          # KLIFS API client
│   │   └── klifs_pipeline.py        # Data processing
│   ├── models/
│   │   ├── kinase_binding.py        # Original GNN
│   │   └── feature_injected_gnn.py  # TODO: Your implementation
│   └── features/
│       └── esm_embeddings.py        # ESM3 feature extraction
├── scripts/
│   ├── hybrid_kinase_model.py       # Baseline hybrid model
│   ├── train_real_klifs.py          # Training on real data
│   └── klifs_validation.py          # Biological validation
├── research/kinase/
│   ├── PROGRESS.md                  # Project history
│   ├── SUMMARY.md                   # Technical summary
│   └── RESEARCH_GUIDE.md            # This document
└── data/
    └── klifs_cache/                 # Cached KLIFS data (download from S3)
```


### 6.2 Key Files to Read First

**Priority 1** (Read in first 2 days):
1. `research/kinase/PROGRESS.md` — Understand what's been done
2. `research/kinase/SUMMARY.md` — Technical details
3. `hyaline/data/klifs_loader.py` — Data infrastructure
4. `scripts/hybrid_kinase_model.py` — Baseline model

**Priority 2** (Week 1):
5. `scripts/klifs_validation.py` — Biological validation
6. `hyaline/models/kinase_binding.py` — Original GNN architecture

### 6.3 External Resources

**KLIFS Database**:
- Website: https://klifs.net/
- API Docs: https://klifs.net/swagger/
- Paper: van Linden et al. (2014) "KLIFS: A structural kinase-ligand interaction database"

**ChEMBL Database**:
- Website: https://www.ebi.ac.uk/chembl/
- API: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
- Use for bioactivity data (Ki, IC50, Kd)

**PubChem**:
- REST API: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- Use for SMILES and drug properties

**ESM3 Embeddings**:
- Model: `esm3_sm_open_v1` (1536-dim)
- Hugging Face: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
- Already integrated in `hyaline/features/esm_embeddings.py`

### 6.4 Computing Resources

**Local Development**:
- CPU: Data processing, feature extraction
- RAM: 16 GB minimum (32 GB recommended)

**GPU Training**:
- Recommended: NVIDIA A100 or V100
- Minimum: RTX 3090 (24 GB VRAM)
- Cloud options: AWS p3.2xlarge, Google Cloud A100

**Estimated Compute Time**:
- Data processing: 2-4 hours (CPU)
- Hybrid model training: 10 minutes (CPU)
- GNN training: 1-2 hours per run (GPU)
- Hyperparameter search: 4-6 hours (GPU)

### 6.5 Communication

**Weekly Check-ins**: Fridays at 2 PM (Zoom link in email)

**Slack Channel**: `#hyaline-kinase`

**Questions**: 
- Technical (code): Post in Slack
- Scientific (biology): Email PI or schedule office hours
- Urgent: Text PI

**Progress Updates**: 
- Brief weekly summary in Slack (Friday EOD)
- Detailed updates in `research/kinase/WEEKLY_UPDATES.md`

---

## 7. Getting Started Checklist

### Week 1: Setup & Exploration

- [ ] Clone repo and checkout `kinase-v1` branch
- [ ] Set up conda environment and install dependencies
- [ ] **CRITICAL**: Download data from S3 (`aws s3 sync s3://amzn-s3-proteinbucket/hyaline/kinase/ ./ --region us-east-1`)
- [ ] Verify data downloaded: `ls klifs_cache/ data/klifs_cache/ checkpoints/`
- [ ] Read `PROGRESS.md` and `SUMMARY.md`
- [ ] Run `scripts/hybrid_kinase_model.py` to verify setup (uses synthetic data, no S3 needed)
- [ ] Explore KLIFS API with example in `hyaline/data/klifs_loader.py`
- [ ] Examine cached data in `klifs_cache/` and `data/klifs_cache/`

### Week 2: Data Engineering

- [ ] Extract conformational features from KLIFS
- [ ] Map structures to bioactivity data
- [ ] Create matched-pair dataset
- [ ] Extract drug features with RDKit
- [ ] Deliverable: 3 CSV files

### Week 3: Biological Validation

- [ ] Validate Type I/II inhibitor classifications
- [ ] Analyze drug size vs. conformation preference
- [ ] Literature review on conformational selectivity
- [ ] Deliverable: Validation report

### Week 4-5: Model Implementation

- [ ] Implement Feature-Injected GNN
- [ ] Create training script
- [ ] Run ablation study (Pure GNN vs Feature-Injected vs Hybrid)
- [ ] Hyperparameter tuning
- [ ] Deliverable: Model code + performance report

### Week 6: Final Report

- [ ] Write comprehensive report
- [ ] Prepare presentation slides
- [ ] Code cleanup and documentation
- [ ] Deliverable: Final report + presentation

---

## 8. Expected Challenges & Solutions

### Challenge 1: Limited Matched Pairs

**Problem**: Most bioactivity data doesn't specify conformation.

**Solution**: 
- Focus on known Type I/II inhibitors first
- Use structural annotations to infer conformation
- Accept smaller dataset (20-50 pairs) as proof-of-concept

### Challenge 2: GNN May Still Underperform

**Problem**: Even with injected features, GNN might not beat Hybrid MLP.

**Solution**:
- This is a valid scientific result!
- Document why: sample size, feature engineering, inductive bias
- Propose future work: pre-training on larger datasets

### Challenge 3: Missing Conformational Features

**Problem**: Not all KLIFS structures have `mobitz_dihedral` computed.

**Solution**:
- Compute from PDB coordinates yourself (see Möbitz 2015 paper)
- Use BioPython to extract dihedral angles
- Validate against KLIFS annotations where available

### Challenge 4: API Rate Limits

**Problem**: KLIFS/ChEMBL APIs have rate limits.

**Solution**:
- Use cached data first (`klifs_cache/`)
- Implement exponential backoff for new requests
- Batch requests when possible

---

## 9. Success Criteria

**Minimum Viable Product** (Pass):
- [ ] 20+ matched pairs with real bioactivity data
- [ ] Feature-Injected GNN implemented and trained
- [ ] Performance comparison showing structure is necessary
- [ ] Final report documenting process and results

**Target Goals** (Good):
- [ ] 50+ matched pairs
- [ ] Feature-Injected GNN R² > 0.3 on real data
- [ ] Biological validation confirms Type I/II predictions
- [ ] Ablation study shows injected features help

**Stretch Goals** (Excellent):
- [ ] Feature-Injected GNN matches or beats Hybrid MLP
- [ ] Cross-kinase generalization (leave-one-out R² > 0.2)
- [ ] Novel predictions on untested kinase-drug pairs
- [ ] Manuscript draft ready for submission

---

## 10. Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Setup & Exploration | Environment ready, data downloaded |
| 2 | Data Engineering | 3 CSV files (features, pairs, drugs) |
| 3 | Biological Validation | Validation report (PDF) |
| 4 | Model Implementation | Feature-Injected GNN code |
| 5 | Training & Evaluation | Performance report (PDF) |
| 6 | Final Report | Final report + presentation |

**Total Duration**: 6 weeks (can extend to 8 if needed)

---

## Appendix A: S3 Data Upload Instructions (For PI)

```bash
# Upload KLIFS cache
aws s3 sync ./klifs_cache/ s3://hyaline-kinase-data/klifs_cache/ --region us-east-1

# Upload checkpoints
aws s3 sync ./checkpoints/ s3://hyaline-kinase-data/checkpoints/ --region us-east-1

# Upload data cache
aws s3 sync ./data/klifs_cache/ s3://hyaline-kinase-data/klifs_cache/ --region us-east-1

# Set public read access (optional, for easier sharing)
aws s3 cp s3://hyaline-kinase-data/ s3://hyaline-kinase-data/ --recursive --acl public-read

# Verify upload
aws s3 ls s3://hyaline-kinase-data/ --recursive --human-readable --summarize
```

