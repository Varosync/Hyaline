# Hyaline Kinase Project (v1.0)

**Branch**: `kinase-v1`  
**Status**: Active Development  
**Last Updated**: February 2026

## Quick Start

This branch contains the kinase conformational selectivity prediction work, extending Hyaline from GPCR activation to kinase DFG-in/out prediction.

### For Graduate Student Researchers

**Start here**: Read [`RESEARCH_GUIDE.md`](./RESEARCH_GUIDE.md) for comprehensive instructions.

**Quick setup**:
```bash
# Clone and checkout
git clone https://github.com/Varosync/Hyaline.git
cd Hyaline
git checkout kinase-v1

# CRITICAL: Download data from S3 (NOT in repo)
aws s3 sync s3://hyaline-kinase-data/ ./ --region us-east-1

# Verify data downloaded
ls klifs_cache/ data/klifs_cache/ checkpoints/

# Install
conda create -n hyaline python=3.10
conda activate hyaline
pip install -e .

# Verify (uses synthetic data, no S3 needed)
python scripts/hybrid_kinase_model.py
```

### For PI/Collaborators

**Upload data to S3**:
```bash
./scripts/upload_to_s3.sh
```

**Current status**: See [`PROGRESS.md`](./PROGRESS.md)

## Project Structure

```
research/kinase/
├── README.md              # This file
├── RESEARCH_GUIDE.md      # Comprehensive guide for students
├── PROGRESS.md            # Project history and current state
└── SUMMARY.md             # Technical summary

scripts/
├── hybrid_kinase_model.py      # Baseline hybrid model (R² ≈ 0.95)
├── train_real_klifs.py         # Training on real KLIFS data
├── klifs_validation.py         # Biological validation
└── upload_to_s3.sh             # Data upload script

hyaline/
├── data/klifs_loader.py        # KLIFS API client
└── models/kinase_binding.py    # GNN architecture

data/
└── klifs_cache/                # 1,661 structures (download from S3)
```

## Key Results

- **Hybrid Model**: R² ≈ 0.95 on synthetic data
- **Feature Importance**: DFG×Drug_Size = 56%
- **Pure GNN**: R² < 0 (failing on small dataset)
- **KLIFS Data**: 1,661 structures across 10 kinases
- **Bioactivity**: 132 records from ChEMBL

## Next Steps

1. Extract real conformational features (mobitz_dihedral, dfg_d_rotation)
2. Create matched structure-affinity pairs
3. Implement Feature-Injected GNN
4. Compare: Pure GNN vs Feature-Injected GNN vs Hybrid MLP

## Contact

- PI: [email]
- Slack: `#hyaline-kinase`
- Issues: GitHub Issues on this branch
