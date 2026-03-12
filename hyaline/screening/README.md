# Type II Kinase Inhibitor Screening

This module provides tools for screening compound databases (ZINC, Enamine) to identify Type II kinase inhibitors.

## Features

- **ZINC Database Integration**: Query 230M+ purchasable compounds
- **Enamine REAL Integration**: Access 6.5B+ make-on-demand compounds
- **Price Filtering**: Filter compounds by cost (default: $30/mg max)
- **GPU Acceleration**: Batch screening with PyTorch
- **Hybrid Model**: Combines pocket geometry + compound descriptors

## Quick Start

```bash
# Screen ABL1 for Type II inhibitors
python scripts/screen_type2_inhibitors.py --kinase ABL1 --max-price 30 --top-n 100

# Screen with custom checkpoint
python scripts/screen_type2_inhibitors.py \
    --kinase EGFR \
    --checkpoint checkpoints/screening/best_model.pt \
    --max-compounds 5000 \
    --batch-size 64
```

## Model Architecture

The `Type2ScreeningModel` combines:

1. **Pocket Features (16-dim)**:
   - DFG-αC distance
   - Hinge-activation loop angle
   - Pocket volume, ESP, hydrophobicity
   - DFG/αC-helix conformations
   - Resolution, quality score

2. **Compound Features (8-dim)**:
   - Molecular weight
   - LogP
   - H-bond donors/acceptors
   - Price per mg
   - Source (ZINC/Enamine)
   - Availability

3. **Outputs**:
   - Type II score (0-1): Probability of Type II binding
   - DFG-out probability (0-1): Likelihood of inducing DFG-out
   - Binding affinity (pKi): Predicted affinity

## API Integration Status

### ZINC Database
- **Status**: Placeholder implementation
- **TODO**: Implement REST API calls to `https://zinc.docking.org/api`
- **Endpoints needed**:
  - `/substances/substructure` - Substructure search
  - `/substances/similarity` - Similarity search
  - `/substances/{zinc_id}` - Get compound details

### Enamine REAL
- **Status**: Placeholder implementation
- **TODO**: Implement API or download catalog
- **Options**:
  - Use Enamine API (requires license)
  - Download REAL catalog (requires storage)
  - Use ChEMBL/PubChem as proxy

## Output Format

Results are saved as JSON and CSV:

```json
{
  "rank": 1,
  "compound_id": "ZINC000012345678",
  "smiles": "CC(C)Nc1ncnc2[nH]ccc12",
  "source": "zinc",
  "type2_score": 0.923,
  "dfg_out_prob": 0.887,
  "binding_affinity_pred": 8.2,
  "price_per_mg": 15.50,
  "availability": "in-stock",
  "mw": 425.3,
  "logp": 3.2,
  "hbd": 2,
  "hba": 5
}
```

## Training the Model

To train the screening model on your own data:

```python
from hyaline.screening import Type2ScreeningModel, train_screening_model
from torch.utils.data import DataLoader

# Create model
model = Type2ScreeningModel(
    pocket_dim=16,
    compound_dim=8,
    hidden_dim=128,
)

# Train
trained_model = train_screening_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    lr=1e-3,
    device='cuda',
    checkpoint_dir='checkpoints/screening',
)
```

## Dependencies

- PyTorch 2.0+
- NumPy
- tqdm
- requests (for API calls)
- RDKit (optional, for SMILES processing)

## References

- ZINC Database: https://zinc.docking.org
- Enamine REAL: https://enamine.net/compound-collections/real-compounds
- Type II Kinase Inhibitors: https://doi.org/10.1021/jm501389q
