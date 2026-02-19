<div align="center">
  <h1>HYALINE</h1>
  <strong>Geometric Deep Learning for Protein Function Modeling</strong>
  <br><br>
  
  [Paper](https://www.biorxiv.org/content/10.64898/2026.01.05.697778v1) | [Data](Supplementary_Data_1.csv)
  <br><br>
</div>

<p align="center">
  <img src="docs/protein_comparison.png" alt="GPCR Structure Prediction" width="500"/>
</p>

HYALINE is Varosync's sequence-to-function modeling framework for proteins.
It currently ships two model families built on shared geometric + equivariant
representation learning machinery:

| Model | Task | Input |
|-------|------|-------|
| **HyalineV2** | GPCR activation state prediction | PDB structure |
| **HyalineTF** | TF function, DNA-binding & regulatory impact | Sequence (+ optional structure) |

## GPCR Activation (HyalineV2)

HYALINE predicts whether a G protein-coupled receptor (GPCR) structure is in an **active** or **inactive** conformational state using E(n)-equivariant graph neural networks. GPCRs are the largest family of drug targets, with ~35% of FDA-approved drugs acting on these receptors.

The model achieves **0.995 AuROC** on cross-validation and **0.819 AuROC** on a temporal held-out test set (structures released 2023–2024), outperforming sequence-only baselines by 6–12%. Notably, HYALINE achieves 87.2% accuracy on Class C GPCRs where sequence-based methods fail (39.4%).

For technical details, see the [paper](https://www.biorxiv.org/content/10.64898/2026.01.05.697778v1).

## Transcription Factor Modeling (HyalineTF)

**HyalineTF** maps TF sequence (and optional structural priors) to functionally
meaningful outputs: TF function class, DNA-binding affinity, and downstream
regulatory impact.

### Technical approach

- **Equivariant / geometric deep learning** when Cα structure is available
  (same E(n)-equivariant GNN backbone as HyalineV2)
- **Protein language model embeddings** (ESM2 / ESM3) as a strong sequence
  prior when structure is unavailable; uses attention-weighted pooling
- **TF domain-aware attention biasing** for interpretability — prioritizes
  zinc fingers, homeodomains, leucine zippers, bHLH basic regions, WRKY
  domains, and ETS helices

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `function_logits` | `[B, 3]` | Activator / repressor / dual classification |
| `binding` | `[B]` | DNA-binding affinity score |
| `regulatory` | `[B]` | Downstream regulatory impact score |

## Installation

```bash
pip install git+https://github.com/Varosync/Hyaline.git
```

Or from source:
```bash
git clone https://github.com/Varosync/Hyaline.git
cd Hyaline; pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.0+, PyTorch Geometric

## Inference

**GPCR activation** (structure-based):
```bash
hyaline predict structure.pdb
```
```
HYALINE PREDICTION
  Score:       0.9521
  Prediction:  Active
  Confidence:  High
```

**TF function** (sequence-based):
```bash
hyaline tf-predict tf_sequences.fasta
```
```
HYALINE TF PREDICTION
  TP53_HUMAN
    TF function:    activator  (conf 0.81)
    DNA binding:    1.4231
    Regulatory:     0.9872
```

**Python API:**
```python
from hyaline import HyalineTF
from hyaline.tf_data import load_tf_sequences, get_esm_embeddings, sequence_to_data

# Load a trained checkpoint for meaningful predictions
model = HyalineTF.from_pretrained("checkpoints/hyaline_tf.pt")
# For architecture exploration without a checkpoint (predictions will be random):
# model = HyalineTF(node_input_dim=1280, hidden_dim=256)

sequences = load_tf_sequences("tf_sequences.fasta")
embeddings = get_esm_embeddings([s for _, s in sequences])

for (name, seq), emb in zip(sequences, embeddings):
    data = sequence_to_data(seq, emb)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    out = model(data)
    print(name, out['function_logits'], out['binding'], out['regulatory'])
```

## Architecture

<p align="center">
  <img src="docs/egnn_layer.png" alt="Enhanced EGNN Layer" width="700"/>
</p>

Both models share the same enhanced E(n)-equivariant GNN core:
- **ESM embeddings** (1280/1536-dim) for sequence features
- **RBF distance encoding** for spatial relationships
- **Learned domain/motif attention** that prioritizes biologically relevant regions

## Data

The curated GPCR dataset contains **1,596 GPCR structures** from the Protein Data Bank with activation state annotations from GPCRdb. The complete list of PDB IDs with train/test splits is provided in [`Supplementary_Data_1.csv`](Supplementary_Data_1.csv).

## Citation

If you use HYALINE in your research, please cite:

```bibtex
@article{hyaline2026,
  title   = {HYALINE: Geometric Deep Learning for Accurate Prediction of 
             G Protein-Coupled Receptor Activation States from Structure},
  author  = {Varosync},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.01.05.697778}
}
```

## License

MIT License
