"""
Hyaline: Geometric Deep Learning for Protein Function Modeling
==============================================================

GPCR Activation (original):
- HyalineV2: Predicts GPCR activation state from PDB structure using
  E(n)-equivariant GNN with ESM3 embeddings and GPCR motif attention.

Transcription Factor Modeling (new):
- HyalineTF: Maps TF sequence (+ optional structure) to TF function
  class, DNA-binding affinity, and downstream regulatory impact.
  Supports sequence-only and structure-augmented inputs.
  Uses ESM-class embeddings with TF domain-aware attention biasing.

Key components:
- model_v2:        HyalineV2 (GPCR)
- tf_model:        HyalineTF (transcription factors)
- motifs:          GPCR motif + TF domain detection
- sota_enhancements: RBF expansion, attention modules, EGNN layers
- tf_data:         TF data utilities (FASTA loading, ESM embeddings)
"""

from .model_v2 import HyalineV2, count_parameters
from .data import load_dataset_with_motifs
from .tf_model import HyalineTF, TF_FUNCTION_CLASSES, count_tf_parameters
from .tf_data import sequence_to_data, load_tf_sequences, get_esm_embeddings

__version__ = "2.1.0"
__all__ = [
    # GPCR (original)
    'HyalineV2',
    'count_parameters',
    'load_dataset_with_motifs',
    # Transcription factor (new)
    'HyalineTF',
    'TF_FUNCTION_CLASSES',
    'count_tf_parameters',
    'sequence_to_data',
    'load_tf_sequences',
    'get_esm_embeddings',
]
