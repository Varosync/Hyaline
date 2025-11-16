"""
Hyaline: Geometric Deep Learning for GPCR Activation State Prediction
=====================================================================

This package provides the HyalineV2 model for predicting GPCR activation
states from PDB structures using E(n)-equivariant graph neural networks
with ESM3 protein language model embeddings.

Key Components:
- HyalineV2: Production model (V2-D configuration)
- graph_data: Data processing and graph construction
- motifs: GPCR motif detection (DRY, NPxxY, CWxP)
- sota_enhancements: RBF expansion, attention modules
"""

from .model_v2 import HyalineV2, count_parameters
from .data import load_dataset_with_motifs

__version__ = "2.0.0"
__all__ = [
    'HyalineV2',
    'count_parameters',
    'load_dataset_with_motifs',
]
