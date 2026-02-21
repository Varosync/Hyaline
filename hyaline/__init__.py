"""
Hyaline: SE(3) Equivariant Deep Learning for Molecular Pocket Prediction
=========================================================================

Key Components:
- HyalineV2: GPCR activation predictor (legacy)
- KinaseBindingPredictor: Kinase DFG conformational binding (KLIFS)
- SpikingEGNN: SE(3) spiking message passing for kinase tasks

Data & Features:
- KLIFS loaders: Kinase structures from KLIFS database
- GeometricFeatureExtractor: Static node/edge features
- ClassicalFeatureExtractor: MD dynamics features
"""

# Legacy GPCR models (optional - requires torch_geometric)
try:
    from .model_v2 import HyalineV2, count_parameters as count_params_v2
    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    HyalineV2 = None
    count_params_v2 = None
    _HAS_TORCH_GEOMETRIC = False

# Kinase models
from .models.spiking_egnn import SpikingEGNN, SpikingEGNNConfig, SpikingEGNNLayer
from .models.kinase_binding import (
    KinaseBindingPredictor,
    KinaseBindingConfig,
    KLIFSLoader,
)

# Feature extractors
from .features.geometric import GeometricFeatureExtractor, extract_from_pdb_file
from .features.classical import ClassicalFeatureExtractor, NormalModeGenerator

# GPCR data (legacy)
from .data import load_dataset_with_motifs

__version__ = "2.1.0"
__all__ = [
    # Kinase
    'KinaseBindingPredictor',
    'KinaseBindingConfig',
    'KLIFSLoader',
    'SpikingEGNN',
    'SpikingEGNNConfig',
    'SpikingEGNNLayer',
    # Legacy GPCR
    'HyalineV2',
    'count_params_v2',
    # Features
    'GeometricFeatureExtractor',
    'extract_from_pdb_file',
    'ClassicalFeatureExtractor',
    'NormalModeGenerator',
    'load_dataset_with_motifs',
]
