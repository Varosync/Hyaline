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

# Deep-learning components (kinase GNN models, feature extractors, GPCR data)
# are optional: they require torch / torch_geometric / h5py. The kinase
# `analyze` path is intentionally dependency-light (numpy + requests only), so
# a kinase-only install must be able to `import hyaline` without these present.
try:
    from .models.spiking_egnn import SpikingEGNN, SpikingEGNNConfig, SpikingEGNNLayer
    from .models.kinase_binding import (
        KinaseBindingPredictor,
        KinaseBindingConfig,
        KLIFSLoader,
    )
    from .models.conformational_prior import ConformationalPrior, ConformationalPriorConfig
    from .features.geometric import GeometricFeatureExtractor, extract_from_pdb_file
    from .features.classical import ClassicalFeatureExtractor, NormalModeGenerator
    from .data import load_dataset_with_motifs
    _HAS_DEEP_LEARNING = True
except ImportError:
    SpikingEGNN = SpikingEGNNConfig = SpikingEGNNLayer = None
    KinaseBindingPredictor = KinaseBindingConfig = KLIFSLoader = None
    ConformationalPrior = ConformationalPriorConfig = None
    GeometricFeatureExtractor = extract_from_pdb_file = None
    ClassicalFeatureExtractor = NormalModeGenerator = None
    load_dataset_with_motifs = None
    _HAS_DEEP_LEARNING = False

__version__ = "2.2.0"
__all__ = [
    # Kinase
    'KinaseBindingPredictor',
    'KinaseBindingConfig',
    'KLIFSLoader',
    'SpikingEGNN',
    'SpikingEGNNConfig',
    'SpikingEGNNLayer',
    'ConformationalPrior',
    'ConformationalPriorConfig',
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
