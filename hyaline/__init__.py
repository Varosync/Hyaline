"""
Hyaline: SE(3) Equivariant Deep Learning for Molecular Pocket Prediction
=========================================================================

Key Components:
- TFActivationModel: Context-dependent TF activation predictor (SCENIC+)
- TFModulator: Model for TF-DNA pocket detection
- HybridEGNN: Model for cryptic pockets with dynamics
- HyalineV2: Legacy GPCR model

Data & Features:
- PDBMiner: Mine TF-DNA structures from RCSB
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

# TF-Modulator components (optional - requires torch_geometric)
try:
    from .models.tf_modulator import TFModulator, TFModulatorConfig
    from .models.hybrid_egnn import HybridEGNN, HybridEGNNWithAttention
except ImportError:
    TFModulator = None
    TFModulatorConfig = None
    HybridEGNN = None
    HybridEGNNWithAttention = None

# Context-dependent TF Activation (SCENIC+ integration)
from .models.context_encoder import ContextEncoder, ContextEncoderConfig, ContextEncoderOutput
from .models.spike_encoder import SpikeEncoder, SpikeEncoderConfig, SpikeEncoderOutput
from .models.spiking_egnn import SpikingEGNN, SpikingEGNNConfig, SpikingEGNNLayer
from .models.activation_head import ActivationHead, ActivationHeadConfig, ActivationHeadOutput
from .models.tf_activation_model import TFActivationModel, TFActivationConfig, TFActivationOutput

# Feature extractors
from .features.geometric import GeometricFeatureExtractor, extract_from_pdb_file
from .features.classical import ClassicalFeatureExtractor, NormalModeGenerator

# Data pipeline (import from submodule to avoid conflict with data.py)
from .loaders import pdb_mining

__version__ = "2.1.0"
__all__ = [
    # TF Activation (SCENIC+ context-dependent)
    'TFActivationModel',
    'TFActivationConfig',
    'TFActivationOutput',
    'ContextEncoder',
    'ContextEncoderConfig',
    'ContextEncoderOutput',
    'SpikeEncoder',
    'SpikeEncoderConfig',
    'SpikeEncoderOutput',
    'SpikingEGNN',
    'SpikingEGNNConfig',
    'SpikingEGNNLayer',
    'ActivationHead',
    'ActivationHeadConfig',
    'ActivationHeadOutput',
    
    # TF-Modulator
    'TFModulator',
    'TFModulatorConfig',
    'HybridEGNN',
    'HybridEGNNWithAttention',
    
    # Legacy (if torch_geometric available)
    'HyalineV2',
    'count_params_v2',
    
    # Features
    'GeometricFeatureExtractor',
    'extract_from_pdb_file',
    'ClassicalFeatureExtractor',
    'NormalModeGenerator',
    
    # Data
    'pdb_mining',
]
