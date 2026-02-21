"""
Hyaline Models
==============

SE(3) Equivariant Graph Neural Networks for molecular pocket prediction
and context-dependent TF activation prediction.
"""

# Optional imports requiring torch_geometric
try:
    from .hybrid_egnn import (
        HybridEGNN,
        HybridEGNNWithAttention,
        EGNNLayer,
        build_graph_from_coords,
    )
    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    HybridEGNN = None
    HybridEGNNWithAttention = None
    EGNNLayer = None
    build_graph_from_coords = None
    _HAS_TORCH_GEOMETRIC = False

try:
    from .tf_modulator import (
        TFModulator,
        TFModulatorConfig,
        PocketHead,
        DruggabilityHead,
        build_tf_graph,
        count_parameters,
    )
except ImportError:
    TFModulator = None
    TFModulatorConfig = None
    PocketHead = None
    DruggabilityHead = None
    build_tf_graph = None
    count_parameters = None

# Context-dependent TF activation (NEW)
from .context_encoder import (
    ContextEncoder,
    ContextEncoderConfig,
    ContextEncoderOutput,
)

from .spike_encoder import (
    SpikeEncoder,
    SpikeEncoderConfig,
    SpikeEncoderOutput,
)

from .spiking_egnn import (
    SpikingEGNN,
    SpikingEGNNConfig,
    SpikingEGNNLayer,
)

from .activation_head import (
    ActivationHead,
    ActivationHeadConfig,
    ActivationHeadOutput,
)

from .tf_activation_model import (
    TFActivationModel,
    TFActivationConfig,
    TFActivationOutput,
)

__all__ = [
    # Hybrid EGNN (for cryptic pockets)
    'HybridEGNN',
    'HybridEGNNWithAttention',
    'EGNNLayer',
    'build_graph_from_coords',
    
    # TF-Modulator (for TF-DNA complexes)
    'TFModulator',
    'TFModulatorConfig',
    'PocketHead',
    'DruggabilityHead',
    'build_tf_graph',
    'count_parameters',
    
    # Context-dependent TF Activation (NEW)
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
    'TFActivationModel',
    'TFActivationConfig',
    'TFActivationOutput',
]
