"""
Hyaline Models
==============

SE(3) Equivariant Graph Neural Networks for kinase conformational binding
and molecular pocket prediction.
"""

from .spiking_egnn import (
    SpikingEGNN,
    SpikingEGNNConfig,
    SpikingEGNNLayer,
)

from .kinase_binding import (
    KinaseBindingPredictor,
    KinaseBindingConfig,
    KLIFSLoader,
)

from .conformational_prior import (
    ConformationalPrior,
    ConformationalPriorConfig,
)

__all__ = [
    'SpikingEGNN',
    'SpikingEGNNConfig',
    'SpikingEGNNLayer',
    'KinaseBindingPredictor',
    'KinaseBindingConfig',
    'KLIFSLoader',
    'ConformationalPrior',
    'ConformationalPriorConfig',
]
