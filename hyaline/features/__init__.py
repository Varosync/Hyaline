"""
Hyaline Feature Extraction
==========================

Feature extraction for molecular pocket prediction.
"""

from .classical import (
    ClassicalFeatureExtractor,
    NormalModeGenerator,
    TrajectoryFeatures,
)

from .geometric import (
    GeometricFeatureExtractor,
    GeometricFeatures,
    extract_from_pdb_file,
    ELEMENT_TYPES,
    AMINO_ACIDS,
    DNA_BASES,
)

from .klifs_client import (
    KLIFSClient,
    StructureInfo,
)

from .pocket_descriptors import (
    PocketDescriptors,
    compute_pocket_descriptors,
    compute_pocket_volume,
    compute_esp_proxy,
    compute_hydrophobicity,
)

from .kinase_geometry import (
    KinaseGeometry,
    compute_kinase_geometry,
    compute_dfg_chelix_distance,
    compute_hinge_activation_angle,
)

__all__ = [
    # Classical dynamics features
    'ClassicalFeatureExtractor',
    'NormalModeGenerator',
    'TrajectoryFeatures',
    
    # Geometric features
    'GeometricFeatureExtractor',
    'GeometricFeatures',
    'extract_from_pdb_file',
    'ELEMENT_TYPES',
    'AMINO_ACIDS',
    'DNA_BASES',

    # KLIFS client
    'KLIFSClient',
    'StructureInfo',

    # Pocket descriptors
    'PocketDescriptors',
    'compute_pocket_descriptors',
    'compute_pocket_volume',
    'compute_esp_proxy',
    'compute_hydrophobicity',

    # Kinase geometry
    'KinaseGeometry',
    'compute_kinase_geometry',
    'compute_dfg_chelix_distance',
    'compute_hinge_activation_angle',
]
