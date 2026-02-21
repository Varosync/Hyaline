"""
Geometric Feature Extraction for TF-Modulator
==============================================

Static geometric and chemical features for molecular graphs.
These are the primary features for TF-DNA pocket prediction.

Features:
- Element type (one-hot)
- Partial charge (Gasteiger)
- Hybridization state (one-hot)
- Residue type (one-hot)
- Bond type (covalent, H-bond, vdW)
- Interface flag (TF-DNA boundary)
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

# Atom element types (heavy atoms in proteins/DNA)
ELEMENT_TYPES = ['C', 'N', 'O', 'S', 'P']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENT_TYPES)}

# Hybridization states
HYBRIDIZATION_STATES = ['sp', 'sp2', 'sp3', 'other']

# 20 standard amino acids
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# DNA bases (for interface detection)
DNA_BASES = ['DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C']


@dataclass
class GeometricFeatures:
    """Container for extracted geometric features."""
    
    node_features: np.ndarray  # [N, node_dim]
    edge_features: np.ndarray  # [E, edge_dim]
    edge_index: np.ndarray     # [2, E]
    pos: np.ndarray            # [N, 3]
    
    # Masks
    is_dna: np.ndarray         # [N] boolean
    is_ca: np.ndarray          # [N] boolean (Cα atoms)
    
    # Metadata
    residue_ids: List[str]     # Per-node residue identifiers
    chain_ids: List[str]       # Per-node chain IDs


class GeometricFeatureExtractor:
    """
    Extract static geometric features from molecular structures.
    
    This extractor computes:
    - Node features: element, charge, hybridization, residue type
    - Edge features: distance, bond type, interface flag
    """
    
    def __init__(
        self,
        protein_cutoff: float = 8.0,
        dna_cutoff: float = 12.0,
        include_hydrogens: bool = False,
    ):
        """
        Args:
            protein_cutoff: Distance cutoff for protein-protein edges (Angstroms)
            dna_cutoff: Distance cutoff for protein-DNA edges (Angstroms)
            include_hydrogens: Whether to include hydrogen atoms
        """
        self.protein_cutoff = protein_cutoff
        self.dna_cutoff = dna_cutoff
        self.include_hydrogens = include_hydrogens
    
    def extract_from_biopython(
        self,
        structure,
        chain_ids: Optional[List[str]] = None,
    ) -> GeometricFeatures:
        """
        Extract features from a BioPython Structure object.
        
        Args:
            structure: BioPython Structure (from PDBParser)
            chain_ids: Optional list of chains to include
            
        Returns:
            GeometricFeatures object
        """
        from Bio.PDB import is_aa
        
        atoms = []
        coords = []
        residue_types = []
        chain_list = []
        is_dna_mask = []
        is_ca_mask = []
        element_indices = []
        
        for model in structure:
            for chain in model:
                if chain_ids and chain.id not in chain_ids:
                    continue
                    
                for residue in chain:
                    res_name = residue.get_resname().strip()
                    is_dna_res = res_name in DNA_BASES
                    
                    for atom in residue:
                        # Skip hydrogens if not included
                        element = atom.element.strip().upper()
                        if not self.include_hydrogens and element == 'H':
                            continue
                        
                        # Skip non-heavy atoms not in our list
                        if element not in ELEMENT_TYPES:
                            continue
                        
                        atoms.append(atom)
                        coords.append(atom.coord)
                        residue_types.append(res_name)
                        chain_list.append(chain.id)
                        is_dna_mask.append(is_dna_res)
                        is_ca_mask.append(atom.name == 'CA')
                        element_indices.append(ELEMENT_TO_IDX.get(element, 0))
        
        if len(atoms) == 0:
            raise ValueError("No atoms found in structure")
        
        coords = np.array(coords, dtype=np.float32)
        is_dna = np.array(is_dna_mask, dtype=bool)
        is_ca = np.array(is_ca_mask, dtype=bool)
        
        # Build node features
        node_features = self._build_node_features(
            element_indices, residue_types, atoms
        )
        
        # Build edges and edge features
        edge_index, edge_features = self._build_edges(
            coords, is_dna
        )
        
        return GeometricFeatures(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
            pos=coords,
            is_dna=is_dna,
            is_ca=is_ca,
            residue_ids=[f"{c}:{r}" for c, r in zip(chain_list, residue_types)],
            chain_ids=chain_list,
        )
    
    def extract_from_coords(
        self,
        coords: np.ndarray,
        elements: List[str],
        residue_types: List[str],
        is_dna: Optional[np.ndarray] = None,
    ) -> GeometricFeatures:
        """
        Extract features from raw coordinate data.
        
        Args:
            coords: [N, 3] atomic coordinates
            elements: List of element symbols per atom
            residue_types: List of residue names per atom
            is_dna: Optional boolean mask for DNA atoms
            
        Returns:
            GeometricFeatures object
        """
        N = len(coords)
        
        # Default: no DNA
        if is_dna is None:
            is_dna = np.array([r in DNA_BASES for r in residue_types])
        
        # Element indices
        element_indices = [ELEMENT_TO_IDX.get(e.upper(), 0) for e in elements]
        
        # Build node features (simplified without BioPython)
        node_features = self._build_node_features_simple(
            element_indices, residue_types
        )
        
        # Build edges
        edge_index, edge_features = self._build_edges(coords, is_dna)
        
        # Cα mask (approximate)
        is_ca = np.zeros(N, dtype=bool)
        
        return GeometricFeatures(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
            pos=coords.astype(np.float32),
            is_dna=is_dna,
            is_ca=is_ca,
            residue_ids=[f"?:{r}" for r in residue_types],
            chain_ids=['?'] * N,
        )
    
    def _build_node_features(
        self,
        element_indices: List[int],
        residue_types: List[str],
        atoms: list,
    ) -> np.ndarray:
        """Build node feature matrix using BioPython atoms."""
        N = len(atoms)
        
        # Element one-hot (5 dim)
        element_oh = np.zeros((N, len(ELEMENT_TYPES)), dtype=np.float32)
        for i, idx in enumerate(element_indices):
            element_oh[i, idx] = 1.0
        
        # Hybridization one-hot (4 dim) - estimate from element
        hybrid_oh = np.zeros((N, len(HYBRIDIZATION_STATES)), dtype=np.float32)
        for i, atom in enumerate(atoms):
            elem = atom.element.strip().upper()
            # Simple heuristic (proper hybridization requires RDKit)
            if elem == 'C':
                hybrid_oh[i, 2] = 1.0  # sp3 default
            elif elem == 'N':
                hybrid_oh[i, 1] = 1.0  # sp2 default
            elif elem == 'O':
                hybrid_oh[i, 2] = 1.0  # sp3 default
            else:
                hybrid_oh[i, 3] = 1.0  # other
        
        # Residue type one-hot (20 dim)
        residue_oh = np.zeros((N, len(AMINO_ACIDS)), dtype=np.float32)
        for i, res in enumerate(residue_types):
            if res in AA_TO_IDX:
                residue_oh[i, AA_TO_IDX[res]] = 1.0
        
        # Partial charge estimate (1 dim) - placeholder
        # Real implementation would use Gasteiger charges via RDKit
        charge = np.zeros((N, 1), dtype=np.float32)
        
        # Placeholder for dynamics features (2 dim) - filled later
        dynamics = np.zeros((N, 2), dtype=np.float32)
        
        # Concatenate: element(5) + hybrid(4) + residue(20) + charge(1) + dynamics(2) = 32
        return np.concatenate([
            element_oh,
            hybrid_oh, 
            residue_oh,
            charge,
            dynamics,
        ], axis=1)
    
    def _build_node_features_simple(
        self,
        element_indices: List[int],
        residue_types: List[str],
    ) -> np.ndarray:
        """Build node features without BioPython atoms."""
        N = len(element_indices)
        
        # Element one-hot
        element_oh = np.zeros((N, len(ELEMENT_TYPES)), dtype=np.float32)
        for i, idx in enumerate(element_indices):
            if 0 <= idx < len(ELEMENT_TYPES):
                element_oh[i, idx] = 1.0
        
        # Hybridization (default sp3)
        hybrid_oh = np.zeros((N, len(HYBRIDIZATION_STATES)), dtype=np.float32)
        hybrid_oh[:, 2] = 1.0  # sp3
        
        # Residue type
        residue_oh = np.zeros((N, len(AMINO_ACIDS)), dtype=np.float32)
        for i, res in enumerate(residue_types):
            if res in AA_TO_IDX:
                residue_oh[i, AA_TO_IDX[res]] = 1.0
        
        # Placeholders
        charge = np.zeros((N, 1), dtype=np.float32)
        dynamics = np.zeros((N, 2), dtype=np.float32)
        
        return np.concatenate([
            element_oh, hybrid_oh, residue_oh, charge, dynamics
        ], axis=1)
    
    def _build_edges(
        self,
        coords: np.ndarray,
        is_dna: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build edges with interface-aware cutoffs."""
        N = len(coords)
        
        # Compute pairwise distances
        diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 3]
        dist_matrix = np.linalg.norm(diff, axis=-1)  # [N, N]
        
        # Interface mask
        is_dna_col = is_dna[:, None]
        is_dna_row = is_dna[None, :]
        crosses_interface = (is_dna_col != is_dna_row)
        
        # Apply different cutoffs
        cutoff_matrix = np.where(
            crosses_interface,
            self.dna_cutoff,
            self.protein_cutoff,
        )
        
        # Build adjacency (exclude self-loops)
        adj = (dist_matrix < cutoff_matrix) & (dist_matrix > 0.1)
        
        # To edge index
        row, col = np.where(adj)
        edge_index = np.stack([row, col], axis=0)
        
        E = edge_index.shape[1]
        
        # Edge features
        # Distance (1 dim)
        distances = dist_matrix[row, col].reshape(-1, 1)
        
        # Bond type one-hot (3 dim): covalent, H-bond, vdW
        bond_type = np.zeros((E, 3), dtype=np.float32)
        covalent_mask = distances.flatten() < 1.8
        hbond_mask = (distances.flatten() >= 1.8) & (distances.flatten() < 3.5)
        vdw_mask = distances.flatten() >= 3.5
        bond_type[covalent_mask, 0] = 1.0
        bond_type[hbond_mask, 1] = 1.0
        bond_type[vdw_mask, 2] = 1.0
        
        # Interface flag (1 dim)
        interface_flag = crosses_interface[row, col].astype(np.float32).reshape(-1, 1)
        
        # Placeholder for dynamics (2 dim) - DCC, MI
        dynamics = np.zeros((E, 2), dtype=np.float32)
        
        # Total: distance(1) + bond(3) + interface(1) + dynamics(2) = 7
        edge_features = np.concatenate([
            distances,
            bond_type,
            interface_flag,
            dynamics,
        ], axis=1)
        
        return edge_index.astype(np.int64), edge_features.astype(np.float32)


def extract_from_pdb_file(
    pdb_path: str,
    chain_ids: Optional[List[str]] = None,
    **kwargs,
) -> GeometricFeatures:
    """
    Convenience function to extract features from a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        chain_ids: Optional list of chains to include
        **kwargs: Additional arguments for GeometricFeatureExtractor
        
    Returns:
        GeometricFeatures object
    """
    from Bio.PDB import PDBParser
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    extractor = GeometricFeatureExtractor(**kwargs)
    return extractor.extract_from_biopython(structure, chain_ids)


if __name__ == "__main__":
    print("Geometric Feature Extractor")
    print("=" * 50)
    
    extractor = GeometricFeatureExtractor()
    
    # Test with synthetic data
    N = 100
    coords = np.random.randn(N, 3) * 5
    elements = ['C'] * 60 + ['N'] * 20 + ['O'] * 20
    residue_types = ['ALA'] * 50 + ['GLY'] * 30 + ['DA'] * 20  # Some DNA
    
    features = extractor.extract_from_coords(coords, elements, residue_types)
    
    print(f"\nExtracted features:")
    print(f"  Nodes: {features.node_features.shape}")
    print(f"  Edges: {features.edge_index.shape[1]}")
    print(f"  Edge features: {features.edge_features.shape}")
    print(f"  DNA atoms: {features.is_dna.sum()}")
    print(f"  Protein atoms: {(~features.is_dna).sum()}")
    
    print("\n✓ Geometric feature extractor ready!")
