"""
Graph Data Generation for Hyaline
======================================

Converts PDB structures to PyG graph format with:
- Nodes: Cα coordinates + ESM3 embeddings
- Edges: Residue pairs within 10Å with geometric features
"""
import numpy as np
import h5py
import torch
from torch_geometric.data import Data, InMemoryDataset
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def extract_ca_coords(pdb_path: str, target_chain: str = None) -> np.ndarray:
    """
    Extract Cα coordinates from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        target_chain: If specified, only extract from this chain
    
    Returns:
        coords: [N, 3] Cα coordinates
    """
    coords = []
    seen_res = set()
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                
                chain = line[21]
                if target_chain and chain != target_chain:
                    continue
                
                res_id = f"{chain}_{line[22:27].strip()}"
                if res_id in seen_res:
                    continue
                seen_res.add(res_id)
                
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    
    return np.array(coords, dtype=np.float32)


def build_radius_edges(
    coords: np.ndarray,
    cutoff: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edge list for residues within cutoff distance.
    
    Args:
        coords: [N, 3] Cα coordinates
        cutoff: Distance cutoff in Angstroms
    
    Returns:
        edge_index: [2, E] source and target indices
        distances: [E] pairwise distances
    """
    N = len(coords)
    
    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 3]
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # [N, N]
    
    # Find edges within cutoff (excluding self-loops)
    mask = (dist < cutoff) & (dist > 0)
    sources, targets = np.where(mask)
    
    edge_index = np.stack([sources, targets], axis=0)
    distances = dist[sources, targets]
    
    return edge_index, distances


def compute_legacy_edge_features(
    coords: np.ndarray,
    residue_energies: np.ndarray,
    residue_coords: np.ndarray,
    edge_index: np.ndarray
) -> np.ndarray:
    """
    LEGACY: Compute edge features (used in original dataset generation).
    
    Note: The production model uses only distance² features.
    This function is retained for dataset regeneration compatibility.
    
    Features:
    1. Squared distance: ||xi - xj||²
    2. Sum of residue energies: ei + ej (legacy, not used in production)
    3. Product of coordination numbers: ci * cj (legacy, not used in production)
    
    Args:
        coords: [N, 3] Cα coordinates
        residue_energies: [N] per-residue dispersion energy
        residue_coords: [N] per-residue coordination number
        edge_index: [2, E]
    
    Returns:
        edge_features: [E, 3]
    """
    sources, targets = edge_index
    
    # Squared distance
    diff = coords[sources] - coords[targets]
    dist_sq = (diff ** 2).sum(axis=-1, keepdims=True)
    
    # Legacy energy features (not used in production model)
    energy_sum = (residue_energies[sources] + residue_energies[targets]).reshape(-1, 1)
    
    # Legacy coordination features (not used in production model)
    coord_prod = (residue_coords[sources] * residue_coords[targets]).reshape(-1, 1)
    
    edge_features = np.concatenate([dist_sq, energy_sum, coord_prod], axis=-1)
    
    return edge_features.astype(np.float32)


class GPCRGraphDataset(InMemoryDataset):
    """
    PyG Dataset for GPCR structures as graphs.
    
    Each graph has:
    - x: ESM3 embeddings [N, 1536]
    - pos: Cα coordinates [N, 3]
    - edge_index: [2, E]
    - edge_attr: [E, 3] (dist², energy_sum, coord_prod)
    - y: Label (0 or 1)
    """
    
    def __init__(
        self,
        root: str = 'data/gpcrdb_all',
        cutoff: float = 10.0,
        transform=None,
        pre_transform=None
    ):
        self.cutoff = cutoff
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        # Note: alchemi file only needed for regenerating cached dataset
        # Production model uses only distance features from cached graphs
        return ['esm3_receptor_only.h5', 'alchemi_receptor_only.h5']
    
    @property
    def processed_file_names(self):
        return [f'gpcr_graphs_{self.cutoff}A.pt']
    
    def download(self):
        pass
    
    def process(self):
        """Convert all structures to graphs."""
        # Legacy dataset generation (cached graphs already exist)
        # Production model uses only column 0 (distance²) from cached graphs
        esm_path = Path(self.root) / 'esm3_receptor_only.h5'
        legacy_path = Path(self.root) / 'alchemi_receptor_only.h5'
        
        print(f"Loading data from {esm_path}...")
        
        with h5py.File(esm_path, 'r') as f:
            embeddings = f['embeddings'][:]
            pdb_ids = [p.decode() for p in f['pdb_ids'][:]]
            labels = f['labels'][:]
            seq_lengths = f['seq_lengths'][:]
        
        with h5py.File(legacy_path, 'r') as f:
            residue_energies = f['residue_energies'][:]
            residue_coords = f['residue_coordinations'][:]
        
        data_list = []
        skipped = 0
        
        print(f"Building graphs for {len(pdb_ids)} structures (cutoff={self.cutoff}Å)...")
        
        for i, pdb_id in enumerate(tqdm(pdb_ids, desc="Building graphs")):
            pdb_path = Path(self.root) / f"{pdb_id}.pdb"
            
            if not pdb_path.exists():
                skipped += 1
                continue
            
            try:
                # Extract Cα coordinates
                coords = extract_ca_coords(str(pdb_path))
                seq_len = min(len(coords), seq_lengths[i], 512)
                
                if seq_len < 50:  # Skip very short structures
                    skipped += 1
                    continue
                
                coords = coords[:seq_len]
                
                # Build edges
                edge_index, distances = build_radius_edges(coords, self.cutoff)
                
                if len(edge_index[0]) < 10:  # Skip if too sparse
                    skipped += 1
                    continue
                
                # Get node features
                node_features = embeddings[i, :seq_len]
                
                # Get edge features
                res_energies = residue_energies[i, :seq_len]
                res_coords = residue_coords[i, :seq_len]
                edge_features = compute_legacy_edge_features(
                    coords, res_energies, res_coords, edge_index
                )
                
                # Normalize edge features
                edge_features[:, 0] /= 100.0  # dist² in 100s
                edge_features[:, 1] /= 100.0  # energy sum
                edge_features[:, 2] /= 100.0  # coord product
                
                # Create PyG Data object
                data = Data(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    pos=torch.tensor(coords, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_features, dtype=torch.float32),
                    y=torch.tensor(labels[i], dtype=torch.float32),
                    pdb_id=pdb_id
                )
                
                data_list.append(data)
                
            except Exception as e:
                print(f"  Error processing {pdb_id}: {e}")
                skipped += 1
                continue
        
        print(f"Built {len(data_list)} graphs, skipped {skipped}")
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_graph_dataset(
    root: str = 'data/gpcrdb_all',
    cutoff: float = 10.0,
    force_reprocess: bool = False
) -> GPCRGraphDataset:
    """
    Create or load GPCR graph dataset.
    
    Args:
        root: Data directory
        cutoff: Edge cutoff in Angstroms
        force_reprocess: If True, regenerate graphs
    
    Returns:
        dataset: GPCRGraphDataset
    """
    processed_path = Path(root) / 'processed' / f'gpcr_graphs_{cutoff}A.pt'
    
    if force_reprocess and processed_path.exists():
        processed_path.unlink()
    
    return GPCRGraphDataset(root=root, cutoff=cutoff)


if __name__ == '__main__':
    # Generate graph dataset
    print("=" * 60)
    print("GENERATING GPCR GRAPH DATASET")
    print("=" * 60)
    
    dataset = create_graph_dataset(cutoff=10.0, force_reprocess=True)
    
    print(f"\nDataset: {len(dataset)} graphs")
    
    # Show example
    sample = dataset[0]
    print(f"\nExample graph:")
    print(f"  Nodes: {sample.x.shape[0]}")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Node features: {sample.x.shape}")
    print(f"  Edge features: {sample.edge_attr.shape}")
    print(f"  Coordinates: {sample.pos.shape}")
    print(f"  Label: {sample.y.item()}")
