"""
Data Loading Module
===================

Provides data loading utilities for Hyaline training and inference.
"""
import torch
import h5py
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from hyaline.graph_data import GPCRGraphDataset
from hyaline.motifs import get_motif_type_embedding


def load_dataset_with_motifs(data_dir='data/gpcrdb_all', cutoff=10.0):
    """
    Load GPCR dataset with motif type features for HyalineV2.
    
    Args:
        data_dir: Path to data directory containing PDB files and H5 embeddings
        cutoff: Distance cutoff for radius graph (Angstroms)
    
    Returns:
        data_list: List of PyG Data objects with motif_types attribute
        labels: Array of labels (0=inactive, 1=active)
        families: List of GPCR family names
        sequences: List of amino acid sequences
    """
    print("Loading dataset...")
    dataset = GPCRGraphDataset(root=data_dir, cutoff=cutoff)
    
    # Load sequences for motif detection
    h5_path = f'{data_dir}/esm3_receptor_only.h5'
    with h5py.File(h5_path, 'r') as f:
        sequences = [s.decode() if isinstance(s, bytes) else s for s in f['sequences'][:]]
        labels = f['labels'][:]
        # Try to load family info if available
        if 'families' in f:
            families = [s.decode() if isinstance(s, bytes) else s for s in f['families'][:]]
        else:
            families = ['Unknown'] * len(labels)
    
    print("Adding motif features...")
    data_list = []
    
    for i in tqdm(range(len(dataset)), desc="Processing"):
        data = dataset[i]
        seq = sequences[i]
        
        # Add motif types
        motif_types = get_motif_type_embedding(seq)
        if len(motif_types) > data.num_nodes:
            motif_types = motif_types[:data.num_nodes]
        elif len(motif_types) < data.num_nodes:
            padding = torch.zeros(data.num_nodes - len(motif_types), dtype=torch.long)
            motif_types = torch.cat([motif_types, padding])
        
        data.motif_types = motif_types
        data_list.append(data)
    
    print(f"Loaded {len(data_list)} structures")
    return data_list, labels, families, sequences


def create_data_loader(data_list, batch_size=8, shuffle=True):
    """Create a PyG DataLoader from data list."""
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
