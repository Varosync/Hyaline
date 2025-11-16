#!/usr/bin/env python3
"""
Hyaline Prediction Module
=========================

Standalone prediction functionality for GPCR activation state.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from hyaline import HyalineV2


# Production V2-D config
V2D_CONFIG = {
    'node_input_dim': 1536,
    'edge_input_dim': 3,
    'hidden_dim': 320,
    'num_layers': 5,
    'num_heads': 4,
    'num_rbf': 96,
    'cutoff': 10.0,
    'dropout': 0.15,
    'update_coords': True,
    'use_motif_bias': True,
    'use_multiscale': False
}

# Default checkpoint path
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / 'checkpoints' / 'hyaline.pt'


def parse_pdb(pdb_path: str) -> Tuple[str, np.ndarray, str]:
    """Parse PDB file to extract sequence and coordinates."""
    coords = []
    sequence = []
    chain = None
    seen_res = set()
    
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                res_name = line[17:20].strip()
                res_id = line[22:27].strip()
                chain = line[21]
                
                if res_id in seen_res:
                    continue
                seen_res.add(res_id)
                
                if res_name in aa_map:
                    sequence.append(aa_map[res_name])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    
    return ''.join(sequence), np.array(coords, dtype=np.float32), chain or 'A'


def get_esm3_embeddings(sequence: str, device: str = 'cuda') -> np.ndarray:
    """Get ESM3 embeddings for a sequence."""
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein
        
        model = ESM3.from_pretrained("esm3-open").to(device)
        protein = ESMProtein(sequence=sequence)
        
        with torch.no_grad():
            output = model.encode(protein)
            embeddings = output.sequence_embeddings.cpu().numpy()
        
        return embeddings.squeeze(0)
    except Exception as e:
        print(f"ESM3 error: {e}")
        print("Using random embeddings (for testing only)")
        return np.random.randn(len(sequence), 1536).astype(np.float32)


def build_radius_edges(coords: np.ndarray, cutoff: float = 10.0):
    """Build edges for residues within cutoff distance."""
    N = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    mask = (dist < cutoff) & (dist > 0)
    sources, targets = np.where(mask)
    edge_index = np.stack([sources, targets], axis=0)
    distances = dist[sources, targets]
    return edge_index, distances


def predict(
    pdb_path: str, 
    checkpoint_path: Optional[str] = None, 
    device: str = 'cuda'
) -> Tuple[Optional[float], Optional[str]]:
    """
    Predict GPCR activation state.
    
    Args:
        pdb_path: Path to PDB file
        checkpoint_path: Path to model checkpoint (optional)
        device: 'cuda' or 'cpu'
    
    Returns:
        score: Activation probability (0-1)
        prediction: 'Active' or 'Inactive'
    """
    from torch_geometric.data import Data
    
    print("=" * 60)
    print("HYALINE PREDICTION")
    print("=" * 60)
    
    # Parse PDB
    print(f"\nInput: {pdb_path}")
    sequence, ca_coords, chain = parse_pdb(pdb_path)
    n_residues = len(ca_coords)
    print(f"Residues: {n_residues}")
    
    if n_residues < 50:
        print("Error: Structure too short for GPCR prediction")
        return None, None
    
    # Build edges
    edge_index, distances = build_radius_edges(ca_coords, cutoff=10.0)
    print(f"Edges: {edge_index.shape[1]}")
    
    # Get embeddings
    print("\nComputing ESM3 embeddings...")
    node_features = get_esm3_embeddings(sequence, device)
    node_features = node_features[:n_residues]
    
    # Edge features
    dist_sq = (distances ** 2).astype(np.float32) / 100.0
    edge_features = np.stack([
        dist_sq,
        distances / 10.0,
        np.ones_like(distances)
    ], axis=-1)
    
    # Create graph
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        pos=torch.tensor(ca_coords, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        batch=torch.zeros(n_residues, dtype=torch.long)
    ).to(device)
    
    # Load model
    print("\nLoading model...")
    model = HyalineV2(**V2D_CONFIG).to(device)
    
    ckpt_path = checkpoint_path or str(DEFAULT_CHECKPOINT)
    if Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded: {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found: {ckpt_path}")
        print("Using untrained model (results will be random)")
    
    model.eval()
    
    # Predict
    with torch.no_grad():
        logits, _ = model(data)
        score = torch.sigmoid(logits).item()
    
    prediction = 'Active' if score > 0.5 else 'Inactive'
    
    # Confidence
    if score > 0.90 or score < 0.10:
        confidence = "High"
    elif score > 0.75 or score < 0.25:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Score:       {score:.4f}")
    print(f"  Prediction:  {prediction}")
    print(f"  Confidence:  {confidence}")
    print("=" * 60)
    
    return score, prediction


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m hyaline.predict <pdb_file>")
        sys.exit(1)
    predict(sys.argv[1])
