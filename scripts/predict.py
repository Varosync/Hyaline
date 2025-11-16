#!/usr/bin/env python3
"""
Hyaline Prediction Script
=========================

Predict GPCR activation state from a PDB structure using the
HyalineV2-D production model.

Usage:
    python scripts/predict.py structure.pdb
    python scripts/predict.py structure.pdb --checkpoint path/to/model.pt
"""
import torch
import numpy as np
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyaline import HyalineV2
from hyaline.graph_data import build_radius_edges

# Amino acid mapping
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M', 'SEC': 'C', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H'
}

# Default checkpoint path
DEFAULT_CHECKPOINT = 'checkpoints/hyaline.pt'

# V2-D Production model configuration
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


def detect_receptor_chain(pdb_path: str) -> str:
    """Detect the receptor chain in a multi-chain PDB."""
    chains = {}
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                res_name = line[17:20].strip()
                res_num = line[22:27].strip()
                
                if res_name not in AA_MAP:
                    continue
                
                if chain not in chains:
                    chains[chain] = set()
                chains[chain].add(f"{chain}_{res_num}")
    
    if not chains:
        return 'A'
    
    # Antibody chain names to avoid
    antibody_chains = {'H', 'L', 'M', 'N', 'G', 'B'}
    
    # Priority 1: Chain R (receptor convention)
    if 'R' in chains and len(chains['R']) > 200:
        return 'R'
    
    # Priority 2: Chain A (most common)
    if 'A' in chains and len(chains['A']) > 200:
        return 'A'
    
    # Priority 3: Longest non-antibody chain
    receptor_candidates = [(c, len(r)) for c, r in chains.items() 
                           if len(r) > 200 and c not in antibody_chains]
    if receptor_candidates:
        return max(receptor_candidates, key=lambda x: x[1])[0]
    
    # Fallback: longest chain
    return max(chains.items(), key=lambda x: len(x[1]))[0]


def parse_pdb(pdb_path: str):
    """Parse PDB for sequence and C-alpha coordinates."""
    target_chain = detect_receptor_chain(pdb_path)
    
    residues = []
    ca_coords = []
    seen_res = set()
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                if chain != target_chain:
                    continue
                    
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_num = line[22:27].strip()
                
                res_key = f"{chain}_{res_num}"
                
                if res_key not in seen_res:
                    seen_res.add(res_key)
                    if res_name in AA_MAP:
                        residues.append(AA_MAP[res_name])
                    else:
                        residues.append('X')
                
                if atom_name == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
    
    sequence = ''.join(residues)
    ca_coords = np.array(ca_coords, dtype=np.float32)
    
    return sequence, ca_coords, target_chain


def get_esm3_embeddings(sequence: str, device='cuda'):
    """Get ESM3 embeddings for sequence."""
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    
    model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    model.eval()
    
    protein = ESMProtein(sequence=sequence)
    
    with torch.no_grad():
        protein_tensor = model.encode(protein)
        tokens = protein_tensor.sequence.unsqueeze(0).to(device)
        embeddings = model.encoder.sequence_embed(tokens)
        embeddings = embeddings.squeeze(0).float()
        
        # Remove BOS/EOS tokens
        if embeddings.shape[0] > len(sequence):
            embeddings = embeddings[1:len(sequence)+1]
    
    return embeddings.cpu().numpy()


def predict(pdb_path: str, checkpoint_path: str = None, device: str = 'cuda'):
    """
    Predict GPCR activation state using HyalineV2-D.
    
    Args:
        pdb_path: Path to PDB file
        checkpoint_path: Path to model checkpoint (default: checkpoints/hyaline.pt)
        device: Device to run inference on
    
    Returns:
        score: Activation probability (0-1)
        prediction: 'Active' or 'Inactive'
    """
    print("=" * 60)
    print("HYALINE V2-D PREDICTION")
    print("Geometric Deep Learning for GPCR Activation State")
    print("=" * 60)
    
    # Parse PDB
    print(f"\nInput: {pdb_path}")
    sequence, ca_coords, chain = parse_pdb(pdb_path)
    n_residues = len(ca_coords)
    print(f"Chain: {chain}")
    print(f"Residues: {n_residues}")
    
    if n_residues < 100:
        print("WARNING: Very short sequence, may not be a full GPCR")
    
    # Build graph edges
    edge_index, distances = build_radius_edges(ca_coords, cutoff=10.0)
    n_edges = edge_index.shape[1]
    print(f"Graph edges (10Å cutoff): {n_edges}")
    
    # Get ESM3 embeddings
    print("\nComputing ESM3 embeddings...")
    node_features = get_esm3_embeddings(sequence, device)
    node_features = node_features[:n_residues]
    print(f"Embedding shape: {node_features.shape}")
    
    # Compute edge features
    dist_sq = (distances ** 2).astype(np.float32) / 100.0
    edge_features = np.stack([
        dist_sq,
        distances / 10.0,
        np.ones_like(distances)
    ], axis=-1)
    
    # Create PyG Data object
    from torch_geometric.data import Data
    
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        pos=torch.tensor(ca_coords, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        batch=torch.zeros(n_residues, dtype=torch.long)
    ).to(device)
    
    # Load model
    print("\nLoading HyalineV2-D model...")
    model = HyalineV2(**V2D_CONFIG).to(device)
    
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded: {checkpoint_path}")
    else:
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None, None
    
    model.eval()
    
    # Predict
    with torch.no_grad():
        logits, attention = model(data)
        score = torch.sigmoid(logits).item()
    
    # Interpret score
    prediction = 'Active' if score > 0.5 else 'Inactive'
    
    if score > 0.90:
        confidence = "High"
        interpretation = "Strong active-state geometric signature"
    elif score > 0.75:
        confidence = "Medium-High"
        interpretation = "Likely active; review structural features"
    elif score > 0.50:
        confidence = "Medium"
        interpretation = "Probable active state"
    elif score > 0.25:
        confidence = "Medium"
        interpretation = "Probable inactive state"
    elif score > 0.10:
        confidence = "Medium-High"
        interpretation = "Likely inactive; check for partial activation"
    else:
        confidence = "High"
        interpretation = "Strong inactive-state geometric profile"
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"  Score:          {score:.4f}")
    print(f"  Prediction:     {prediction}")
    print(f"  Confidence:     {confidence}")
    print(f"  Interpretation: {interpretation}")
    print("=" * 60)
    
    return score, prediction


def main():
    parser = argparse.ArgumentParser(
        description='Hyaline V2-D: GPCR Activation State Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/predict.py 7D7M.pdb
    python scripts/predict.py structure.pdb --checkpoint checkpoints/my_model.pt
    python scripts/predict.py structure.pdb --device cpu

Score Interpretation:
    > 0.90  High-confidence Active
    > 0.50  Likely Active
    < 0.50  Likely Inactive
    < 0.10  High-confidence Inactive
        """
    )
    parser.add_argument('pdb_file', help='Path to PDB structure file')
    parser.add_argument('--checkpoint', default=None, 
                        help=f'Model checkpoint path (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not Path(args.pdb_file).exists():
        print(f"ERROR: PDB file not found: {args.pdb_file}")
        sys.exit(1)
    
    predict(args.pdb_file, args.checkpoint, args.device)


if __name__ == '__main__':
    main()
