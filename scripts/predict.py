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
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein
    except ImportError:
        import sys
        if sys.version_info < (3, 10):
            version_msg = f"\n\nNOTE: ESM3 requires Python >= 3.10. You are using Python {sys.version_info.major}.{sys.version_info.minor}."
        else:
            version_msg = ""
        raise RuntimeError(
            "ESM3 package not found. Install it with:\n"
            "  pip install 'esm>=3.0.0'\n"
            f"Or: pip install 'hyaline[esm]'{version_msg}"
        )
    
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


def _classify(score):
    """Derive prediction label, confidence, and interpretation from score."""
    prediction = 'Active' if score > 0.5 else 'Inactive'
    if score > 0.90:
        confidence, interpretation = "High", "Strong active-state geometric signature"
    elif score > 0.75:
        confidence, interpretation = "Medium-High", "Likely active; review structural features"
    elif score > 0.50:
        confidence, interpretation = "Medium", "Probable active state"
    elif score > 0.25:
        confidence, interpretation = "Medium", "Probable inactive state"
    elif score > 0.10:
        confidence, interpretation = "Medium-High", "Likely inactive; check for partial activation"
    else:
        confidence, interpretation = "High", "Strong inactive-state geometric profile"
    return prediction, confidence, interpretation


def predict(pdb_path: str, checkpoint_path: str = None, device: str = 'cuda',
            quiet: bool = False):
    """
    Predict GPCR activation state using HyalineV2-D.
    
    Args:
        pdb_path: Path to PDB file
        checkpoint_path: Path to model checkpoint (default: checkpoints/hyaline.pt)
        device: Device to run inference on
        quiet: Suppress per-file output (used in batch mode)
    
    Returns:
        score: Activation probability (0-1)
        prediction: 'Active' or 'Inactive'
    """
    log = (lambda *a, **kw: None) if quiet else print

    log("=" * 60)
    log("HYALINE V2-D PREDICTION")
    log("Geometric Deep Learning for GPCR Activation State")
    log("=" * 60)
    
    # Parse PDB
    log(f"\nInput: {pdb_path}")
    sequence, ca_coords, chain = parse_pdb(pdb_path)
    n_residues = len(ca_coords)
    log(f"Chain: {chain}")
    log(f"Residues: {n_residues}")
    
    if n_residues < 100:
        log("WARNING: Very short sequence, may not be a full GPCR")
    
    # Build graph edges
    edge_index, distances = build_radius_edges(ca_coords, cutoff=10.0)
    n_edges = edge_index.shape[1]
    log(f"Graph edges (10Å cutoff): {n_edges}")
    
    # Get ESM3 embeddings
    log("\nComputing ESM3 embeddings...")
    node_features = get_esm3_embeddings(sequence, device)
    node_features = node_features[:n_residues]
    log(f"Embedding shape: {node_features.shape}")
    
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
    log("\nLoading HyalineV2-D model...")
    model = HyalineV2(**V2D_CONFIG).to(device)
    
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        log(f"Loaded: {checkpoint_path}")
    else:
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None, None
    
    model.eval()
    
    # Predict
    with torch.no_grad():
        logits, attention = model(data)
        score = torch.sigmoid(logits).item()
    
    prediction, confidence, interpretation = _classify(score)
    
    # Print results
    log("\n" + "=" * 60)
    log("PREDICTION RESULTS")
    log("=" * 60)
    log(f"  Score:          {score:.4f}")
    log(f"  Prediction:     {prediction}")
    log(f"  Confidence:     {confidence}")
    log(f"  Interpretation: {interpretation}")
    log("=" * 60)
    
    return score, prediction


def predict_batch(pdb_dir, checkpoint_path=None, device='cuda', output_csv=None):
    """Run batch prediction on a directory of PDB files with summary output."""
    pdb_dir = Path(pdb_dir)
    pdb_files = sorted(pdb_dir.glob('*.pdb'))

    if not pdb_files:
        print(f"ERROR: No .pdb files found in {pdb_dir}")
        sys.exit(1)

    n = len(pdb_files)
    print("=" * 70)
    print("HYALINE V2-D BATCH PREDICTION")
    print(f"Directory:  {pdb_dir}")
    print(f"PDB files:  {n}")
    print("=" * 70)

    results = []
    failed = []

    for i, pdb_file in enumerate(pdb_files, 1):
        name = pdb_file.name
        print(f"  [{i:>{len(str(n))}}/{n}] {name:<40s}", end="", flush=True)
        try:
            score, prediction = predict(
                str(pdb_file), checkpoint_path, device, quiet=True
            )
            if score is not None:
                _, confidence, _ = _classify(score)
                results.append((name, score, prediction, confidence))
                print(f" {score:.4f}  {prediction}")
            else:
                failed.append((name, "Too short"))
                print(" SKIPPED (too short)")
        except Exception as e:
            failed.append((name, str(e)))
            print(" ERROR")

    results.sort(key=lambda r: r[1], reverse=True)
    n_active = sum(1 for r in results if r[2] == 'Active')
    n_inactive = len(results) - n_active

    print("\n" + "=" * 70)
    print("RESULTS  (ranked by activation score)")
    print("=" * 70)
    print(f"  {'Rank':<6}{'File':<40}{'Score':>7}  {'State':<10}{'Confidence'}")
    print(f"  {'-'*5} {'-'*39} {'-'*7}  {'-'*9} {'-'*10}")
    for rank, (name, score, pred, conf) in enumerate(results, 1):
        print(f"  {rank:<6}{name:<40}{score:>7.4f}  {pred:<10}{conf}")

    if failed:
        print(f"\n  Skipped / errors ({len(failed)}):")
        for name, reason in failed:
            print(f"    {name}: {reason}")

    print(f"\n  Active:   {n_active:>4} / {len(results)}")
    print(f"  Inactive: {n_inactive:>4} / {len(results)}")
    if failed:
        print(f"  Failed:   {len(failed):>4}")
    print("=" * 70)

    csv_path = Path(output_csv) if output_csv else pdb_dir / "hyaline_results.csv"
    with open(csv_path, 'w') as f:
        f.write("rank,file,score,prediction,confidence\n")
        for rank, (name, score, pred, conf) in enumerate(results, 1):
            f.write(f"{rank},{name},{score:.4f},{pred},{conf}\n")
    print(f"\n  Results saved to: {csv_path}")

    return results


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
    parser.add_argument('pdb_file', help='Path to PDB structure file or directory of PDB files')
    parser.add_argument('--checkpoint', default=None, 
                        help=f'Model checkpoint path (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path for batch results (default: <input_dir>/hyaline_results.csv)')
    
    args = parser.parse_args()
    input_path = Path(args.pdb_file)
    
    if not input_path.exists():
        print(f"ERROR: Path not found: {args.pdb_file}")
        sys.exit(1)
    
    if input_path.is_dir():
        predict_batch(str(input_path), args.checkpoint, args.device, args.output)
    else:
        predict(str(input_path), args.checkpoint, args.device)


if __name__ == '__main__':
    main()
