#!/usr/bin/env python3
"""
Hyaline Prediction Module
=========================

Standalone prediction functionality for GPCR activation state.
"""
import os
import sys
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

# GitHub Release URL for checkpoint auto-download
CHECKPOINT_URL = (
    "https://github.com/Varosync/Hyaline/releases/download/v2.0.0/hyaline.pt"
)
CHECKPOINT_CACHE_DIR = Path.home() / '.hyaline' / 'checkpoints'


def download_checkpoint(url: str = CHECKPOINT_URL) -> Optional[Path]:
    """Download checkpoint from GitHub Releases and cache locally."""
    import urllib.request
    import urllib.error

    cache_path = CHECKPOINT_CACHE_DIR / 'hyaline.pt'
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model checkpoint...")
    print(f"  From: {url}")
    print(f"  To:   {cache_path}")

    try:
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                mb = block_num * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  Progress: {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)

        urllib.request.urlretrieve(url, str(cache_path), reporthook=_progress)
        print()  # newline after progress
        print(f"  ✓ Download complete")
        return cache_path
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"\n  ✗ Download failed: {e}")
        # Clean up partial download
        if cache_path.exists():
            cache_path.unlink()
        return None


def find_checkpoint() -> Optional[Path]:
    """Search for checkpoint in standard locations, auto-downloading if needed."""
    # 1. Environment variable (highest priority)
    env_path = os.environ.get('HYALINE_CHECKPOINT')
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2. User home directory (~/.hyaline/checkpoints/)
    home_path = CHECKPOINT_CACHE_DIR / 'hyaline.pt'
    if home_path.exists():
        return home_path

    # 3. Relative to package (works for dev installs / cloned repo)
    pkg_path = Path(__file__).parent.parent / 'checkpoints' / 'hyaline.pt'
    if pkg_path.exists():
        return pkg_path

    # 4. Current working directory
    cwd_path = Path.cwd() / 'checkpoints' / 'hyaline.pt'
    if cwd_path.exists():
        return cwd_path

    # 5. Auto-download from GitHub Releases
    print("\nCheckpoint not found locally. Attempting download...")
    downloaded = download_checkpoint()
    if downloaded:
        return downloaded

    return None


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
    """Get ESM3 embeddings for a sequence.

    Tries multiple import paths for compatibility across ESM package versions.
    Raises RuntimeError if ESM3 is not available.
    """
    ESM3 = None
    ESMProtein = None

    # Try primary import path (esm >= 3.0)
    try:
        from esm.models.esm3 import ESM3 as _ESM3
        from esm.sdk.api import ESMProtein as _ESMProtein
        ESM3, ESMProtein = _ESM3, _ESMProtein
    except ImportError:
        pass

    # Try alternative import path
    if ESM3 is None:
        try:
            from esm3 import ESM3 as _ESM3
            from esm3.sdk.api import ESMProtein as _ESMProtein
            ESM3, ESMProtein = _ESM3, _ESMProtein
        except ImportError:
            pass

    if ESM3 is None:
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

    try:
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        model.eval()
        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():
            protein_tensor = model.encode(protein)
            tokens = protein_tensor.sequence.unsqueeze(0).to(device)
            embeddings = model.encoder.sequence_embed(tokens)
            embeddings = embeddings.squeeze(0).float()

            # Remove BOS/EOS tokens (match training pipeline)
            if embeddings.shape[0] > len(sequence):
                embeddings = embeddings[1:len(sequence)+1]

        return embeddings.cpu().numpy()
    except Exception as e:
        error_msg = str(e)
        if 'gated' in error_msg.lower() or '401' in error_msg or 'login' in error_msg.lower():
            raise RuntimeError(
                f"ESM3 model requires HuggingFace authentication.\n"
                f"  1. Accept the license at: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1\n"
                f"  2. Log in: huggingface-cli login\n"
                f"  Original error: {e}"
            ) from e
        raise RuntimeError(f"ESM3 embedding computation failed: {e}") from e


def get_random_embeddings(sequence: str) -> np.ndarray:
    """Generate random embeddings for testing only."""
    print("WARNING: Using random embeddings (for testing only)")
    print("         Results will NOT be meaningful.")
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


def _classify(score: float) -> Tuple[str, str]:
    """Derive prediction label and confidence from a raw score."""
    prediction = 'Active' if score > 0.5 else 'Inactive'
    if score > 0.90 or score < 0.10:
        confidence = "High"
    elif score > 0.75 or score < 0.25:
        confidence = "Medium"
    else:
        confidence = "Low"
    return prediction, confidence


def predict(
    pdb_path: str, 
    checkpoint_path: Optional[str] = None, 
    device: str = 'cuda',
    allow_random: bool = False,
    quiet: bool = False
) -> Tuple[Optional[float], Optional[str]]:
    """
    Predict GPCR activation state.
    
    Args:
        pdb_path: Path to PDB file
        checkpoint_path: Path to model checkpoint (optional)
        device: 'cuda' or 'cpu'
        allow_random: Allow random embeddings for testing
        quiet: Suppress per-file output (used in batch mode)
    
    Returns:
        score: Activation probability (0-1)
        prediction: 'Active' or 'Inactive'
    """
    from torch_geometric.data import Data
    
    log = (lambda *a, **kw: None) if quiet else print
    
    log("=" * 60)
    log("HYALINE PREDICTION")
    log("=" * 60)
    
    # Parse PDB
    log(f"\nInput: {pdb_path}")
    sequence, ca_coords, chain = parse_pdb(pdb_path)
    n_residues = len(ca_coords)
    log(f"Residues: {n_residues}")
    
    if n_residues < 50:
        log("Error: Structure too short for GPCR prediction")
        return None, None
    
    # Build edges
    edge_index, distances = build_radius_edges(ca_coords, cutoff=10.0)
    log(f"Edges: {edge_index.shape[1]}")
    
    # Get embeddings
    log("\nComputing ESM3 embeddings...")
    try:
        node_features = get_esm3_embeddings(sequence, device)
    except RuntimeError as e:
        if allow_random:
            node_features = get_random_embeddings(sequence)
        else:
            print(f"\nERROR: {e}")
            print("\nTo run with random embeddings (testing only), use --allow-random")
            sys.exit(1)
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
    log("\nLoading model...")
    model = HyalineV2(**V2D_CONFIG).to(device)
    
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
    else:
        ckpt_path = find_checkpoint()

    if ckpt_path and ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        log(f"Loaded: {ckpt_path}")
    elif allow_random:
        log("WARNING: No checkpoint found. Using untrained model.")
        log("         Results will NOT be meaningful.")
    else:
        print("ERROR: Model checkpoint not found.")
        print("Please provide the checkpoint in one of these locations:")
        print(f"  1. Set env var: HYALINE_CHECKPOINT=/path/to/hyaline.pt")
        print(f"  2. Place at:    ~/.hyaline/checkpoints/hyaline.pt")
        print(f"  3. Place at:    ./checkpoints/hyaline.pt")
        print(f"  4. Pass flag:   hyaline predict --checkpoint /path/to/hyaline.pt")
        sys.exit(1)
    
    model.eval()
    
    # Predict
    with torch.no_grad():
        logits, _ = model(data)
        score = torch.sigmoid(logits).item()
    
    prediction, confidence = _classify(score)
    
    # Results
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"  Score:       {score:.4f}")
    log(f"  Prediction:  {prediction}")
    log(f"  Confidence:  {confidence}")
    log("=" * 60)
    
    return score, prediction


def predict_batch(
    pdb_dir: str,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
    allow_random: bool = False,
    output_csv: Optional[str] = None
) -> list:
    """
    Run batch prediction on a directory of PDB files.

    Prints a live progress line per file, a ranked summary table at the end,
    and writes results to a CSV for downstream analysis.

    Returns:
        List of (filename, score, prediction, confidence) tuples sorted by
        score descending.
    """
    pdb_dir = Path(pdb_dir)
    pdb_files = sorted(pdb_dir.glob('*.pdb'))

    if not pdb_files:
        print(f"Error: No .pdb files found in {pdb_dir}")
        sys.exit(1)

    n = len(pdb_files)
    print("=" * 70)
    print("HYALINE BATCH PREDICTION")
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
                str(pdb_file), checkpoint_path, device, allow_random, quiet=True
            )
            if score is not None:
                _, confidence = _classify(score)
                results.append((name, score, prediction, confidence))
                print(f" {score:.4f}  {prediction}")
            else:
                failed.append((name, "Too short (<50 residues)"))
                print(" SKIPPED (too short)")
        except Exception as e:
            failed.append((name, str(e)))
            print(f" ERROR")

    # Sort by score descending for ranking
    results.sort(key=lambda r: r[1], reverse=True)

    n_active = sum(1 for r in results if r[2] == 'Active')
    n_inactive = len(results) - n_active

    # Summary table
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

    # Write CSV
    csv_path = Path(output_csv) if output_csv else pdb_dir / "hyaline_results.csv"
    with open(csv_path, 'w') as f:
        f.write("rank,file,score,prediction,confidence\n")
        for rank, (name, score, pred, conf) in enumerate(results, 1):
            f.write(f"{rank},{name},{score:.4f},{pred},{conf}\n")
    print(f"\n  Results saved to: {csv_path}")

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m hyaline.predict <pdb_file_or_directory>")
        sys.exit(1)
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: Path not found: {sys.argv[1]}")
        sys.exit(1)
    if input_path.is_dir():
        predict_batch(sys.argv[1])
    else:
        predict(sys.argv[1])
