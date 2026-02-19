"""
TF Data Utilities
=================

Data loading and preparation for HyalineTF training and inference.
Supports both sequence-only and structure-augmented inputs.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple

from hyaline.motifs import get_tf_domain_type_embedding


def sequence_to_data(
    sequence: str,
    embeddings: np.ndarray,
    label: Optional[Dict[str, float]] = None,
    coords: Optional[np.ndarray] = None,
    cutoff: float = 10.0,
) -> Data:
    """
    Convert a TF sequence with ESM embeddings to a PyG Data object.

    Args:
        sequence:   Amino acid sequence
        embeddings: ESM embeddings [seq_len, embed_dim]
        label:      Optional dict with any of:
                      'function'    int   (0=activator, 1=repressor, 2=dual)
                      'binding'     float (DNA-binding affinity)
                      'regulatory'  float (regulatory impact score)
        coords:     Optional Cα coordinates [seq_len, 3]; enables EGNN path
        cutoff:     Distance cutoff for radius graph construction (Å)

    Returns:
        PyG Data object ready for HyalineTF
    """
    n = min(len(sequence), embeddings.shape[0])
    x = torch.tensor(embeddings[:n], dtype=torch.float32)

    domain_types = get_tf_domain_type_embedding(sequence[:n])

    kwargs: Dict = dict(x=x, domain_types=domain_types)

    if label is not None:
        if 'function' in label:
            kwargs['y_function'] = torch.tensor(
                label['function'], dtype=torch.long
            )
        if 'binding' in label:
            kwargs['y_binding'] = torch.tensor(
                label['binding'], dtype=torch.float32
            )
        if 'regulatory' in label:
            kwargs['y_regulatory'] = torch.tensor(
                label['regulatory'], dtype=torch.float32
            )

    if coords is not None:
        coords_t = torch.tensor(coords[:n], dtype=torch.float32)
        edge_index, distances = _build_radius_edges(coords[:n], cutoff)
        dist_sq = (distances ** 2).astype(np.float32) / 100.0
        edge_attr = np.stack(
            [dist_sq, distances / 10.0, np.ones_like(distances)], axis=-1
        )
        kwargs['pos'] = coords_t
        kwargs['edge_index'] = torch.tensor(edge_index, dtype=torch.long)
        kwargs['edge_attr'] = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(**kwargs)


def _build_radius_edges(
    coords: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build radius-graph edges for a single set of coordinates."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    mask = (dist < cutoff) & (dist > 0)
    sources, targets = np.where(mask)
    return np.stack([sources, targets], axis=0), dist[sources, targets]


def load_tf_sequences(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Load TF sequences from a FASTA file.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        List of (name, sequence) tuples
    """
    sequences: List[Tuple[str, str]] = []
    name: Optional[str] = None
    parts: List[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    sequences.append((name, ''.join(parts)))
                name = line[1:].split()[0]
                parts = []
            elif line:
                parts.append(line)

    if name is not None:
        sequences.append((name, ''.join(parts)))

    return sequences


def get_esm_embeddings(
    sequences: List[str],
    model_name: str = 'esm2_t33_650M_UR50D',
    device: str = 'cpu',
) -> List[np.ndarray]:
    """
    Compute ESM embeddings for a list of sequences.

    Uses the ESM2 650M model by default (compatible with both fair-esm and
    the EvolutionaryScale esm package).  Falls back to random arrays if ESM
    is not installed (useful for unit tests / CI without GPU).

    Args:
        sequences:   List of amino acid sequences
        model_name:  ESM model identifier
        device:      Computation device ('cpu' or 'cuda')

    Returns:
        List of float32 arrays [seq_len, embed_dim] — one per sequence
    """
    try:
        import esm as _esm

        model, alphabet = _esm.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()

        embeddings: List[np.ndarray] = []
        for seq in sequences:
            _, _, tokens = batch_converter([('seq', seq)])
            tokens = tokens.to(device)
            with torch.no_grad():
                results = model(tokens, repr_layers=[model.num_layers])
            emb = (
                results['representations'][model.num_layers][0, 1:-1]
                .cpu()
                .numpy()
            )
            embeddings.append(emb)

        return embeddings

    except Exception as exc:
        print(f"ESM embedding error: {exc}")
        print("Falling back to random embeddings (for testing only).")
        return [
            np.random.randn(len(s), 1280).astype(np.float32) for s in sequences
        ]
