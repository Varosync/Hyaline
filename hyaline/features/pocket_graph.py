"""
Pocket Graph Builder
====================

Converts KLIFS 85-residue pocket Cα coordinates into PyTorch Geometric
graphs for input to the Spiking EGNN.

Uses scipy KDTree for k-NN construction (no torch-cluster dependency).

Key design decisions:
- Gap residues (coords [0,0,0], sequence '-') are EXCLUDED from the graph
- k=10 default for k-NN edges (typical for protein pocket graphs)
- Node features: one-hot AA (22 classes) + normalized Cα coords (3 values) = 25-dim
- Edge attributes: Euclidean distance (1-dim)
- No self-loops
- Returns mask tensor for mapping back to 85-position KLIFS alignment
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-X"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
NUM_AA_CLASSES = len(AA_VOCAB)  # 22


def _knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    """Build k-NN edge index using scipy KDTree. No self-loops.

    Returns shape (2, E) int64 array.
    """
    tree = cKDTree(coords)
    # k+1 because query includes the point itself
    dists, indices = tree.query(coords, k=k + 1)

    N = coords.shape[0]
    rows = []
    cols = []
    for i in range(N):
        for j_idx in range(k + 1):
            j = indices[i, j_idx]
            if i != j:  # skip self-loop
                rows.append(i)
                cols.append(j)

    return np.array([rows, cols], dtype=np.int64)


def build_pocket_graph(
    pocket_sequence: str,
    pocket_coords: np.ndarray,
    k: int = 10,
    structure_id: Optional[int] = None,
) -> Tuple[Data, torch.Tensor]:
    """Build a k-NN graph from KLIFS pocket coordinates.

    Parameters
    ----------
    pocket_sequence : str
        85-character KLIFS pocket sequence. '-' indicates gap.
    pocket_coords : ndarray, shape (85, 3)
        Cα coordinates for each of the 85 KLIFS positions.
        Gap positions should have [0, 0, 0].
    k : int
        Number of nearest neighbors for graph construction. Default 10.
    structure_id : int, optional
        KLIFS structure ID for error messages.

    Returns
    -------
    data : torch_geometric.data.Data
        Graph with:
        - x: [N, 25] node features (22 one-hot AA + 3 normalized coords)
        - edge_index: [2, E] k-NN edges (no self-loops)
        - edge_attr: [E, 1] Euclidean distances
        - pos: [N, 3] raw Cα coordinates
        - num_nodes: N (number of resolved residues)
    mask : Tensor, shape (85,)
        Boolean mask: True for positions included in the graph.

    Raises
    ------
    ValueError
        If coordinates contain NaN or Inf values, or too few resolved residues.
    """
    pocket_sequence = pocket_sequence[:85].ljust(85, '-')
    pocket_coords = np.asarray(pocket_coords, dtype=np.float32)
    sid_str = f" for structure {structure_id}" if structure_id else ""

    if pocket_coords.shape != (85, 3):
        raise ValueError(f"Expected pocket_coords shape (85, 3), got {pocket_coords.shape}{sid_str}")

    # Validate for NaN/Inf
    bad_mask = ~np.isfinite(pocket_coords)
    if bad_mask.any():
        bad_residues = np.where(bad_mask.any(axis=1))[0]
        raise ValueError(f"Coordinates contain NaN or Inf at residue indices: {bad_residues.tolist()}{sid_str}")

    # Identify resolved (non-gap) residues
    mask = np.zeros(85, dtype=bool)
    for i in range(85):
        is_gap_char = pocket_sequence[i] == '-'
        is_zero_coord = np.allclose(pocket_coords[i], 0.0, atol=1e-6)
        mask[i] = not is_gap_char and not is_zero_coord

    resolved_indices = np.where(mask)[0]
    N = len(resolved_indices)

    if N < 3:
        raise ValueError(f"Too few resolved residues ({N}) to build a graph{sid_str}")

    effective_k = min(k, N - 1)

    # Extract resolved coordinates
    coords = pocket_coords[resolved_indices]  # [N, 3]

    # Build one-hot amino acid features
    aa_indices = []
    for idx in resolved_indices:
        aa = pocket_sequence[idx]
        aa_indices.append(AA_TO_IDX.get(aa, 21))
    aa_tensor = torch.tensor(aa_indices, dtype=torch.long)
    one_hot = torch.zeros(N, NUM_AA_CLASSES)
    one_hot.scatter_(1, aa_tensor.unsqueeze(1), 1.0)

    # Normalize coordinates (zero-mean, unit-std per dimension)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    coord_mean = coords_tensor.mean(dim=0, keepdim=True)
    coord_std = coords_tensor.std(dim=0, keepdim=True).clamp(min=1e-6)
    coords_normalized = (coords_tensor - coord_mean) / coord_std

    # Node features: [N, 25] = 22 one-hot + 3 normalized coords
    x = torch.cat([one_hot, coords_normalized], dim=1)

    # Build k-NN graph using scipy
    edge_index_np = _knn_edges(coords, k=effective_k)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # Compute edge distances
    row, col = edge_index
    edge_vec = coords_tensor[row] - coords_tensor[col]
    edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)  # [E, 1]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_dist,
        pos=coords_tensor,
        num_nodes=N,
    )

    # Metadata for reconstruction
    data.klifs_mask = torch.tensor(mask, dtype=torch.bool)
    data.resolved_indices = torch.tensor(resolved_indices, dtype=torch.long)
    data.coord_mean = coord_mean.squeeze(0)
    data.coord_std = coord_std.squeeze(0)
    if structure_id is not None:
        data.structure_id = structure_id

    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    return data, mask_tensor


def batch_build_pocket_graphs(
    sequences: list[str],
    coords_array: np.ndarray,
    k: int = 10,
    structure_ids: Optional[list[int]] = None,
) -> Tuple[list[Data], torch.Tensor]:
    """Build pocket graphs for a batch of structures.

    Parameters
    ----------
    sequences : list of str
        List of 85-char pocket sequences.
    coords_array : ndarray, shape (B, 85, 3)
        Batch of Cα coordinates.
    k : int
        k-NN parameter.
    structure_ids : list of int, optional
        Structure IDs for error reporting.

    Returns
    -------
    graphs : list of Data
        One PyG graph per structure.
    masks : Tensor, shape (B, 85)
        Batch of boolean masks.
    """
    B = len(sequences)
    graphs = []
    masks = []
    skipped = 0

    for i in range(B):
        sid = structure_ids[i] if structure_ids else None
        try:
            data, mask = build_pocket_graph(sequences[i], coords_array[i], k=k, structure_id=sid)
            graphs.append(data)
            masks.append(mask)
        except ValueError as e:
            logger.warning(f"Skipping structure {sid}: {e}")
            skipped += 1

    if skipped > 0:
        logger.info(f"Built {len(graphs)} graphs, skipped {skipped}")

    masks_tensor = torch.stack(masks) if masks else torch.zeros(0, 85, dtype=torch.bool)
    return graphs, masks_tensor
