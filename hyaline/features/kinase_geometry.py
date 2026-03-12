"""
Kinase Geometric Features
==========================

Conformational measurements derived from the KLIFS 85-residue pocket
alignment:

1. **DFG–αC-helix distance** – Euclidean distance between the DFG motif
   centroid (positions 80–82) and the αC-helix centroid (positions 20–30).
   Discriminates DFG-in (~10–14 Å) from DFG-out (~16–22 Å) conformations.

2. **Hinge–activation-loop angle** – angle formed by hinge centroid
   (positions 46–48), catalytic lysine (position 17), and activation loop
   centroid (positions 72–85).

All functions take Cα coordinates indexed by 1-based KLIFS pocket position
and handle missing residues gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KLIFS pocket position ranges (1-indexed → stored 0-indexed internally)
# ---------------------------------------------------------------------------

# αC-helix: KLIFS positions 20–30  (0-indexed: 19–29)
AC_HELIX_RANGE = (19, 30)

# Hinge region: positions 46–48  (0-indexed: 45–47)
HINGE_RANGE = (45, 48)

# DFG motif: positions 80–82  (0-indexed: 79–81)
DFG_RANGE = (79, 82)

# Activation loop: positions 72–85  (0-indexed: 71–84)
ACTIVATION_LOOP_RANGE = (71, 85)

# Catalytic lysine: position 17  (0-indexed: 16)
CATALYTIC_LYS_POS = 16

# G-rich loop: positions 1–9  (0-indexed: 0–8)
GRICH_RANGE = (0, 9)

# Gap characters
GAP_CHARS = {"_", "-", " ", ""}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class KinaseGeometry:
    """Kinase-specific geometric features from KLIFS pocket coordinates."""

    dfg_chelix_distance: float = 0.0     # Å
    hinge_activation_angle: float = 0.0  # degrees
    dfg_centroid: Optional[Tuple[float, float, float]] = None
    chelix_centroid: Optional[Tuple[float, float, float]] = None
    hinge_centroid: Optional[Tuple[float, float, float]] = None
    activation_centroid: Optional[Tuple[float, float, float]] = None
    grich_centroid: Optional[Tuple[float, float, float]] = None

    # Derived booleans
    is_dfg_out_predicted: bool = False   # distance > 15 Å heuristic

    # Quality
    n_dfg_resolved: int = 0
    n_chelix_resolved: int = 0
    n_hinge_resolved: int = 0
    n_activation_resolved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON serialisation
        for key in ("dfg_centroid", "chelix_centroid", "hinge_centroid",
                     "activation_centroid", "grich_centroid"):
            if d[key] is not None:
                d[key] = list(d[key])
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_coords(
    all_coords: np.ndarray,
    pocket_seq: str,
    start: int,
    end: int,
) -> np.ndarray:
    """Extract resolved Cα coords for a KLIFS position range.

    Parameters
    ----------
    all_coords : ndarray, shape (N, 3)
        Coordinates for resolved residues *only* (rows correspond to
        non-gap positions in ``pocket_seq`` in order).
    pocket_seq : str
        85-character KLIFS pocket string.
    start, end : int
        0-indexed half-open range [start, end) into the 85 positions.

    Returns
    -------
    ndarray, shape (M, 3)
        Coordinates for resolved residues within the range.
    """
    # Build a mapping: pocket-position → row index in all_coords
    resolved_idx = 0
    pos_to_row: Dict[int, int] = {}
    for pos, ch in enumerate(pocket_seq):
        if ch.upper() not in GAP_CHARS:
            pos_to_row[pos] = resolved_idx
            resolved_idx += 1

    rows = []
    for pos in range(start, min(end, len(pocket_seq))):
        if pos in pos_to_row:
            row = pos_to_row[pos]
            if row < all_coords.shape[0]:
                rows.append(row)

    if not rows:
        return np.empty((0, 3))

    return all_coords[rows]


def _centroid(coords: np.ndarray) -> Optional[np.ndarray]:
    if coords.shape[0] == 0:
        return None
    return coords.mean(axis=0)


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ∠ABC in degrees (B is the vertex)."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_dfg_chelix_distance(
    coords: np.ndarray,
    pocket_seq: str,
) -> Tuple[float, int, int]:
    """Compute distance between DFG centroid and αC-helix centroid.

    Returns (distance, n_dfg_resolved, n_chelix_resolved).
    """
    dfg_coords = _extract_coords(coords, pocket_seq, *DFG_RANGE)
    chelix_coords = _extract_coords(coords, pocket_seq, *AC_HELIX_RANGE)

    dfg_c = _centroid(dfg_coords)
    chelix_c = _centroid(chelix_coords)

    if dfg_c is None or chelix_c is None:
        return 0.0, dfg_coords.shape[0], chelix_coords.shape[0]

    dist = float(np.linalg.norm(dfg_c - chelix_c))
    return dist, dfg_coords.shape[0], chelix_coords.shape[0]


def compute_hinge_activation_angle(
    coords: np.ndarray,
    pocket_seq: str,
) -> Tuple[float, int, int]:
    """Compute hinge–catalytic-Lys–activation-loop angle.

    The angle vertex is the catalytic lysine (KLIFS position 17).

    Returns (angle_degrees, n_hinge_resolved, n_activation_resolved).
    """
    hinge_coords = _extract_coords(coords, pocket_seq, *HINGE_RANGE)
    actloop_coords = _extract_coords(coords, pocket_seq, *ACTIVATION_LOOP_RANGE)

    hinge_c = _centroid(hinge_coords)
    actloop_c = _centroid(actloop_coords)

    # Catalytic lysine
    lys_coords = _extract_coords(coords, pocket_seq, CATALYTIC_LYS_POS, CATALYTIC_LYS_POS + 1)
    lys_c = _centroid(lys_coords)

    if hinge_c is None or actloop_c is None or lys_c is None:
        return 0.0, hinge_coords.shape[0], actloop_coords.shape[0]

    angle = _angle_between(hinge_c, lys_c, actloop_c)
    return angle, hinge_coords.shape[0], actloop_coords.shape[0]


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def compute_kinase_geometry(
    coords: np.ndarray,
    pocket_seq: str,
) -> KinaseGeometry:
    """Compute all kinase geometric features.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Cα coordinates for the N resolved (non-gap) pocket residues.
    pocket_seq : str
        85-character KLIFS pocket alignment string.

    Returns
    -------
    KinaseGeometry
    """
    # DFG–αC distance
    dfg_dist, n_dfg, n_chelix = compute_dfg_chelix_distance(coords, pocket_seq)

    # Hinge–activation angle
    hinge_angle, n_hinge, n_actloop = compute_hinge_activation_angle(coords, pocket_seq)

    # Centroids for debugging / downstream
    dfg_coords = _extract_coords(coords, pocket_seq, *DFG_RANGE)
    chelix_coords = _extract_coords(coords, pocket_seq, *AC_HELIX_RANGE)
    hinge_coords = _extract_coords(coords, pocket_seq, *HINGE_RANGE)
    actloop_coords = _extract_coords(coords, pocket_seq, *ACTIVATION_LOOP_RANGE)
    grich_coords = _extract_coords(coords, pocket_seq, *GRICH_RANGE)

    def _to_tuple(c):
        if c is None:
            return None
        return tuple(float(x) for x in c)

    return KinaseGeometry(
        dfg_chelix_distance=dfg_dist,
        hinge_activation_angle=hinge_angle,
        dfg_centroid=_to_tuple(_centroid(dfg_coords)),
        chelix_centroid=_to_tuple(_centroid(chelix_coords)),
        hinge_centroid=_to_tuple(_centroid(hinge_coords)),
        activation_centroid=_to_tuple(_centroid(actloop_coords)),
        grich_centroid=_to_tuple(_centroid(grich_coords)),
        is_dfg_out_predicted=(dfg_dist > 15.0),
        n_dfg_resolved=n_dfg,
        n_chelix_resolved=n_chelix,
        n_hinge_resolved=n_hinge,
        n_activation_resolved=n_actloop,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic pocket with known positions
    pocket = "A" * 85  # all resolved
    np.random.seed(42)
    coords = np.random.randn(85, 3) * 10  # random positions

    geom = compute_kinase_geometry(coords, pocket)
    print("Kinase Geometry (synthetic):")
    for k, v in geom.to_dict().items():
        print(f"  {k}: {v}")
