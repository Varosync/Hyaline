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

3. **Glu–Lys salt bridge distance** – Distance between the conserved
   αC-Glutamate (KLIFS position 24) and β3-Lysine (position 17). A tight
   distance (<4 Å Cα–Cα) confirms αC-in; broken (>8 Å) indicates αC-out.

4. **Asp–Mg²⁺ coordination proxy** – Distance between DFG-Asp (position 80)
   and catalytic Lys (position 17) as a proxy for Asp–Mg²⁺ engagement.
   Tight coordination (<6 Å) indicates a catalytically competent DFG-in.

5. **Activation loop B-factors** – Average crystallographic temperature
   factors for the activation loop residues. High B-factors (>60 Å²) indicate
   the loop coordinates are unreliable ("the crystallographer's camera was
   blurry").

6. **R-spine integrity** – Distances between the four Regulatory Spine
   residues (RS1–RS4) that physically couple the DFG to the αC-helix. An
   intact R-spine has tight inter-residue distances; a broken spine indicates
   the structure is in an inactive conformation.

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

# Catalytic lysine / β3-Lys: position 17  (0-indexed: 16)
CATALYTIC_LYS_POS = 16

# Conserved αC-Glutamate: position 24  (0-indexed: 23)
# Forms a salt bridge with β3-Lys when αC-helix is "in"
AC_GLU_POS = 23

# DFG-Aspartate: KLIFS position 81 (0-indexed: 80) — the "D" of DFG
# Note: KLIFS pos 80 is the residue *before* the DFG (xDFG)
DFG_ASP_POS = 80

# G-rich loop: positions 1–9  (0-indexed: 0–8)
GRICH_RANGE = (0, 9)

# --------------------------------------------------------------------------
# R-spine (Regulatory Spine) positions — KLIFS 1-indexed → 0-indexed
# The four residues that physically couple the DFG motif to the αC-helix:
#   RS1 (HRD-His):      KLIFS position 68  → index 67
#   RS2 (DFG-Phe):      KLIFS position 81  → index 80
#   RS3 (αC-helix Xxx): KLIFS position 28  → index 27
#   RS4 (C-lobe Xxx):   KLIFS position 38  → index 37
# --------------------------------------------------------------------------
RSPINE_POSITIONS = {
    'RS1_HRD_His': 67,   # KLIFS position 68
    'RS2_DFG_Phe': 81,   # KLIFS position 82 (the "F" of DFG)
    'RS3_aC_helix': 27,  # KLIFS position 28
    'RS4_C_lobe': 37,    # KLIFS position 38
}
RSPINE_ORDER = ['RS1_HRD_His', 'RS2_DFG_Phe', 'RS3_aC_helix', 'RS4_C_lobe']

# Gap characters
GAP_CHARS = {"_", "-", " ", ""}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class KinaseGeometry:
    """Kinase-specific geometric features from KLIFS pocket coordinates."""

    # --- Original metrics ---
    dfg_chelix_distance: float = 0.0     # Å
    hinge_activation_angle: float = 0.0  # degrees

    # --- NEW: Salt bridge (αC-Glu ↔ β3-Lys) ---
    glu_lys_distance: float = 0.0        # Å  (Cα–Cα)
    salt_bridge_intact: bool = False      # True if < 13.0 Å (Cα–Cα)

    # --- NEW: DFG-Asp ↔ catalytic Lys proxy for Mg²⁺ coordination ---
    asp_lys_distance: float = 0.0        # Å  (DFG-Asp Cα ↔ β3-Lys Cα)
    dfg_asp_engaged: bool = False         # True if < 14.0 Å (Cα–Cα)

    # --- NEW: Activation loop B-factors ---
    activation_loop_bfactor_mean: float = 0.0   # Å²
    activation_loop_bfactor_std: float = 0.0    # Å²
    bfactors_reliable: bool = True              # True if mean < 60 Å²

    # --- NEW: R-spine integrity ---
    rspine_distances: Optional[Dict[str, float]] = None  # pairwise Cα distances
    rspine_mean_distance: float = 0.0    # Å  (mean of 3 consecutive pairs)
    rspine_intact: bool = False          # True if mean < 9.0 Å

    # --- Centroids ---
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

    # --- NEW: Confidence tier ---
    confidence_tier: int = 0             # 1 = High confidence, 2 = Ambiguous

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
# NEW: Salt Bridge (Glu-Lys)
# ---------------------------------------------------------------------------

def compute_glu_lys_salt_bridge(
    coords: np.ndarray,
    pocket_seq: str,
) -> Tuple[float, bool]:
    """Compute αC-Glu ↔ β3-Lys Cα–Cα distance (salt bridge metric).

    The Cα–Cα distance between these KLIFS positions is typically
    ~10–12 Å when the salt bridge is formed (αC-in) and >14 Å when
    αC is out.  Side-chain Oε–Nζ would be ~2.8 Å for a true salt
    bridge, but Cα–Cα is robustly available from KLIFS pocket coords.

    Returns (distance_angstrom, is_intact).

    Threshold calibration (from ABL1 crystal structures):
      αC-in:  Cα–Cα ≈ 10–12 Å → intact
      αC-out: Cα–Cα ≈ 14–18 Å → broken
    """
    glu_coords = _extract_coords(coords, pocket_seq, AC_GLU_POS, AC_GLU_POS + 1)
    lys_coords = _extract_coords(coords, pocket_seq, CATALYTIC_LYS_POS, CATALYTIC_LYS_POS + 1)

    if glu_coords.shape[0] == 0 or lys_coords.shape[0] == 0:
        return 0.0, False

    dist = float(np.linalg.norm(glu_coords[0] - lys_coords[0]))
    return dist, dist < 13.0  # Cα–Cα threshold for αC-in


# ---------------------------------------------------------------------------
# NEW: DFG-Asp ↔ Catalytic Lys (Mg²⁺ coordination proxy)
# ---------------------------------------------------------------------------

def compute_asp_coordination(
    coords: np.ndarray,
    pocket_seq: str,
) -> Tuple[float, bool]:
    """Compute DFG-Asp ↔ β3-Lys Cα–Cα distance.

    This serves as a proxy for Asp–Mg²⁺ coordination.  When the DFG is "in"
    and catalytically competent, the Asp coordinates Mg²⁺ which bridges to
    the Lys.  Cα–Cα < ~14 Å indicates the DFG-Asp is structurally engaged.

    Threshold calibration:
      DFG-in:  Cα–Cα ≈ 10–13 Å → engaged
      DFG-out: Cα–Cα ≈ 15–22 Å → disengaged

    Returns (distance_angstrom, is_engaged).
    """
    asp_coords = _extract_coords(coords, pocket_seq, DFG_ASP_POS, DFG_ASP_POS + 1)
    lys_coords = _extract_coords(coords, pocket_seq, CATALYTIC_LYS_POS, CATALYTIC_LYS_POS + 1)

    if asp_coords.shape[0] == 0 or lys_coords.shape[0] == 0:
        return 0.0, False

    dist = float(np.linalg.norm(asp_coords[0] - lys_coords[0]))
    return dist, dist < 14.0


# ---------------------------------------------------------------------------
# NEW: Activation Loop B-factors
# ---------------------------------------------------------------------------

def compute_activation_loop_bfactors(
    bfactors: np.ndarray,
    pocket_seq: str,
) -> Tuple[float, float, bool]:
    """Compute mean/std B-factors for activation loop residues.

    Parameters
    ----------
    bfactors : ndarray, shape (N,)
        Per-residue B-factors (temperature factors) for the N resolved
        (non-gap) residues, in the same order as coordinates.
    pocket_seq : str
        85-character KLIFS pocket string.

    Returns
    -------
    (mean_bfactor, std_bfactor, is_reliable)
        is_reliable is True if mean B-factor < 60 Å².
    """
    # Reuse _extract_coords logic to get the row indices
    resolved_idx = 0
    pos_to_row: Dict[int, int] = {}
    for pos, ch in enumerate(pocket_seq):
        if ch.upper() not in GAP_CHARS:
            pos_to_row[pos] = resolved_idx
            resolved_idx += 1

    rows = []
    start, end = ACTIVATION_LOOP_RANGE
    for pos in range(start, min(end, len(pocket_seq))):
        if pos in pos_to_row:
            row = pos_to_row[pos]
            if row < bfactors.shape[0]:
                rows.append(row)

    if not rows:
        return 0.0, 0.0, False

    loop_bfactors = bfactors[rows]
    mean_b = float(np.mean(loop_bfactors))
    std_b = float(np.std(loop_bfactors))
    return mean_b, std_b, mean_b < 60.0


# ---------------------------------------------------------------------------
# NEW: R-Spine Integrity
# ---------------------------------------------------------------------------

def compute_rspine_integrity(
    coords: np.ndarray,
    pocket_seq: str,
) -> Tuple[Dict[str, float], float, bool]:
    """Compute pairwise Cα distances between the four R-spine residues.

    The Regulatory Spine consists of four stacked hydrophobic residues:
      RS1 (HRD-His, pos 68) → RS2 (DFG-Phe, pos 81) → RS3 (αC, pos 28) → RS4 (C-lobe, pos 38)

    An intact spine has tight packing (~6–9 Å Cα–Cα between consecutive
    residues).  When the DFG flips, RS2 moves away, breaking the spine.

    Returns
    -------
    (pairwise_distances, mean_distance, is_intact)
        pairwise_distances: dict with keys like "RS1-RS2", etc.
        is_intact: True if mean consecutive distance < 9 Å.
    """
    spine_coords = {}
    for name, pos in RSPINE_POSITIONS.items():
        c = _extract_coords(coords, pocket_seq, pos, pos + 1)
        if c.shape[0] > 0:
            spine_coords[name] = c[0]

    # Compute consecutive pairwise distances
    pairwise: Dict[str, float] = {}
    dists: List[float] = []
    for i in range(len(RSPINE_ORDER) - 1):
        name_a = RSPINE_ORDER[i]
        name_b = RSPINE_ORDER[i + 1]
        key = f"{name_a.split('_')[0]}-{name_b.split('_')[0]}"
        if name_a in spine_coords and name_b in spine_coords:
            d = float(np.linalg.norm(spine_coords[name_a] - spine_coords[name_b]))
            pairwise[key] = d
            dists.append(d)

    if not dists:
        return pairwise, 0.0, False

    mean_d = float(np.mean(dists))
    return pairwise, mean_d, mean_d < 9.0


def get_rspine_coordinates(
    coords: np.ndarray,
    pocket_seq: str,
) -> np.ndarray:
    """Extract the 4 R-spine Cα coordinates for GNN featurisation.

    Returns ndarray of shape (4, 3).  Missing residues get coords [0, 0, 0].
    Order: RS1 (HRD-His), RS2 (DFG-Phe), RS3 (αC), RS4 (C-lobe).
    """
    result = np.zeros((4, 3), dtype=np.float32)
    for i, name in enumerate(RSPINE_ORDER):
        pos = RSPINE_POSITIONS[name]
        c = _extract_coords(coords, pocket_seq, pos, pos + 1)
        if c.shape[0] > 0:
            result[i] = c[0]
    return result


# ---------------------------------------------------------------------------
# NEW: Confidence Tiering
# ---------------------------------------------------------------------------

def assign_confidence_tier(geom: 'KinaseGeometry') -> int:
    """Assign a confidence tier based on structural integrity metrics.

    Tier 1 (High Confidence) — Training Set:
      - Salt bridge intact (Glu-Lys < 7 Å)
      - DFG-Asp engaged (< 8 Å) OR is_dfg_out_predicted (DFG-out doesn't
        need Asp coordination)
      - B-factors reliable (mean < 60 Å²)
      - R-spine assessment consistent with conformation

    Tier 2 (Ambiguous) — Hard Test Set:
      Everything else: broken geometry, high B-factors, or missing data.
    """
    # Require B-factors to be reliable
    if not geom.bfactors_reliable and geom.activation_loop_bfactor_mean > 0:
        return 2

    # For DFG-in structures, require salt bridge and Asp coordination
    if not geom.is_dfg_out_predicted:
        if geom.glu_lys_distance > 0 and not geom.salt_bridge_intact:
            return 2
        if geom.asp_lys_distance > 0 and not geom.dfg_asp_engaged:
            return 2

    # For DFG-out structures, require geometry consistent with DFG flip
    if geom.is_dfg_out_predicted:
        # DFG-out should have large DFG-αC distance and broken Asp engagement
        if geom.asp_lys_distance > 0 and geom.dfg_asp_engaged:
            # DFG is supposedly out but Asp is still engaged — suspicious
            return 2

    # If we have R-spine data and it looks broken in a supposedly active struct
    if (geom.rspine_mean_distance > 0 and not geom.rspine_intact
            and not geom.is_dfg_out_predicted):
        return 2

    return 1


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def compute_kinase_geometry(
    coords: np.ndarray,
    pocket_seq: str,
    bfactors: Optional[np.ndarray] = None,
) -> KinaseGeometry:
    """Compute all kinase geometric features.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Cα coordinates for the N resolved (non-gap) pocket residues.
    pocket_seq : str
        85-character KLIFS pocket alignment string.
    bfactors : ndarray, shape (N,), optional
        Per-residue B-factors for the N resolved residues.  If not provided,
        B-factor metrics will be zeroed and ``bfactors_reliable`` defaults
        to True (optimistic).

    Returns
    -------
    KinaseGeometry
    """
    # DFG–αC distance
    dfg_dist, n_dfg, n_chelix = compute_dfg_chelix_distance(coords, pocket_seq)

    # Hinge–activation angle
    hinge_angle, n_hinge, n_actloop = compute_hinge_activation_angle(coords, pocket_seq)

    # Salt bridge
    glu_lys_dist, sb_intact = compute_glu_lys_salt_bridge(coords, pocket_seq)

    # Asp coordination
    asp_lys_dist, asp_engaged = compute_asp_coordination(coords, pocket_seq)

    # B-factors
    if bfactors is not None and bfactors.shape[0] > 0:
        bf_mean, bf_std, bf_reliable = compute_activation_loop_bfactors(
            bfactors, pocket_seq)
    else:
        bf_mean, bf_std, bf_reliable = 0.0, 0.0, True

    # R-spine
    rspine_pw, rspine_mean, rspine_ok = compute_rspine_integrity(coords, pocket_seq)

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

    geom = KinaseGeometry(
        dfg_chelix_distance=dfg_dist,
        hinge_activation_angle=hinge_angle,
        glu_lys_distance=glu_lys_dist,
        salt_bridge_intact=sb_intact,
        asp_lys_distance=asp_lys_dist,
        dfg_asp_engaged=asp_engaged,
        activation_loop_bfactor_mean=bf_mean,
        activation_loop_bfactor_std=bf_std,
        bfactors_reliable=bf_reliable,
        rspine_distances=rspine_pw if rspine_pw else None,
        rspine_mean_distance=rspine_mean,
        rspine_intact=rspine_ok,
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

    # Assign confidence tier
    geom.confidence_tier = assign_confidence_tier(geom)

    return geom


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic pocket with known positions
    pocket = "A" * 85  # all resolved
    np.random.seed(42)
    coords = np.random.randn(85, 3) * 10  # random positions
    bfactors = np.random.uniform(10, 80, size=85).astype(np.float32)

    geom = compute_kinase_geometry(coords, pocket, bfactors=bfactors)
    print("Kinase Geometry (synthetic):")
    for k, v in geom.to_dict().items():
        print(f"  {k}: {v}")

    print(f"\nConfidence Tier: {geom.confidence_tier}")
    print(f"Salt Bridge: {geom.glu_lys_distance:.1f} Å (intact={geom.salt_bridge_intact})")
    print(f"Asp Coordination: {geom.asp_lys_distance:.1f} Å (engaged={geom.dfg_asp_engaged})")
    print(f"Activation Loop B-factor: {geom.activation_loop_bfactor_mean:.1f} Å²")
    print(f"R-spine mean: {geom.rspine_mean_distance:.1f} Å (intact={geom.rspine_intact})")

    # Test R-spine coordinate extraction (for GNN)
    rspine_coords = get_rspine_coordinates(coords, pocket)
    print(f"\nR-spine coords shape: {rspine_coords.shape}")
    print(f"R-spine coords:\n{rspine_coords}")
