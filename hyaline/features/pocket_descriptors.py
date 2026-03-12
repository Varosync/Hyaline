"""
Pocket Physicochemical Descriptors
===================================

Compute pocket-level descriptors from the KLIFS 85-residue pocket alignment:

1. **Pocket Volume** – convex-hull volume of Cα coordinates (Å³)
2. **Electrostatic Surface Potential (proxy)** – charge distribution from
   residue-level partial charges (PARSE scale)
3. **Hydrophobicity** – Kyte-Doolittle scale statistics across the pocket

All functions work with the 85-character KLIFS pocket string and/or a
(≤85, 3) array of Cα coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physicochemical scales
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity (higher = more hydrophobic)
KYTE_DOOLITTLE: Dict[str, float] = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# PARSE partial charges (net charge at pH 7 per residue type)
PARSE_CHARGES: Dict[str, float] = {
    "A":  0.0, "C":  0.0, "D": -1.0, "E": -1.0, "F":  0.0,
    "G":  0.0, "H":  0.5, "I":  0.0, "K":  1.0, "L":  0.0,
    "M":  0.0, "N":  0.0, "P":  0.0, "Q":  0.0, "R":  1.0,
    "S":  0.0, "T":  0.0, "V":  0.0, "W":  0.0, "Y":  0.0,
}

# Molecular weight of each amino acid (daltons) – for sanity metrics
AA_MW: Dict[str, float] = {
    "A":  89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2,
    "G":  75.0, "H": 155.2, "I": 131.2, "K": 146.2, "L": 131.2,
    "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.2, "R": 174.2,
    "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2,
}

# Gap characters in KLIFS pocket alignment
GAP_CHARS = {"_", "-", " ", ""}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class PocketDescriptors:
    """Computed pocket-level physicochemical descriptors."""

    # Volume
    volume: float = 0.0                 # Å³ (convex hull)

    # Electrostatic surface potential proxy
    esp_mean: float = 0.0               # mean charge
    esp_std: float = 0.0                # charge std
    esp_range: float = 0.0              # max - min charge
    net_charge: float = 0.0             # sum of charges
    n_positive: int = 0                 # count of positively charged residues
    n_negative: int = 0                 # count of negatively charged residues

    # Hydrophobicity
    hydro_mean: float = 0.0
    hydro_std: float = 0.0
    hydro_max: float = 0.0
    hydro_min: float = 0.0
    frac_hydrophobic: float = 0.0       # fraction with KD > 0
    frac_hydrophilic: float = 0.0       # fraction with KD < -1

    # Composition
    n_resolved: int = 0                 # residues actually resolved (non-gap)
    n_missing: int = 0                  # gap positions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def compute_pocket_volume(coords: np.ndarray) -> float:
    """Convex-hull volume of pocket Cα coordinates.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Cα positions in Ångströms.  Must have N ≥ 4 non-degenerate points.

    Returns
    -------
    float
        Volume in ų.  Returns 0.0 if the hull cannot be computed.
    """
    if coords.shape[0] < 4:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords)
        return float(hull.volume)
    except Exception as exc:
        logger.debug("ConvexHull failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Electrostatic proxy
# ---------------------------------------------------------------------------

def _residue_charges(pocket_seq: str) -> np.ndarray:
    """Return an array of per-residue partial charges for non-gap positions."""
    charges = []
    for ch in pocket_seq:
        if ch.upper() in GAP_CHARS:
            continue
        charges.append(PARSE_CHARGES.get(ch.upper(), 0.0))
    return np.array(charges, dtype=np.float64)


def compute_esp_proxy(pocket_seq: str) -> Dict[str, float]:
    """Electrostatic surface potential proxy from residue charges.

    Returns a dict with keys: mean, std, range, net_charge, n_positive,
    n_negative.
    """
    charges = _residue_charges(pocket_seq)
    if len(charges) == 0:
        return dict(mean=0, std=0, range=0, net_charge=0, n_positive=0, n_negative=0)

    return dict(
        mean=float(np.mean(charges)),
        std=float(np.std(charges)),
        range=float(np.ptp(charges)),
        net_charge=float(np.sum(charges)),
        n_positive=int(np.sum(charges > 0)),
        n_negative=int(np.sum(charges < 0)),
    )


# ---------------------------------------------------------------------------
# Hydrophobicity
# ---------------------------------------------------------------------------

def _residue_hydro(pocket_seq: str) -> np.ndarray:
    vals = []
    for ch in pocket_seq:
        if ch.upper() in GAP_CHARS:
            continue
        vals.append(KYTE_DOOLITTLE.get(ch.upper(), 0.0))
    return np.array(vals, dtype=np.float64)


def compute_hydrophobicity(pocket_seq: str) -> Dict[str, float]:
    """Kyte-Doolittle hydrophobicity statistics for the pocket.

    Returns a dict with keys: mean, std, max, min, frac_hydrophobic,
    frac_hydrophilic.
    """
    vals = _residue_hydro(pocket_seq)
    if len(vals) == 0:
        return dict(mean=0, std=0, max=0, min=0, frac_hydrophobic=0, frac_hydrophilic=0)

    n = len(vals)
    return dict(
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        max=float(np.max(vals)),
        min=float(np.min(vals)),
        frac_hydrophobic=float(np.sum(vals > 0) / n),
        frac_hydrophilic=float(np.sum(vals < -1.0) / n),
    )


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def compute_pocket_descriptors(
    pocket_seq: str,
    coords: Optional[np.ndarray] = None,
) -> PocketDescriptors:
    """Compute all pocket descriptors from a KLIFS pocket string and optional Cα coords.

    Parameters
    ----------
    pocket_seq : str
        85-character KLIFS pocket sequence (gaps as ``_`` or ``-``).
    coords : ndarray, shape (N, 3), optional
        Cα coordinates for resolved pocket residues.  If provided, pocket
        volume is computed via convex hull.

    Returns
    -------
    PocketDescriptors
    """
    n_resolved = sum(1 for ch in pocket_seq if ch.upper() not in GAP_CHARS)
    n_missing = len(pocket_seq) - n_resolved

    # Volume
    volume = 0.0
    if coords is not None and coords.shape[0] >= 4:
        volume = compute_pocket_volume(coords)

    # ESP
    esp = compute_esp_proxy(pocket_seq)

    # Hydrophobicity
    hydro = compute_hydrophobicity(pocket_seq)

    return PocketDescriptors(
        volume=volume,
        esp_mean=esp["mean"],
        esp_std=esp["std"],
        esp_range=esp["range"],
        net_charge=esp["net_charge"],
        n_positive=esp["n_positive"],
        n_negative=esp["n_negative"],
        hydro_mean=hydro["mean"],
        hydro_std=hydro["std"],
        hydro_max=hydro["max"],
        hydro_min=hydro["min"],
        frac_hydrophobic=hydro["frac_hydrophobic"],
        frac_hydrophilic=hydro["frac_hydrophilic"],
        n_resolved=n_resolved,
        n_missing=n_missing,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example with a real-ish KLIFS pocket sequence for ABL1
    example_pocket = "KVEVGDIVEFITENYGEIIDELAQELKLYSVDAKFKVLKTEKADFILREATVEQLNEASDALEKDLKIVKD"
    # pad to 85
    example_pocket = example_pocket.ljust(85, "_")

    desc = compute_pocket_descriptors(example_pocket)
    print("Pocket Descriptors:")
    for k, v in desc.to_dict().items():
        print(f"  {k}: {v}")
