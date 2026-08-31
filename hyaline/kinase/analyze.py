"""
Kinase structure annotation (Phase 1 `analyze`).
================================================

Annotate a single kinase structure with its conformational state and an
inhibitor-class call, from interpretable geometric descriptors computed on the
real 85-residue KLIFS pocket.

Input may be:
  * a KLIFS ``structure_ID`` (int)              -> experimental
  * a 4-letter PDB code present in KLIFS (str)  -> experimental
  * a local KLIFS-format pocket ``.mol2`` file  -> provenance set by caller
    (this is the path for predicted / AlphaFold models once their pocket has
    been extracted into KLIFS 85-residue numbering)

Output is a fixed-schema dict (``schema_version`` 1.0). Every field records what
drove it, and provenance (experimental vs predicted) is always tagged.

The DFG call uses a logistic model over the two descriptors trained on real
KLIFS data (``dfg_model.json``; grouped leave-one-kinase-out AUROC 0.834). The
raw descriptors are training-free physical measurements and cannot leak.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import requests

BASE = "https://klifs.net/api_v2"
SCHEMA_VERSION = "1.0"
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "dfg_model.json")

# Distance above which the pocket is so open that an allosteric / back-pocket
# mode is likely accessible (heuristic, from the DFG-out tail in real data).
ALLOSTERIC_DISTANCE_A = 16.0


@dataclass
class AnalysisResult:
    schema_version: str
    identifier: str
    source: str
    provenance: str
    dfg_achelix_distance_A: Optional[float]
    hinge_activation_angle_deg: Optional[float]
    n_pocket_residues_resolved: int
    dfg_call: Optional[str]
    dfg_confidence: Optional[float]
    dfg_driver: str
    achelix_state: str
    achelix_source: str
    inhibitor_class: Optional[str]
    inhibitor_rationale: str
    warnings: list

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Descriptor computation (self-contained; mirrors scripts/kinase_descriptors.py)
# ---------------------------------------------------------------------------

def _parse_pocket_ca(mol2_text: str) -> Dict[int, np.ndarray]:
    ca, in_atom = {}, False
    for line in mol2_text.splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>") and "ATOM" not in line:
            in_atom = False
        if in_atom and line.strip():
            p = line.split()
            if len(p) >= 7 and p[1] == "CA":
                try:
                    ca[int(p[6])] = np.array([float(p[2]), float(p[3]), float(p[4])])
                except ValueError:
                    pass
    return ca


def _dfg_achelix_distance(ca: Dict[int, np.ndarray]) -> Optional[float]:
    dfg = [ca[i] for i in (80, 81, 82) if i in ca]
    ac = [ca[i] for i in range(20, 31) if i in ca]
    if not dfg or not ac:
        return None
    return float(np.linalg.norm(np.mean(dfg, 0) - np.mean(ac, 0)))


def _hinge_activation_angle(ca: Dict[int, np.ndarray]) -> Optional[float]:
    hinge = [ca[i] for i in (46, 47, 48) if i in ca]
    actloop = [ca[i] for i in range(72, 86) if i in ca]
    lys = ca.get(17)
    if not hinge or not actloop or lys is None:
        return None
    a = np.mean(hinge, 0) - lys
    b = np.mean(actloop, 0) - lys
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


# ---------------------------------------------------------------------------
# DFG model (dependency-light: plain sigmoid over the two descriptors)
# ---------------------------------------------------------------------------

def _load_model() -> Dict:
    with open(_MODEL_PATH) as f:
        return json.load(f)


def _dfg_probability(distance: float, angle: float, model: Dict) -> float:
    w = model["weights"]
    z = w[0] * distance + w[1] * angle + model["intercept"]
    return 1.0 / (1.0 + math.exp(-z))  # P(DFG-out)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: Dict):
    return requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)


def _resolve_klifs(structure_id: int) -> Dict:
    """Fetch pocket mol2 + metadata for a KLIFS structure_ID."""
    txt = _get("structure_get_pocket", {"structure_ID": structure_id}).text
    meta = {}
    try:
        # structures_pdb_list needs pdb; instead pull DFG/aC via list if cheap.
        pass
    except Exception:
        pass
    return {"mol2": txt, "meta": meta}


def _structure_from_pdb(pdb: str) -> Optional[Dict]:
    r = _get("structures_pdb_list", {"pdb-codes": [pdb.lower()]})
    data = r.json() if r.status_code == 200 else []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _load_sibling(module_name: str):
    """Load a sibling module by path (works standalone or as a package)."""
    import importlib.util
    import sys
    p = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(f"hyaline_kinase_{module_name}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _load_pocket_extract():
    return _load_sibling("pocket_extract")


def analyze(
    identifier=None,
    provenance: str = "experimental",
    local_mol2: Optional[str] = None,
    local_pdb: Optional[str] = None,
    kinase: Optional[str] = None,
    chain: Optional[str] = None,
    pymol_out: Optional[str] = None,
) -> AnalysisResult:
    """Annotate one kinase structure.

    Parameters
    ----------
    identifier : int | str
        KLIFS ``structure_ID`` (int), a PDB code (str), or a label for a local
        file input.
    provenance : str
        ``experimental`` (default), ``predicted`` (e.g. AlphaFold), or ``unknown``.
    local_mol2 : str, optional
        Path to a KLIFS-format 85-residue pocket ``.mol2`` (coordinates read directly).
    local_pdb : str, optional
        Path to an arbitrary PDB (crystal or predicted/AlphaFold). Requires
        ``kinase``; the 85-residue pocket is extracted by aligning to a KLIFS
        reference for that kinase.
    kinase : str, optional
        Kinase name (required with ``local_pdb``).
    chain : str, optional
        Chain to use in ``local_pdb`` (default: the chain with the most residues).
    """
    warnings: list = []
    achelix_state, achelix_source = "unknown", "not_computed"
    ca = None
    mol2 = None
    extract_info = None
    if local_pdb:
        source = "local_pdb"
    elif local_mol2:
        source = "local_mol2"
    elif isinstance(identifier, int):
        source = "klifs_structure_id"
    else:
        source = "pdb"

    # --- obtain pocket coordinates ---
    if local_pdb:
        if not kinase:
            raise ValueError("local_pdb requires the 'kinase' argument")
        pe = _load_pocket_extract()
        ca, info = pe.extract_pocket_ca(local_pdb, kinase, chain=chain)
        extract_info = info
        identifier = identifier or os.path.basename(local_pdb)
        if info["pocket_positions_mapped"] < 60:
            warnings.append(f"only {info['pocket_positions_mapped']}/85 pocket "
                            "positions mapped; descriptors may be unreliable")
        if info["identity_to_reference"] < 0.5:
            warnings.append(f"low identity to KLIFS reference "
                            f"({info['identity_to_reference']}); wrong kinase?")
    elif local_mol2:
        with open(local_mol2) as f:
            mol2 = f.read()
    elif isinstance(identifier, int):
        mol2 = _resolve_klifs(identifier)["mol2"]
    else:
        meta = _structure_from_pdb(str(identifier))
        if meta is None:
            warnings.append(f"PDB '{identifier}' not found in KLIFS")
            return AnalysisResult(
                SCHEMA_VERSION, str(identifier), source, provenance,
                None, None, 0, None, None, "dfg_achelix_distance_A",
                achelix_state, achelix_source, None,
                "no structure available", warnings)
        sid = meta["structure_ID"]
        mol2 = _resolve_klifs(sid)["mol2"]
        achelix_state = str(meta.get("aC_helix") or "unknown")
        achelix_source = "klifs_annotation"

    if ca is None:  # not the local_pdb path
        ca = _parse_pocket_ca(mol2)
    dist = _dfg_achelix_distance(ca)
    angle = _hinge_activation_angle(ca)
    n_res = len(ca)

    if dist is None or angle is None:
        warnings.append("insufficient resolved pocket residues for descriptors")
        return AnalysisResult(
            SCHEMA_VERSION, str(identifier), source, provenance,
            dist, angle, n_res, None, None, "dfg_achelix_distance_A",
            achelix_state, achelix_source, None,
            "descriptors unavailable", warnings)

    # --- DFG call ---
    model = _load_model()
    p_out = _dfg_probability(dist, angle, model)
    dfg_call = "DFG-out" if p_out >= 0.5 else "DFG-in"
    confidence = round(p_out if dfg_call == "DFG-out" else 1.0 - p_out, 3)

    # --- inhibitor-class call ---
    if dist >= ALLOSTERIC_DISTANCE_A:
        inhibitor = "allosteric-accessible"
        rationale = (f"DFG-to-alphaC distance {dist:.1f} A exceeds "
                     f"{ALLOSTERIC_DISTANCE_A:.0f} A: back pocket likely open")
    elif dfg_call == "DFG-out":
        inhibitor = "Type II"
        rationale = "DFG-out exposes the allosteric pocket engaged by Type II inhibitors"
    else:
        inhibitor = "Type I"
        rationale = "DFG-in active-like pocket engaged by Type I inhibitors"

    # --- optional annotated PyMOL session ---
    if pymol_out:
        try:
            px = _load_sibling("pymol_export")
            annotation = f"{dfg_call} | {inhibitor} | DFG-aC {dist:.1f}A"
            if extract_info is not None:  # local PDB
                load_cmd = f"load {os.path.abspath(local_pdb)}, model"
                px.write_session(pymol_out, load_cmd, extract_info.get("chain"),
                                 {int(k): v for k, v in extract_info["resnums"].items()},
                                 annotation)
            elif mol2 is not None:  # KLIFS / PDB / local mol2
                pdb, ch, pos_resnum = px.parse_pocket_meta(mol2)
                if pdb:
                    load_cmd = f"fetch {pdb}, async=0"
                elif local_mol2:
                    load_cmd = f"load {os.path.abspath(local_mol2)}"
                else:
                    load_cmd = "# no loadable structure"
                px.write_session(pymol_out, load_cmd, ch, pos_resnum, annotation)
            else:
                warnings.append("pymol export skipped: no structure source")
        except Exception as e:  # never fail the analysis over the export
            warnings.append(f"pymol export failed: {e}")

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        identifier=str(identifier),
        source=source,
        provenance=provenance,
        dfg_achelix_distance_A=round(dist, 3),
        hinge_activation_angle_deg=round(angle, 3),
        n_pocket_residues_resolved=n_res,
        dfg_call=dfg_call,
        dfg_confidence=confidence,
        dfg_driver="dfg_achelix_distance_A",
        achelix_state=achelix_state,
        achelix_source=achelix_source,
        inhibitor_class=inhibitor,
        inhibitor_rationale=rationale,
        warnings=warnings,
    )
