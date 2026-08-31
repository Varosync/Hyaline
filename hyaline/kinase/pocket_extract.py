"""
Extract the KLIFS 85-residue pocket from an arbitrary local structure.
=====================================================================

Given a local PDB file (a crystal structure or a predicted / AlphaFold model)
and the kinase name, map it onto the KLIFS 85-residue pocket numbering and
return Cα coordinates per pocket position, so the geometric descriptors can be
computed without KLIFS having pre-processed the structure.

Method (no external alignment tools):
  1. Build a reference for the kinase from KLIFS: its full domain sequence
     (``structure_get_protein``) plus the mapping pocket-position -> residue
     number (``structure_get_pocket``).
  2. Global-align (Needleman-Wunsch) the local chain sequence to the reference.
  3. Transfer each of the 85 pocket positions from the reference to the local
     residue it aligns to, and read that residue's Cα coordinate.

References are cached under ``klifs_cache/refs/``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

BASE = "https://klifs.net/api_v2"
REF_DIR = "klifs_cache/refs"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C", "PYL": "K",
}


def _get(endpoint: str, params: Dict):
    return requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)


# ---------------------------------------------------------------------------
# mol2 parsing (KLIFS reference)
# ---------------------------------------------------------------------------

def _iter_ca(mol2_text: str):
    """Yield (subst_id, subst_name, xyz) for each CA atom in a mol2."""
    in_atom = False
    for line in mol2_text.splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>") and "ATOM" not in line:
            in_atom = False
        if in_atom and line.strip():
            p = line.split()
            if len(p) >= 8 and p[1] == "CA":
                try:
                    yield int(p[6]), p[7], np.array([float(p[2]), float(p[3]), float(p[4])])
                except ValueError:
                    pass


def _split_resname(subst_name: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"([A-Za-z]+)(-?\d+)", subst_name)
    if not m:
        return subst_name, None
    return m.group(1).upper(), int(m.group(2))


# ---------------------------------------------------------------------------
# Reference construction (per kinase, cached)
# ---------------------------------------------------------------------------

def _kinase_id(name: str) -> Optional[int]:
    d = _get("kinase_ID", {"kinase_name": name, "species": "Human"}).json()
    return d[0]["kinase_ID"] if d else None


def build_reference(kinase_name: str) -> Dict:
    """Return {'sequence': str, 'pos_to_index': {pocket_pos: ref_seq_index}}."""
    os.makedirs(REF_DIR, exist_ok=True)
    cache = os.path.join(REF_DIR, f"{kinase_name}.json")
    if os.path.exists(cache):
        return json.load(open(cache))

    kid = _kinase_id(kinase_name)
    if kid is None:
        raise ValueError(f"kinase '{kinase_name}' not found in KLIFS")
    sl = _get("structures_list", {"kinase_ID": [kid]}).json()
    ref = sorted([s for s in sl if s.get("pocket")],
                 key=lambda s: s.get("quality_score", 0), reverse=True)[0]
    sid = ref["structure_ID"]

    # reference domain sequence + resnum -> seq index
    prot = _get("structure_get_protein", {"structure_ID": sid}).text
    residues = []  # (resnum, aa1)
    for _, subst_name, _xyz in _iter_ca(prot):
        resname, resnum = _split_resname(subst_name)
        if resnum is None:
            continue
        residues.append((resnum, AA3TO1.get(resname, "X")))
    resnum_to_index = {rn: i for i, (rn, _) in enumerate(residues)}
    sequence = "".join(aa for _, aa in residues)

    # pocket position -> reference resnum -> ref seq index
    pocket = _get("structure_get_pocket", {"structure_ID": sid}).text
    pos_to_index = {}
    for subst_id, subst_name, _xyz in _iter_ca(pocket):
        _resname, resnum = _split_resname(subst_name)
        if resnum in resnum_to_index:
            pos_to_index[subst_id] = resnum_to_index[resnum]

    ref = {"sequence": sequence, "pos_to_index": {str(k): v for k, v in pos_to_index.items()},
           "structure_ID": sid, "pdb": ref.get("pdb")}
    json.dump(ref, open(cache, "w"))
    return ref


# ---------------------------------------------------------------------------
# Local PDB parsing
# ---------------------------------------------------------------------------

def parse_pdb_ca(path: str, chain: Optional[str] = None) -> List[Tuple[int, str, np.ndarray]]:
    """Return [(resseq, aa1, xyz), ...] for the chosen chain, in file order."""
    per_chain: Dict[str, List] = {}
    seen = set()
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        ch = line[21]
        resname = line[17:20].strip().upper()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        icode = line[26]
        key = (ch, resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        per_chain.setdefault(ch, []).append((resseq, AA3TO1.get(resname, "X"), xyz))
    if not per_chain:
        raise ValueError(f"no CA atoms parsed from {path}")
    if chain is not None:
        if chain not in per_chain:
            raise ValueError(f"chain '{chain}' not found; chains: {list(per_chain)}")
        return per_chain[chain]
    # default: chain with the most residues
    return max(per_chain.values(), key=len)


# ---------------------------------------------------------------------------
# Needleman-Wunsch global alignment
# ---------------------------------------------------------------------------

def nw_align(a: str, b: str, match=1, mismatch=-1, gap=-1) -> Dict[int, int]:
    """Global-align a vs b; return {index_in_a: index_in_b} for aligned pairs."""
    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    score[:, 0] = np.arange(n + 1) * gap  # first column: i gaps
    score[0, :] = np.arange(m + 1) * gap  # first row: j gaps
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = score[i], score[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (match if ai == b[j - 1] else mismatch)
            row[j] = max(diag, prev[j] + gap, row[j - 1] + gap)
    # traceback
    i, j = n, m
    amap: Dict[int, int] = {}
    while i > 0 and j > 0:
        cur = score[i, j]
        if cur == score[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            amap[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif cur == score[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return amap


# ---------------------------------------------------------------------------
# Public: extract pocket CA from a local structure
# ---------------------------------------------------------------------------

def extract_pocket_ca(pdb_path: str, kinase_name: str,
                      chain: Optional[str] = None) -> Tuple[Dict[int, np.ndarray], Dict]:
    """Return ({pocket_pos: CA xyz}, info) for a local PDB of ``kinase_name``."""
    ref = build_reference(kinase_name)
    ref_seq = ref["sequence"]
    pos_to_ref = {int(k): v for k, v in ref["pos_to_index"].items()}

    local = parse_pdb_ca(pdb_path, chain)
    local_seq = "".join(aa for _, aa, _ in local)

    ref_to_local = nw_align(ref_seq, local_seq)  # ref index -> local index

    ca: Dict[int, np.ndarray] = {}
    mapped = 0
    for pos, ref_idx in pos_to_ref.items():
        li = ref_to_local.get(ref_idx)
        if li is not None:
            ca[pos] = local[li][2]
            mapped += 1
    info = {"kinase": kinase_name, "reference_pdb": ref.get("pdb"),
            "pocket_positions_mapped": mapped, "local_residues": len(local),
            "identity_to_reference": round(
                sum(1 for r, l in ref_to_local.items() if ref_seq[r] == local_seq[l])
                / max(len(ref_to_local), 1), 3)}
    return ca, info
