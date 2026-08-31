"""
Write an annotated PyMOL session script (.pml) for an analyzed kinase structure.

Opening the ``.pml`` in PyMOL loads the structure, colours the DFG motif and the
alphaC-helix, and places a label with the DFG call, inhibitor class, and the
DFG-to-alphaC distance -- so the annotation is visible on open.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

# KLIFS pocket positions used by the descriptors.
DFG_POS = (80, 81, 82)
ACHELIX_POS = tuple(range(20, 31))


def parse_pocket_meta(mol2_text: str) -> Tuple[Optional[str], Optional[str], Dict[int, int]]:
    """From a KLIFS pocket mol2: (pdb_code, chain, {pocket_pos: residue_number})."""
    pdb, chain = None, None
    lines = mol2_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("@<TRIPOS>MOLECULE") and i + 1 < len(lines):
            name = lines[i + 1].strip()  # e.g. HUMAN/ABL1_4twp_chainB
            m = re.search(r"_([0-9a-zA-Z]{4})_chain([A-Za-z0-9])", name)
            if m:
                pdb, chain = m.group(1).lower(), m.group(2)
            break
    pos_resnum: Dict[int, int] = {}
    in_atom = False
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>") and "ATOM" not in line:
            in_atom = False
        if in_atom and line.split() and line.split()[1] == "CA":
            p = line.split()
            m = re.match(r"[A-Za-z]+(-?\d+)", p[7])
            if m:
                pos_resnum[int(p[6])] = int(m.group(1))
    return pdb, chain, pos_resnum


def _sel(chain: Optional[str], resnums: List[int]) -> str:
    body = "resi " + "+".join(str(r) for r in resnums)
    return (f"chain {chain} and {body}") if chain else body


def build_pml(load_cmd: str, chain: Optional[str], pos_resnum: Dict[int, int],
              annotation: str) -> str:
    dfg = [pos_resnum[p] for p in DFG_POS if p in pos_resnum]
    ac = [pos_resnum[p] for p in ACHELIX_POS if p in pos_resnum]
    lines = [
        "# Hyaline kinase annotation (auto-generated)",
        f"# {annotation}",
        "reinitialize",
        load_cmd,
        "hide everything",
        "bg_color white",
        "show cartoon",
        "color grey80",
        "set cartoon_transparency, 0.25",
    ]
    if dfg:
        lines += [f"select dfg, {_sel(chain, dfg)}",
                  "color firebrick, dfg", "show sticks, dfg"]
    if ac:
        lines += [f"select achelix, {_sel(chain, ac)}",
                  "color marine, achelix"]
    if dfg:
        lines += [
            "pseudoatom hyaline_anno, selection=(dfg and name CA)",
            f'label hyaline_anno, "{annotation}"',
            "set label_size, 16",
            "set label_color, black",
            "orient dfg or achelix",
        ]
    lines += [f'print "Hyaline: {annotation}"', "deselect"]
    return "\n".join(lines) + "\n"


def write_session(out_path: str, load_cmd: str, chain: Optional[str],
                  pos_resnum: Dict[int, int], annotation: str) -> str:
    pml = build_pml(load_cmd, chain, pos_resnum, annotation)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(pml)
    return out_path
