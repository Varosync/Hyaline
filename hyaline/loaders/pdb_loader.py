"""
PDB Structure Loader for TF-DNA Complexes
=========================================
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

TF_PDB_MAPPING = {
    # Cardiac / Hematopoietic
    'GATA4': ['4xc3', '3dfx', '4hca'],
    'GATA3': ['3dfx', '4hca'],
    'NKX2-5': ['3rkq'],
    'RUNX1': ['1io4', '1h9d'],  # Runt domain + CBFβ
    
    # Neural / Melanocyte
    'SOX10': ['4ema', '3u2b'],
    'SOX2': ['6t78'],
    'SOX11': ['4ybq'],
    'MITF': ['4ati'],
    'PAX6': ['6pax', '1mdm'],  # Paired domain + Homeodomain
    
    # Hepatocyte
    'HNF4A': ['4iqr', '3cbb'],
    
    # ETS family
    'ETV1': ['4avp', '4bnc'],  # ETS domain
    'ETS1': ['1k79'],  # High resolution ETS
}

ELEMENT_ENCODING = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5, 'ZN': 6}
AA_ENCODING = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6,
    'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13,
    'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'DA': 20, 'DT': 21, 'DG': 22, 'DC': 23, 'A': 20, 'T': 21, 'G': 22, 'C': 23,
}

@dataclass
class StructureFeatures:
    node_features: np.ndarray
    pos: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    is_dna: np.ndarray
    pdb_id: str
    tf_name: str

def load_pdb_structure(pdb_path: Path, tf_name: str = "", cutoff: float = 8.0,
                       node_dim: int = 32, edge_dim: int = 8, max_atoms: int = 1500):
    if not HAS_BIOPYTHON:
        raise ImportError("Biopython required")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, pdb_path)
    
    coords, elements, res_types, is_dna = [], [], [], []
    dna_res = {'DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C', 'U'}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.resname.strip()
                res_is_dna = res_name in dna_res
                res_is_protein = is_aa(residue, standard=True)
                if not res_is_dna and not res_is_protein:
                    continue
                for atom in residue:
                    coords.append(atom.coord)
                    elements.append(atom.element.strip())
                    res_types.append(res_name)
                    is_dna.append(res_is_dna)
        break
    
    if len(coords) == 0:
        return None
    
    coords = np.array(coords, dtype=np.float32)
    N = len(coords)
    
    if N > max_atoms:
        idx = np.sort(np.random.choice(N, max_atoms, replace=False))
        coords = coords[idx]
        elements = [elements[i] for i in idx]
        res_types = [res_types[i] for i in idx]
        is_dna = [is_dna[i] for i in idx]
        N = max_atoms
    
    # Build edges
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    mask = (dist < cutoff) & (dist > 0.5)
    edge_index = np.array(np.where(mask), dtype=np.int64)
    
    # Edge features
    if edge_index.shape[1] > 0:
        src, dst = edge_index
        edge_dist = dist[src, dst]
        edge_dir = diff[src, dst] / (edge_dist[:, None] + 1e-8)
        edge_attr = np.concatenate([edge_dist[:, None], edge_dir,
                                    (edge_dist < 3.5).astype(np.float32)[:, None]], axis=-1)
        if edge_attr.shape[-1] < edge_dim:
            edge_attr = np.pad(edge_attr, [(0,0), (0, edge_dim - edge_attr.shape[-1])])
    else:
        edge_attr = np.zeros((0, edge_dim), dtype=np.float32)
    
    # Node features
    node_features = np.zeros((N, node_dim), dtype=np.float32)
    for i, (elem, res, dna) in enumerate(zip(elements, res_types, is_dna)):
        elem_idx = ELEMENT_ENCODING.get(elem, 7)
        if elem_idx < 8:
            node_features[i, elem_idx] = 1.0
        res_idx = AA_ENCODING.get(res, 24)
        if 8 + res_idx < node_dim:
            node_features[i, 8 + res_idx] = 1.0
    
    return StructureFeatures(
        node_features=node_features, pos=coords, edge_index=edge_index,
        edge_attr=edge_attr[:, :edge_dim], is_dna=np.array(is_dna, dtype=bool),
        pdb_id=pdb_path.stem, tf_name=tf_name,
    )

def load_tf_structures(data_dir: Path, tf_names: Optional[List[str]] = None, **kwargs):
    if tf_names is None:
        tf_names = list(TF_PDB_MAPPING.keys())
    results = {}
    for tf in tf_names:
        pdb_ids = TF_PDB_MAPPING.get(tf, [])
        structs = []
        for pdb_id in pdb_ids:
            path = data_dir / f"{pdb_id}.pdb"
            if path.exists():
                feat = load_pdb_structure(path, tf_name=tf, **kwargs)
                if feat:
                    structs.append(feat)
                    print(f"Loaded {pdb_id} for {tf}: {len(feat.pos)} atoms")
        if structs:
            results[tf] = structs
    return results
