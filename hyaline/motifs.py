"""
GPCR Motif Detection Module
===========================
Detects conserved activation motifs in GPCR sequences using
sequence pattern matching. Provides motif masks for attention weighting.

Key Motifs (Class A GPCRs):
- DRY (TM3): Ionic lock, D/E-R-Y pattern
- NPxxY (TM7): Rotation switch, N-P-x-x-Y pattern  
- CWxP (TM6): Toggle switch, C-W-x-P pattern
- Sodium pocket: D2.50, S3.39, N7.45, S7.46

References:
- Ballesteros-Weinstein numbering system
- GPCRdb structure annotations
"""
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


# Motif patterns (regex)
MOTIF_PATTERNS = {
    'DRY': r'[DE]R[YF]',      # TM3 ionic lock (D/E-R-Y/F)
    'NPxxY': r'NP.{2}Y',      # TM7 rotation switch
    'CWxP': r'CW.P',          # TM6 toggle switch
    'PIF': r'P..[FY]',        # Connector motif (approximate)
}

# Conserved residue positions (approximate sequence positions)
# These are rough estimates for typical Class A GPCRs
# In production, use GPCRdb BW mapping
CONSERVED_POSITIONS = {
    # Format: 'motif_name': (start_fraction, end_fraction) of sequence
    'DRY': (0.35, 0.45),      # TM3 region
    'NPxxY': (0.85, 0.95),    # TM7 region (near C-terminus)
    'CWxP': (0.60, 0.75),     # TM6 region
}


def detect_motif(sequence: str, pattern: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of a motif pattern in sequence.
    
    Args:
        sequence: Amino acid sequence
        pattern: Regex pattern for motif
        
    Returns:
        List of (start, end) indices for each match
    """
    matches = []
    for match in re.finditer(pattern, sequence):
        matches.append((match.start(), match.end()))
    return matches


def detect_all_motifs(sequence: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Detect all conserved GPCR motifs in a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary mapping motif names to list of (start, end) positions
    """
    results = {}
    for motif_name, pattern in MOTIF_PATTERNS.items():
        matches = detect_motif(sequence, pattern)
        
        # Filter to expected region if we have position hints
        if motif_name in CONSERVED_POSITIONS:
            seq_len = len(sequence)
            start_frac, end_frac = CONSERVED_POSITIONS[motif_name]
            expected_start = int(seq_len * start_frac)
            expected_end = int(seq_len * end_frac)
            
            # Keep only matches in expected region
            filtered = [
                (s, e) for s, e in matches 
                if s >= expected_start and e <= expected_end
            ]
            # If no matches in expected region, keep all matches
            matches = filtered if filtered else matches
        
        results[motif_name] = matches
    
    return results


def get_motif_mask(sequence: str, motif_types: Optional[List[str]] = None) -> torch.Tensor:
    """
    Create binary mask indicating which residues are part of conserved motifs.
    
    Args:
        sequence: Amino acid sequence
        motif_types: List of motif types to include (default: all)
        
    Returns:
        Binary tensor [seq_len] where 1 = motif residue
    """
    seq_len = len(sequence)
    mask = torch.zeros(seq_len, dtype=torch.float32)
    
    motifs = detect_all_motifs(sequence)
    
    if motif_types is None:
        motif_types = list(MOTIF_PATTERNS.keys())
    
    for motif_name, positions in motifs.items():
        if motif_name in motif_types:
            for start, end in positions:
                mask[start:end] = 1.0
    
    return mask


def get_motif_type_embedding(sequence: str, num_types: int = 5) -> torch.Tensor:
    """
    Create motif type embedding for each residue.
    
    Args:
        sequence: Amino acid sequence
        num_types: Number of motif types (0=none, 1=DRY, 2=NPxxY, 3=CWxP, 4=PIF)
        
    Returns:
        Tensor [seq_len] with motif type ID for each residue
    """
    seq_len = len(sequence)
    types = torch.zeros(seq_len, dtype=torch.long)
    
    motifs = detect_all_motifs(sequence)
    
    type_mapping = {'DRY': 1, 'NPxxY': 2, 'CWxP': 3, 'PIF': 4}
    
    for motif_name, positions in motifs.items():
        if motif_name in type_mapping:
            type_id = type_mapping[motif_name]
            for start, end in positions:
                types[start:end] = type_id
    
    return types


def compute_key_distances(coords: np.ndarray, sequence: str) -> Dict[str, float]:
    """
    Compute distances between key residue pairs that indicate activation state.
    
    Key pairs:
    - DRY-TM6: D3.49 to E6.30 (ionic lock)
    - NPxxY: N7.49 to core residues
    
    Args:
        coords: [N, 3] Cα coordinates
        sequence: Amino acid sequence
        
    Returns:
        Dictionary of key distance measurements
    """
    results = {}
    motifs = detect_all_motifs(sequence)
    
    # DRY ionic lock distance (approximate)
    if motifs['DRY'] and motifs['CWxP']:
        dry_start = motifs['DRY'][0][0]  # D position
        cwxp_start = motifs['CWxP'][0][0]  # C position
        
        if dry_start < len(coords) and cwxp_start < len(coords):
            dist = np.linalg.norm(coords[dry_start] - coords[cwxp_start])
            results['DRY_CWxP_distance'] = float(dist)
    
    # NPxxY to DRY distance (activation signature)
    if motifs['NPxxY'] and motifs['DRY']:
        npxxy_start = motifs['NPxxY'][0][0]  # N position
        dry_start = motifs['DRY'][0][0]  # D position
        
        if npxxy_start < len(coords) and dry_start < len(coords):
            dist = np.linalg.norm(coords[npxxy_start] - coords[dry_start])
            results['NPxxY_DRY_distance'] = float(dist)
    
    return results


# Test function
def test_motif_detection():
    """Test motif detection on sample sequences."""
    # Sample Class A GPCR sequence fragment
    test_seq = "MVFLLAGFPFFQMGLSNTGVLDVVTCTCTRGTWPLSYYTNAGYFLNLAISLDRYLVALPLY"
    test_seq += "MLVFVSFVPNSLSRLWYYWFCWLPFFLALAMSFPCVPFCAQKNEKLSMFLAVFFNLMFPIIYAFSSQKVLVFLLK"
    test_seq += "CRYWNEQNPLFYVFFTNSFNKVNPTVYSPVVAMFF"
    
    print("Testing motif detection on sample sequence...")
    print(f"Sequence length: {len(test_seq)}")
    
    motifs = detect_all_motifs(test_seq)
    for name, positions in motifs.items():
        if positions:
            print(f"  {name}: {positions}")
            for start, end in positions:
                print(f"    Sequence: {test_seq[start:end]}")
        else:
            print(f"  {name}: Not found")
    
    mask = get_motif_mask(test_seq)
    print(f"\nMotif residues: {mask.sum().item()}/{len(test_seq)}")
    
    return motifs


if __name__ == '__main__':
    test_motif_detection()
