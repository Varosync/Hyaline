#!/usr/bin/env python3
"""
Validate kinase model on real KLIFS data.
Tests if model correctly predicts Type I/II inhibitor preferences.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

# Known Type I/II inhibitors for validation
VALIDATION_DRUGS = {
    # Type II (DFG-out binders) - should predict negative ΔpKi
    'STI': {'name': 'Imatinib', 'type': 'II', 'targets': ['ABL1'], 'expected': 'negative'},
    'NIL': {'name': 'Nilotinib', 'type': 'II', 'targets': ['ABL1'], 'expected': 'negative'},
    'BAX': {'name': 'Sorafenib', 'type': 'II', 'targets': ['BRAF'], 'expected': 'negative'},
    
    # Type I (DFG-in binders) - should predict positive ΔpKi  
    'IRE': {'name': 'Gefitinib', 'type': 'I', 'targets': ['EGFR'], 'expected': 'positive'},
    'AEE': {'name': 'Erlotinib', 'type': 'I', 'targets': ['EGFR'], 'expected': 'positive'},
    'P30': {'name': 'Dasatinib', 'type': 'I', 'targets': ['ABL1', 'SRC'], 'expected': 'positive'},
}

BASE_URL = "https://klifs.net/api_v2"


def get_kinase_id(name: str) -> Optional[int]:
    """Get KLIFS kinase ID by name."""
    try:
        r = requests.get(f"{BASE_URL}/kinase_ID", params={'kinase_name': name, 'species': 'Human'}, timeout=30)
        data = r.json()
        if data and len(data) > 0:
            return data[0].get('kinase_ID')
    except Exception as e:
        print(f"  Error getting kinase ID for {name}: {e}")
    return None


def get_structures(kinase_id: int) -> List[Dict]:
    """Get all structures for a kinase."""
    try:
        r = requests.get(f"{BASE_URL}/structures_list", params={'kinase_ID': kinase_id}, timeout=30)
        return r.json() if r.status_code == 200 else []
    except Exception as e:
        print(f"  Error getting structures: {e}")
        return []


def analyze_kinase(kinase_name: str):
    """Analyze conformational distribution for a kinase."""
    print(f"\n{'='*60}")
    print(f"Kinase: {kinase_name}")
    print('='*60)
    
    kinase_id = get_kinase_id(kinase_name)
    if not kinase_id:
        print(f"  Could not find kinase ID")
        return None
    
    print(f"  KLIFS ID: {kinase_id}")
    
    structures = get_structures(kinase_id)
    print(f"  Total structures: {len(structures)}")
    
    if not structures:
        return None
    
    # Count conformations
    dfg_in = [s for s in structures if s.get('DFG') == 'in']
    dfg_out = [s for s in structures if 'out' in str(s.get('DFG', '')).lower()]
    
    print(f"  DFG-in: {len(dfg_in)}")
    print(f"  DFG-out: {len(dfg_out)}")
    
    # Analyze ligands
    ligands_in = set(s.get('ligand') for s in dfg_in if s.get('ligand'))
    ligands_out = set(s.get('ligand') for s in dfg_out if s.get('ligand'))
    
    print(f"  Unique ligands (DFG-in): {len(ligands_in)}")
    print(f"  Unique ligands (DFG-out): {len(ligands_out)}")
    
    # Check known drugs
    print(f"\n  Known inhibitors:")
    for pdb_code, drug_info in VALIDATION_DRUGS.items():
        if kinase_name in drug_info['targets']:
            in_dfg_in = pdb_code in ligands_in
            in_dfg_out = pdb_code in ligands_out
            conformation = "DFG-in" if in_dfg_in else ("DFG-out" if in_dfg_out else "not found")
            expected = "DFG-out" if drug_info['type'] == 'II' else "DFG-in"
            match = "✓" if conformation == expected else ("?" if conformation == "not found" else "✗")
            print(f"    {pdb_code} ({drug_info['name']}): Type {drug_info['type']} → {conformation} {match}")
    
    return {
        'kinase': kinase_name,
        'kinase_id': kinase_id,
        'n_structures': len(structures),
        'n_dfg_in': len(dfg_in),
        'n_dfg_out': len(dfg_out),
        'ligands_in': list(ligands_in)[:10],
        'ligands_out': list(ligands_out)[:10],
    }


def main():
    print("="*70)
    print("KLIFS DATA VALIDATION")
    print("Testing kinases with known Type I/II inhibitors")
    print("="*70)
    
    # Key kinases for validation
    kinases = ['ABL1', 'EGFR', 'BRAF', 'SRC', 'KIT']
    
    results = []
    for kinase in kinases:
        result = analyze_kinase(kinase)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Kinase':<10} {'Total':>8} {'DFG-in':>8} {'DFG-out':>8} {'Ratio':>10}")
    print("-"*50)
    for r in results:
        ratio = r['n_dfg_out'] / max(r['n_dfg_in'], 1)
        print(f"{r['kinase']:<10} {r['n_structures']:>8} {r['n_dfg_in']:>8} {r['n_dfg_out']:>8} {ratio:>10.1f}")
    
    # Save results
    Path('checkpoints').mkdir(exist_ok=True)
    with open('checkpoints/klifs_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved to checkpoints/klifs_validation.json")


if __name__ == "__main__":
    main()
