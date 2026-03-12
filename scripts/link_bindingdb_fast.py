#!/usr/bin/env python3
"""
Fast BindingDB Integration - Only processes PDB entries we need
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

print("\n" + "="*70)
print("  FAST BINDINGDB INTEGRATION")
print("="*70)

# Load our dataset
df = pd.read_csv('data/klifs_with_real_compound_features.csv')
print(f"\n  Dataset: {len(df)} structures")

# Get unique PDB-ligand pairs we need
needed_pairs = set()
for _, row in df.iterrows():
    pdb = str(row['pdb']).lower()
    ligand = str(row['ligand']).upper()
    needed_pairs.add((pdb, ligand))

print(f"  Need data for: {len(needed_pairs)} unique PDB-ligand pairs")

# Check if BindingDB exists
bindingdb_file = 'data/BindingDB_All.tsv'
import os
if not os.path.exists(bindingdb_file):
    print(f"\n  BindingDB not found - keeping mock pKi values")
    df.to_csv('data/klifs_with_bioactivity.csv', index=False)
    print(f"  ✓ Saved to: data/klifs_with_bioactivity.csv")
    exit(0)

print(f"\n  Scanning BindingDB for matching entries...")
print(f"  (Only reading rows with PDB data - much faster)")

# Scan file line by line, only process relevant rows
pdb_ligand_affinity = {}
header = None
pdb_col_idx = None
ligand_col_idx = None
ki_col_idx = None
ic50_col_idx = None
kd_col_idx = None

with open(bindingdb_file, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(tqdm(f, desc="Scanning", unit=" lines")):
        if i == 0:
            # Parse header
            header = line.strip().split('\t')
            # Find columns by partial match (handle encoding issues)
            for idx, col in enumerate(header):
                col_clean = col.strip()
                if 'Ligand HET ID' in col_clean:
                    ligand_col_idx = idx
                elif 'PDB ID(s) of Target Chain' in col_clean and pdb_col_idx is None:
                    pdb_col_idx = idx
                elif col_clean == 'Ki (nM)':
                    ki_col_idx = idx
                elif col_clean == 'IC50 (nM)':
                    ic50_col_idx = idx
                elif col_clean == 'Kd (nM)':
                    kd_col_idx = idx
            
            if pdb_col_idx is None or ligand_col_idx is None:
                print(f"  ✗ Required columns not found")
                print(f"  Available columns: {header[:30]}")
                break
            print(f"  ✓ Found columns: PDB={pdb_col_idx}, Ligand={ligand_col_idx}, Ki={ki_col_idx}, IC50={ic50_col_idx}, Kd={kd_col_idx}")
            continue
        
        # Quick check if line has PDB data (before splitting)
        if '\t\t' in line or not any(c.isdigit() for c in line):
            continue
        
        parts = line.strip().split('\t')
        if len(parts) <= max(pdb_col_idx, ligand_col_idx, ki_col_idx, ic50_col_idx, kd_col_idx):
            continue
        
        pdb_ids = parts[pdb_col_idx] if pdb_col_idx < len(parts) else ''
        ligand_het = parts[ligand_col_idx] if ligand_col_idx < len(parts) else ''
        
        if not pdb_ids or not ligand_het:
            continue
        
        # Check if this is a pair we need
        ligand_het = ligand_het.strip().upper()
        found_match = False
        for pdb_id in pdb_ids.split(','):
            pdb_id = pdb_id.strip().lower()
            if (pdb_id, ligand_het) in needed_pairs:
                found_match = True
                break
        
        if not found_match:
            continue
        
        # Extract best affinity
        best_pki = None
        for col_idx in [ki_col_idx, ic50_col_idx, kd_col_idx]:
            try:
                value_str = parts[col_idx] if col_idx < len(parts) else ''
                if value_str and value_str.strip():
                    value = float(value_str)
                    if value > 0:
                        pki = -np.log10(value * 1e-9)  # nM to M, then pKi
                        if best_pki is None or pki > best_pki:
                            best_pki = pki
            except:
                pass
        
        if best_pki:
            for pdb_id in pdb_ids.split(','):
                pdb_id = pdb_id.strip().lower()
                key = (pdb_id, ligand_het)
                if key in needed_pairs:
                    if key not in pdb_ligand_affinity or best_pki > pdb_ligand_affinity[key]:
                        pdb_ligand_affinity[key] = best_pki

print(f"\n  ✓ Found {len(pdb_ligand_affinity)} matches in BindingDB")

# Merge with dataset
df['pki_bindingdb'] = df.apply(
    lambda row: pdb_ligand_affinity.get((str(row['pdb']).lower(), str(row['ligand']).upper()), np.nan),
    axis=1
)

matched = df['pki_bindingdb'].notna().sum()
print(f"  ✓ Matched {matched}/{len(df)} structures ({100*matched/len(df):.1f}%)")

# Use BindingDB where available, keep mock otherwise
df['pki_original'] = df['pki']
df['pki'] = df['pki_bindingdb'].fillna(df['pki'])

# Save
df.to_csv('data/klifs_with_bioactivity.csv', index=False)
print(f"\n  ✓ Saved to: data/klifs_with_bioactivity.csv")

print(f"\n  pKi Statistics:")
print(f"    Range: {df['pki'].min():.2f} - {df['pki'].max():.2f}")
print(f"    Mean: {df['pki'].mean():.2f} ± {df['pki'].std():.2f}")
print(f"    Real values: {matched} ({100*matched/len(df):.1f}%)")
print(f"    Mock values: {len(df) - matched} ({100*(len(df)-matched)/len(df):.1f}%)")
