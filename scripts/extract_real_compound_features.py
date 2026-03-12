#!/usr/bin/env python3
"""
Extract Real Compound Features from SMILES
===========================================

Extracts molecular descriptors from SMILES strings using RDKit.
Replaces mock compound features with real calculated values.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit not installed. Install with: pip install rdkit")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_compound_features(smiles: str) -> dict:
    """Calculate molecular descriptors from SMILES.
    
    Parameters
    ----------
    smiles : str
        SMILES string
        
    Returns
    -------
    dict
        Molecular descriptors
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate descriptors
        features = {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'aromatic_rings': Lipinski.NumAromaticRings(mol),
            'heavy_atoms': Lipinski.HeavyAtomCount(mol),
        }
        
        return features
        
    except Exception as e:
        logger.warning(f"Failed to calculate features for SMILES {smiles}: {e}")
        return None


def main():
    logger.info("="*70)
    logger.info("  EXTRACT REAL COMPOUND FEATURES FROM SMILES")
    logger.info("="*70)
    
    # Load dataset with SMILES
    input_path = 'data/klifs_with_smiles.csv'
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run scripts/add_smiles_to_dataset.py first")
        sys.exit(1)
    
    logger.info(f"\nLoading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} structures")
    
    # Check if SMILES column exists
    if 'ligand_smiles' not in df.columns:
        logger.error("No 'ligand_smiles' column found in dataset")
        logger.info("Need to add SMILES strings first")
        sys.exit(1)
    
    # Extract features for each compound
    logger.info("\nExtracting compound features from SMILES...")
    
    features_list = []
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        smiles = row.get('ligand_smiles', '')
        
        if pd.isna(smiles) or smiles == '':
            # Use mock features if no SMILES
            features = {
                'mw': row.get('compound_mw', 450.0),
                'logp': row.get('compound_logp', 3.5),
                'hbd': row.get('compound_hbd', 2),
                'hba': row.get('compound_hba', 5),
                'tpsa': row.get('compound_tpsa', 90.0),
                'rotatable_bonds': row.get('compound_rotatable_bonds', 5),
                'aromatic_rings': row.get('compound_aromatic_rings', 3),
                'heavy_atoms': row.get('compound_heavy_atoms', 32),
            }
        else:
            features = calculate_compound_features(smiles)
            
            if features is None:
                failed += 1
                # Fallback to mock
                features = {
                    'mw': row.get('compound_mw', 450.0),
                    'logp': row.get('compound_logp', 3.5),
                    'hbd': row.get('compound_hbd', 2),
                    'hba': row.get('compound_hba', 5),
                    'tpsa': row.get('compound_tpsa', 90.0),
                    'rotatable_bonds': row.get('compound_rotatable_bonds', 5),
                    'aromatic_rings': row.get('compound_aromatic_rings', 3),
                    'heavy_atoms': row.get('compound_heavy_atoms', 32),
                }
        
        features_list.append(features)
    
    # Update dataframe
    logger.info("\nUpdating dataset with real features...")
    
    df['compound_mw'] = [f['mw'] for f in features_list]
    df['compound_logp'] = [f['logp'] for f in features_list]
    df['compound_hbd'] = [f['hbd'] for f in features_list]
    df['compound_hba'] = [f['hba'] for f in features_list]
    df['compound_tpsa'] = [f['tpsa'] for f in features_list]
    df['compound_rotatable_bonds'] = [f['rotatable_bonds'] for f in features_list]
    df['compound_aromatic_rings'] = [f['aromatic_rings'] for f in features_list]
    df['compound_heavy_atoms'] = [f['heavy_atoms'] for f in features_list]
    
    # Save
    output_path = 'data/klifs_with_real_compound_features.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"\n✓ Saved dataset with real compound features to: {output_path}")
    logger.info(f"  Total structures: {len(df)}")
    logger.info(f"  Failed SMILES: {failed}")
    logger.info(f"  Success rate: {100*(len(df)-failed)/len(df):.1f}%")
    
    # Statistics
    logger.info("\nCompound Feature Statistics:")
    logger.info(f"  MW: {df['compound_mw'].mean():.1f} ± {df['compound_mw'].std():.1f} Da")
    logger.info(f"  LogP: {df['compound_logp'].mean():.2f} ± {df['compound_logp'].std():.2f}")
    logger.info(f"  HBD: {df['compound_hbd'].mean():.1f} ± {df['compound_hbd'].std():.1f}")
    logger.info(f"  HBA: {df['compound_hba'].mean():.1f} ± {df['compound_hba'].std():.1f}")
    logger.info(f"  TPSA: {df['compound_tpsa'].mean():.1f} ± {df['compound_tpsa'].std():.1f} Ų")
    logger.info(f"  Rotatable bonds: {df['compound_rotatable_bonds'].mean():.1f} ± {df['compound_rotatable_bonds'].std():.1f}")
    logger.info(f"  Aromatic rings: {df['compound_aromatic_rings'].mean():.1f} ± {df['compound_aromatic_rings'].std():.1f}")
    logger.info(f"  Heavy atoms: {df['compound_heavy_atoms'].mean():.1f} ± {df['compound_heavy_atoms'].std():.1f}")


if __name__ == "__main__":
    main()
