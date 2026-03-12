#!/usr/bin/env python3
"""
Add SMILES to Dataset
=====================

Fetches SMILES strings for ligands from PDB Chemical Component Dictionary.
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_ligand_smiles(ligand_code: str, cache: dict) -> str:
    """Fetch SMILES for a ligand from PDB.
    
    Parameters
    ----------
    ligand_code : str
        3-letter PDB ligand code
    cache : dict
        Cache of already fetched SMILES
        
    Returns
    -------
    str
        SMILES string or empty string if not found
    """
    if ligand_code in cache:
        return cache[ligand_code]
    
    # Skip common non-drug ligands
    skip_ligands = {'ATP', 'ADP', 'ANP', 'AMP', 'GTP', 'GDP', 'GNP', 'ACP', 
                    'AGS', 'AMP', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'ACT',
                    'DMS', 'BME', 'DTT', 'TRS', 'HOH', 'WAT', '0'}
    
    if ligand_code in skip_ligands or ligand_code == '0':
        cache[ligand_code] = ''
        return ''
    
    try:
        # PDB Chemical Component Dictionary API
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_code}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            smiles = data.get('rcsb_chem_comp_descriptor', {}).get('smiles', '')
            cache[ligand_code] = smiles
            return smiles
        else:
            cache[ligand_code] = ''
            return ''
            
    except Exception as e:
        logger.debug(f"Failed to fetch SMILES for {ligand_code}: {e}")
        cache[ligand_code] = ''
        return ''


def main():
    logger.info("="*70)
    logger.info("  ADD SMILES TO DATASET")
    logger.info("="*70)
    
    # Load dataset
    input_path = 'data/klifs_with_compound_features.csv'
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info(f"\nLoading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} structures")
    
    # Get unique ligands
    unique_ligands = df['ligand'].unique()
    logger.info(f"Found {len(unique_ligands)} unique ligands")
    
    # Fetch SMILES
    logger.info("\nFetching SMILES from PDB Chemical Component Dictionary...")
    smiles_cache = {}
    
    for ligand in tqdm(unique_ligands, desc="Fetching SMILES"):
        get_ligand_smiles(ligand, smiles_cache)
        time.sleep(0.1)  # Rate limiting
    
    # Add SMILES column
    logger.info("\nAdding SMILES to dataset...")
    df['ligand_smiles'] = df['ligand'].map(smiles_cache)
    
    # Statistics
    has_smiles = (df['ligand_smiles'] != '').sum()
    logger.info(f"\nSMILES Statistics:")
    logger.info(f"  Total structures: {len(df)}")
    logger.info(f"  With SMILES: {has_smiles} ({100*has_smiles/len(df):.1f}%)")
    logger.info(f"  Without SMILES: {len(df)-has_smiles} ({100*(len(df)-has_smiles)/len(df):.1f}%)")
    
    # Save
    output_path = 'data/klifs_with_smiles.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Saved dataset with SMILES to: {output_path}")
    
    # Show examples
    logger.info("\nExample SMILES:")
    examples = df[df['ligand_smiles'] != ''].head(5)
    for _, row in examples.iterrows():
        logger.info(f"  {row['ligand']}: {row['ligand_smiles'][:60]}...")


if __name__ == "__main__":
    main()
