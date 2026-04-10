#!/usr/bin/env python3
"""
Pull real Ki/Kd/IC50 values from ChEMBL for KLIFS kinase-ligand pairs.

Strategy:
1. Get unique SMILES from our dataset
2. For each kinase, query ChEMBL for bioactivity data
3. Match by InChIKey (canonical structure match)
4. Keep only experimental Ki/Kd values (not IC50 which is assay-dependent)
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def get_chembl_target_id(kinase_name):
    """Look up ChEMBL target ID for a kinase name."""
    try:
        resp = requests.get(
            f"{CHEMBL_API}/target/search.json",
            params={"q": kinase_name, "limit": 5},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            targets = data.get("targets", [])
            for t in targets:
                if t.get("organism", "") == "Homo sapiens" and t.get("target_type", "") == "SINGLE PROTEIN":
                    return t.get("target_chembl_id")
        return None
    except Exception as e:
        logger.warning(f"ChEMBL target lookup failed for {kinase_name}: {e}")
        return None


def get_bioactivity_for_target(target_id, limit=1000):
    """Get Ki/Kd bioactivity data for a ChEMBL target."""
    try:
        resp = requests.get(
            f"{CHEMBL_API}/activity.json",
            params={
                "target_chembl_id": target_id,
                "standard_type__in": "Ki,Kd",
                "standard_relation": "=",
                "limit": limit,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("activities", [])
        return []
    except Exception as e:
        logger.warning(f"ChEMBL activity query failed for {target_id}: {e}")
        return []


def smiles_to_inchikey(smiles):
    """Convert SMILES to InChIKey for matching."""
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolFromInchi, MolToInchi, InchiToInchiKey
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = MolToInchi(mol)
        if inchi is None:
            return None
        return InchiToInchiKey(inchi)
    except Exception:
        return None


def main():
    df = pd.read_csv("data/klifs_with_bioactivity.csv")
    logger.info(f"Loaded {len(df)} structures, {df['kinase_name'].nunique()} kinases")

    # Build InChIKey index from our SMILES
    logger.info("Computing InChIKeys for KLIFS ligands...")
    smiles_to_ik = {}
    for smi in tqdm(df["ligand_smiles"].dropna().unique(), desc="InChIKeys"):
        ik = smiles_to_inchikey(str(smi))
        if ik:
            smiles_to_ik[smi] = ik

    logger.info(f"Computed {len(smiles_to_ik)} InChIKeys from {df['ligand_smiles'].notna().sum()} SMILES")

    # Add InChIKey column to df
    df["inchikey"] = df["ligand_smiles"].map(smiles_to_ik)

    # Query ChEMBL for top kinases (by structure count)
    top_kinases = df["kinase_name"].value_counts().head(50).index.tolist()

    all_matches = []
    chembl_cache = {}

    for kinase in tqdm(top_kinases, desc="ChEMBL queries"):
        target_id = get_chembl_target_id(kinase)
        if target_id is None:
            continue

        activities = get_bioactivity_for_target(target_id)
        if not activities:
            continue

        # Build ChEMBL InChIKey -> pKi map
        chembl_pki = {}
        for act in activities:
            smi = act.get("canonical_smiles")
            val = act.get("standard_value")
            units = act.get("standard_units")

            if smi is None or val is None:
                continue

            try:
                val = float(val)
            except (ValueError, TypeError):
                continue

            # Convert to pKi (val is in nM for Ki/Kd)
            if units == "nM" and val > 0:
                pki = -np.log10(val * 1e-9)
            elif units == "uM" and val > 0:
                pki = -np.log10(val * 1e-6)
            else:
                continue

            ik = smiles_to_inchikey(smi)
            if ik:
                if ik not in chembl_pki or pki > chembl_pki[ik]:
                    chembl_pki[ik] = pki

        # Match to our structures
        kinase_df = df[df["kinase_name"] == kinase]
        for idx, row in kinase_df.iterrows():
            ik = row.get("inchikey")
            if ik and ik in chembl_pki:
                all_matches.append({
                    "structure_id": row["structure_id"],
                    "kinase_name": kinase,
                    "pdb": row["pdb"],
                    "ligand": row["ligand"],
                    "dfg": row["dfg"],
                    "chembl_pki": chembl_pki[ik],
                    "original_pki": row["pki"],
                    "inchikey": ik,
                })

        if len(chembl_pki) > 0:
            n_matched = sum(1 for _, r in kinase_df.iterrows() if r.get("inchikey") in chembl_pki)
            logger.info(f"  {kinase}: {len(activities)} ChEMBL activities, {len(chembl_pki)} unique, {n_matched} matched to KLIFS")

    matches_df = pd.DataFrame(all_matches)
    logger.info(f"\nTotal ChEMBL matches: {len(matches_df)}")
    logger.info(f"Unique structures: {matches_df['structure_id'].nunique()}")
    logger.info(f"Unique kinases: {matches_df['kinase_name'].nunique()}")

    if len(matches_df) > 0:
        # Merge back into main dataset
        chembl_map = matches_df.set_index("structure_id")["chembl_pki"].to_dict()
        df["pki_chembl"] = df["structure_id"].map(chembl_map)

        n_with_chembl = df["pki_chembl"].notna().sum()
        logger.info(f"Structures with real ChEMBL pKi: {n_with_chembl}/{len(df)}")

        # Compare imputed vs real
        both = df[df["pki_chembl"].notna()].copy()
        if len(both) > 0:
            corr = both["pki"].corr(both["pki_chembl"])
            mae = (both["pki"] - both["pki_chembl"]).abs().mean()
            print(f"\nImputed vs ChEMBL pKi:")
            print(f"  Correlation: {corr:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  ChEMBL pKi range: [{both['pki_chembl'].min():.1f}, {both['pki_chembl'].max():.1f}]")

        # Save
        df.to_csv("data/klifs_with_real_pki.csv", index=False)
        matches_df.to_csv("data/chembl_matches.csv", index=False)
        print(f"\nSaved to data/klifs_with_real_pki.csv")
        print(f"Saved to data/chembl_matches.csv")
    else:
        print("No ChEMBL matches found")


if __name__ == "__main__":
    main()
