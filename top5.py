#rdkit is package that analyzes chemical structures
#Chem is module for molecular creation and manipulation
from rdkit import Chem

#FIlterCatalog allows applying chemical filters
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

#AllChem has molecular fingerprints
from rdkit.Chem import AllChem, DataStructs

#Fingerprint Generaator is used to compare molecular similarity
from rdkit.Chem import rdFingerprintGenerator

import pandas as pd

#read top 50 hits csv
df = pd.read_csv("top50_upgraded_hits.csv")

#convert SMILES strings to RDKit molecules
df['molecules'] = df['ligand_smiles'].apply(lambda x: Chem.MolFromSmiles(x))

#removes rows where SMILES not converted
df = df[df['molecules'].notnull()].copy()

#creates the empty set of filter
parameter = FilterCatalogParams()
#add PAINS filter
parameter.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
#create catalog using parameter
catalog = FilterCatalog(parameter)

#checks if molecule is PAINS compound
def is_pains(molecule):
    return catalog.HasMatch(molecule)

#applies PAINS filter to all molecules
df['is_pains'] = df['molecules'].apply(is_pains)

#keep molecules that are not PAINS
df_no_pains = df[df['is_pains'] == False].copy()

#SMARTS patterns for reactive/toxic groups
toxic_smarts = [
    #aldehydes
    '[CX3H1](=O)[#6]',
    #peroxides
    '[OX2][OX2]', 
    #isocyanates
    'N=C=O',
    #azides
    '[NX3][NX2]=[NX2]'
]

patterns = []

#convert SMARTS strings to RDKit molecules
for toxic in toxic_smarts:
    patterns.append(Chem.MolFromSmarts(toxic))

#function checks if molecule contains toxic group
def has_toxic_group(molecule):
    for pattern in patterns:
        if molecule.HasSubstructMatch(pattern):
            return True
    return False

#applies toxic filter
df_no_pains['toxic_flag'] = df_no_pains['molecules'].apply(has_toxic_group)

#keep molecules that are not toxic
df_no_toxic = df_no_pains[df_no_pains['toxic_flag'] == False]

#sort molecules predicted activity score with highest first
df_no_toxic = df_no_toxic.sort_values(by='type2_score', ascending=False).reset_index(drop=True)

#create a Morgan fingerprint generator with radius = 2
fingerprint_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

#apply generator to all molecules
df_no_toxic['fingerprints'] = df_no_toxic['molecules'].apply(lambda m: fingerprint_gen.GetFingerprint(m))

#stores the indices of the selected molecules
selected_indices = []

#pick top 5 diverse molecules -> Tanimoto measures simialrity between 2 molecules, 1.0 means identical
for index, row in df_no_toxic.iterrows():
    current_fingerprint = row['fingerprints']
    if all(DataStructs.TanimotoSimilarity(current_fingerprint, df_no_toxic.loc[i, 'fingerprints']) < 0.5
        for i in selected_indices):
            selected_indices.append(index)
    if len(selected_indices) == 5:
        break

#top 5 rows
top5 = df_no_toxic.loc[selected_indices].drop(columns=['fingerprints'])

top5.to_csv("top5_final_hits.csv", index=False)
print("Top 5 filtered compounds saved to top5_final_hits.csv")