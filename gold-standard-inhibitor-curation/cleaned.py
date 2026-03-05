import pandas as pd

# CSV files from KLIFS
df_in = pd.read_csv("KLIFS_DFG-in_export.csv")
df_out = pd.read_csv("KLIFS_DFG-Out_export.csv")

# Remove any duplicate ligands
df_in_unique = df_in.drop_duplicates(subset=["LIGAND"])
df_out_unique = df_out.drop_duplicates(subset=["LIGAND"])

# Add column for type I/II
df_in_unique.insert(13, "TYPE", "Type I")
df_out_unique.insert(13, "TYPE", "Type II")

# Create CSV with updates
df_in_unique.to_csv("DFG-in_No_Duplicates_export.csv", index=False)
df_out_unique.to_csv("DFG-out_No_Duplicates_export.csv", index=False)

# Open lit validated CSV files
df_in_updated = pd.read_csv("DFG-in_Validated_export.csv")
df_out_updated = pd.read_csv("DFG-out_Validated_export.csv")

# Filter based off of only Yes validation 
df_out_filtered = df_out_updated[df_out_updated["TYPE Validated"] == "Y"]
df_out_filtered.to_csv("DFG-out_Gold_Standard.csv", index=False)

df_in_filtered = df_in_updated[df_in_updated["TYPE Validated"] == "Y"]
df_in_filtered.to_csv("DFG-in_Gold_Standard.csv", index=False)

# Merged CSV with all type I/II inhibitors
df_merged = pd.concat([df_in_filtered, df_out_filtered], ignore_index=True)
df_merged.to_csv("data/known_inhibitors_curated.csv", index=False)

# Number of inhibitors for each type
print("Total DFG-in(type I) inhibitors: ", len(df_in_filtered) - 1)
print("Total DFG-out(type II) inhibitors: ", len(df_out_filtered) - 1)

