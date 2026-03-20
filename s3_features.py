import pandas as pd

df = pd.read_csv("klifs_with_bioactivity.csv")

#targeted structures
structures = ["8ru8", "8tu4", "5x02", "2osc", "7a52", "3lpb", "7avx", "4xv9"]

#create a subset with specific targets and specified columns
subset = df[df["pdb"].str.lower().isin(structures)]
subset = subset[["kinase_name", "pdb", "dfg", "ac_helix", "type", "dfg_chelix_distance"]]

#flag based on biological inconsistencies
subset["Flag"] = None
subset.loc[(subset["ac_helix"] == "in") & (subset["dfg_chelix_distance"] > 10), "Flag"] = "Ac-helix in but large distance"
subset.loc[(subset["ac_helix"] == "out") & (subset["dfg_chelix_distance"] < 10), "Flag"] = "Ac-helix out but small distance"
subset.loc[(subset["dfg"] == "in") & (subset["dfg_chelix_distance"] > 10), "Flag"] = "DFG-in but large distance"
subset.loc[(subset["dfg"] == "out") & (subset["dfg_chelix_distance"] < 10), "Flag"] = "DFG-out but large distance"

subset.to_csv("targeted_kinases", index = False)