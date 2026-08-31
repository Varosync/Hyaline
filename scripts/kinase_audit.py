#!/usr/bin/env python3
"""
Kinase reproducibility audit
============================

Regenerates the real-data numbers reported in research/kinase/README.md so that
every claim is backed by a command:

  1. Builds (and caches) a conformation-annotated KLIFS dataset.
  2. DFG-state classification from the 85-residue pocket SEQUENCE, evaluated
     ungrouped (leaks kinase identity) vs grouped leave-one-kinase-out (honest).
  3. DFG-state separation from a real GEOMETRIC descriptor (DFG-to-alphaC
     distance) parsed from KLIFS pocket coordinates -- training-free, no leakage.

No GPU, no retraining. Network access to https://klifs.net is required on the
first run; results are cached under klifs_cache/.

Usage:
    python scripts/kinase_audit.py
"""
import json
import os

import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = "https://klifs.net/api_v2"
CACHE = "klifs_cache"
POCKETS = os.path.join(CACHE, "pockets")
AA = "ACDEFGHIKLMNPQRSTVWY_-"
AA_IDX = {c: i for i, c in enumerate(AA)}

# Kinases used for the audit (name -> resolved at runtime). Chosen for coverage
# of both DFG-in and DFG-out structures.
KINASES = ["ABL1", "EGFR", "BRAF", "SRC", "KIT", "MAPK14", "MET", "KDR",
           "CDK2", "LCK", "MAPK1", "AKT1", "CSF1R", "DDR1", "FGFR1"]

# Kinases with enough DFG-out structures to test the geometric descriptor cheaply.
GEOM_KINASES = ["ABL1", "EGFR", "BRAF", "KIT", "CDK2", "MAPK14", "MET"]


def _get(endpoint, params):
    return requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)


def kinase_id(name):
    d = _get("kinase_ID", {"kinase_name": name, "species": "Human"}).json()
    return d[0]["kinase_ID"] if d else None


def build_dataset():
    """Fetch (and cache) structures with a pocket sequence and a DFG label."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "dfg_dataset.json")
    if os.path.exists(path):
        return json.load(open(path))
    rows = []
    for name in KINASES:
        kid = kinase_id(name)
        if kid is None:
            continue
        for s in _get("structures_list", {"kinase_ID": [kid]}).json():
            dfg, pocket = s.get("DFG", ""), s.get("pocket", "")
            if not dfg or not pocket:
                continue
            if dfg == "in":
                y = 0
            elif "out" in dfg.lower():
                y = 1
            else:
                continue
            rows.append({"kinase": name, "pocket": pocket, "y": y,
                         "structure_ID": s["structure_ID"]})
    json.dump(rows, open(path, "w"))
    return rows


def onehot(seq):
    m = np.zeros((85, 22))
    for i, c in enumerate(seq[:85]):
        m[i, AA_IDX.get(c, 21)] = 1
    return m.flatten()


def sequence_classifier(rows):
    """Ungrouped vs grouped leave-one-kinase-out on pocket sequence."""
    X = np.array([onehot(r["pocket"]) for r in rows])
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["kinase"] for r in rows])
    maj = max(y.mean(), 1 - y.mean())

    def rf():
        return RandomForestClassifier(300, n_jobs=-1, random_state=0)

    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    p_un = cross_val_predict(rf(), X, y, cv=skf)
    pr_un = cross_val_predict(rf(), X, y, cv=skf, method="predict_proba")[:, 1]

    logo = LeaveOneGroupOut()
    p_g = cross_val_predict(rf(), X, y, cv=logo, groups=groups)
    pr_g = cross_val_predict(rf(), X, y, cv=logo, groups=groups,
                             method="predict_proba")[:, 1]

    return {
        "n_structures": len(rows),
        "n_kinases": len(set(groups)),
        "dfg_in": int((y == 0).sum()),
        "dfg_out": int((y == 1).sum()),
        "majority_baseline": round(float(maj), 3),
        "ungrouped_acc": round(accuracy_score(y, p_un), 3),
        "ungrouped_auroc": round(roc_auc_score(y, pr_un), 3),
        "grouped_acc": round(accuracy_score(y, p_g), 3),
        "grouped_auroc": round(roc_auc_score(y, pr_g), 3),
    }


def parse_pocket_ca(mol2_text):
    """pocket position (1..85) -> CA xyz, parsed from a KLIFS pocket mol2."""
    ca, in_atom = {}, False
    for line in mol2_text.splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>") and "ATOM" not in line:
            in_atom = False
        if in_atom and line.strip():
            p = line.split()
            if len(p) >= 7 and p[1] == "CA":
                try:
                    ca[int(p[6])] = np.array([float(p[2]), float(p[3]), float(p[4])])
                except ValueError:
                    pass
    return ca


def dfg_achelix_distance(ca):
    """Centroid distance between DFG motif (pos 80-82) and alphaC (pos 20-30)."""
    dfg = [ca[i] for i in (80, 81, 82) if i in ca]
    ac = [ca[i] for i in range(20, 31) if i in ca]
    if not dfg or not ac:
        return None
    return float(np.linalg.norm(np.mean(dfg, 0) - np.mean(ac, 0)))


def geometric_descriptor(cap_per_kinase=12):
    """Training-free DFG separation from the real DFG-to-alphaC distance."""
    os.makedirs(POCKETS, exist_ok=True)
    ys, ds = [], []
    for name in GEOM_KINASES:
        kid = kinase_id(name)
        sl = _get("structures_list", {"kinase_ID": [kid]}).json()
        ins = [s for s in sl if s.get("DFG") == "in"][:cap_per_kinase]
        outs = [s for s in sl if "out" in str(s.get("DFG", "")).lower()][:cap_per_kinase]
        for s in ins + outs:
            sid = s["structure_ID"]
            fp = os.path.join(POCKETS, f"{sid}.mol2")
            if os.path.exists(fp):
                txt = open(fp).read()
            else:
                txt = _get("structure_get_pocket", {"structure_ID": sid}).text
                open(fp, "w").write(txt)
            d = dfg_achelix_distance(parse_pocket_ca(txt))
            if d is not None:
                ys.append(0 if s["DFG"] == "in" else 1)
                ds.append(d)
    y, d = np.array(ys), np.array(ds)
    return {
        "n_structures": len(y),
        "dfg_in_mean_A": round(float(d[y == 0].mean()), 1),
        "dfg_in_std_A": round(float(d[y == 0].std()), 1),
        "dfg_out_mean_A": round(float(d[y == 1].mean()), 1),
        "dfg_out_std_A": round(float(d[y == 1].std()), 1),
        "auroc_raw_distance": round(roc_auc_score(y, d), 3),
    }


def main():
    print("=" * 60)
    print("KINASE REPRODUCIBILITY AUDIT")
    print("=" * 60)

    rows = build_dataset()
    seq = sequence_classifier(rows)
    print("\n[1] Dataset:", seq["n_structures"], "structures,",
          seq["n_kinases"], "kinases | DFG-in", seq["dfg_in"],
          "DFG-out", seq["dfg_out"], "| majority", seq["majority_baseline"])
    print("\n[2] DFG classifier from pocket SEQUENCE")
    print(f"    Ungrouped 5-fold : acc={seq['ungrouped_acc']}  AUROC={seq['ungrouped_auroc']}  (LEAKS identity)")
    print(f"    Grouped LOKO     : acc={seq['grouped_acc']}  AUROC={seq['grouped_auroc']}  (honest)")

    geom = geometric_descriptor()
    print("\n[3] DFG separation from real GEOMETRIC descriptor (training-free, no leakage)")
    print(f"    DFG-in  : {geom['dfg_in_mean_A']} +/- {geom['dfg_in_std_A']} A")
    print(f"    DFG-out : {geom['dfg_out_mean_A']} +/- {geom['dfg_out_std_A']} A")
    print(f"    AUROC of raw DFG-to-alphaC distance: {geom['auroc_raw_distance']}")

    os.makedirs("checkpoints", exist_ok=True)
    json.dump({"sequence_classifier": seq, "geometric_descriptor": geom},
              open("checkpoints/kinase_audit.json", "w"), indent=2)
    print("\nSaved: checkpoints/kinase_audit.json")


if __name__ == "__main__":
    main()
