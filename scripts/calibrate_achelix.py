#!/usr/bin/env python3
"""
Calibrate the alphaC-helix (aC-in / aC-out) model.
==================================================

The aC-helix state is defined by the beta3-Lys -- aC-Glu salt bridge: in the
active "aC-in" state the conserved Lys (KLIFS pocket position 17) and Glu
(position 24) are close; in "aC-out" the helix swings away and they separate.

With only Calpha coordinates we use the Lys17--Glu24 Calpha distance as the
descriptor and fit a logistic model to the KLIFS aC_helix annotation. The
honest, defensible number is the grouped leave-one-kinase-out AUROC (no kinase
in both train and test). The final model is then fit on all labelled structures
and written to hyaline/kinase/achelix_model.json (mirrors dfg_model.json).

Fully offline: reads cached pockets (klifs_cache/pockets/*.mol2) and cached
per-structure KLIFS metadata (klifs_cache/atlas_structures.json).

Usage:
    python scripts/calibrate_achelix.py
"""
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

POCKETS = "klifs_cache/pockets"
ATLAS_CACHE = "klifs_cache/atlas_structures.json"
MODEL_OUT = "hyaline/kinase/achelix_model.json"

# KLIFS pocket positions of the conserved salt-bridge pair.
LYS_POS, GLU_POS = 17, 24


def parse_pocket_ca(txt):
    ca, in_atom = {}, False
    for line in txt.splitlines():
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


def ke_distance(ca):
    if LYS_POS in ca and GLU_POS in ca:
        return float(np.linalg.norm(ca[LYS_POS] - ca[GLU_POS]))
    return None


def load_labels():
    raw = json.load(open(ATLAS_CACHE))
    sid2 = {}
    for _kid, entry in raw.items():
        name = entry["meta"].get("name")
        for s in entry["structures"]:
            sid2[str(s["structure_ID"])] = (s.get("aC_helix"), name)
    return sid2


def main():
    sid2 = load_labels()
    X, y, groups = [], [], []
    for fp in glob.glob(os.path.join(POCKETS, "*.mol2")):
        sid = os.path.splitext(os.path.basename(fp))[0]
        if sid not in sid2:
            continue
        ac, name = sid2[sid]
        if ac not in ("in", "out"):
            continue
        d = ke_distance(parse_pocket_ca(open(fp).read()))
        if d is None:
            continue
        X.append([d])
        y.append(1 if ac == "out" else 0)
        groups.append(name)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    n_out = int(y.sum())
    print(f"Labelled structures: {len(y)}  (aC-out={n_out}, aC-in={len(y) - n_out}) "
          f"across {len(set(groups))} kinases")

    logo = LeaveOneGroupOut()
    clf = LogisticRegression(max_iter=2000)
    pr = cross_val_predict(clf, X, y, cv=logo, groups=groups, method="predict_proba")[:, 1]
    pd = cross_val_predict(clf, X, y, cv=logo, groups=groups)
    auroc = roc_auc_score(y, pr)
    acc = accuracy_score(y, pd)
    print(f"Grouped leave-one-kinase-out: AUROC {auroc:.3f}  acc {acc:.3f}")

    # Final model on all labelled data (deployment model, like dfg_model.json).
    clf.fit(X, y)
    model = {
        "features": ["achelix_ke_distance_A"],
        "weights": [float(clf.coef_[0][0])],
        "intercept": float(clf.intercept_[0]),
        "positive_class": "aC-out",
        "n_train": int(len(y)),
        "note": (f"logistic on the beta3-Lys(17)--aC-Glu(24) Calpha distance; "
                 f"grouped leave-one-kinase-out AUROC {auroc:.3f}"),
        "grouped_loko_auroc": round(float(auroc), 3),
    }
    with open(MODEL_OUT, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Wrote {MODEL_OUT}")


if __name__ == "__main__":
    main()
