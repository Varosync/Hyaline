#!/usr/bin/env python3
"""
Kinase benchmark: the one defensible number, reproducibly.

Reads the geometric descriptors (research/kinase/data/kinase_descriptors.csv)
and evaluates DFG-state classification under grouped leave-one-kinase-out (each
kinase is a fold; no kinase appears in train and test). Writes:

  * checkpoints/kinase_benchmark.json  -- grouped AUROC / accuracy + per-kinase
  * research/kinase/data/splits.csv    -- the LOKO folds with out-of-fold preds

No network; deterministic. Run `python scripts/kinase_descriptors.py` first if the
descriptor CSV does not yet exist.
"""
import csv
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

DESC = "research/kinase/data/kinase_descriptors.csv"
SPLITS = "research/kinase/data/splits.csv"
OUT = "checkpoints/kinase_benchmark.json"


def main():
    if not os.path.exists(DESC):
        raise SystemExit(f"missing {DESC}; run scripts/kinase_descriptors.py first")
    rows = list(csv.DictReader(open(DESC)))
    X = np.array([[float(r["distance"]), float(r["hinge_angle"])] for r in rows])
    y = np.array([int(r["y"]) for r in rows])
    groups = np.array([r["kinase"] for r in rows])

    logo = LeaveOneGroupOut()
    clf = LogisticRegression(max_iter=2000)
    pred = cross_val_predict(clf, X, y, cv=logo, groups=groups)
    prob = cross_val_predict(clf, X, y, cv=logo, groups=groups,
                             method="predict_proba")[:, 1]

    # per-kinase (fold) breakdown
    per = {}
    for k in sorted(set(groups)):
        m = groups == k
        per[k] = {"n": int(m.sum()),
                  "acc": round(accuracy_score(y[m], pred[m]), 3),
                  "auroc": (round(roc_auc_score(y[m], prob[m]), 3)
                            if len(set(y[m])) == 2 else None)}

    result = {
        "protocol": "grouped leave-one-kinase-out (kinase = fold)",
        "n_structures": len(y), "n_kinases": len(set(groups)),
        "grouped_auroc": round(roc_auc_score(y, prob), 3),
        "grouped_accuracy": round(accuracy_score(y, pred), 3),
        "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 3),
        "per_kinase": per,
    }

    os.makedirs("checkpoints", exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)

    with open(SPLITS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["structure_ID", "fold_kinase", "y_true", "distance",
                    "hinge_angle", "y_pred", "prob_dfg_out"])
        for r, p, pr in zip(rows, pred, prob):
            w.writerow([r["structure_ID"], r["kinase"], r["y"], r["distance"],
                        r["hinge_angle"], int(p), round(float(pr), 4)])

    print(f"Grouped leave-one-kinase-out over {result['n_kinases']} kinases, "
          f"{result['n_structures']} structures")
    print(f"  AUROC = {result['grouped_auroc']}   accuracy = {result['grouped_accuracy']}"
          f"   (majority {result['majority_baseline']})")
    print(f"  wrote {OUT} and {SPLITS}")


if __name__ == "__main__":
    main()
