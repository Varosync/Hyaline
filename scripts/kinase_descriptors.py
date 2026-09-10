#!/usr/bin/env python3
"""
Kinase geometric descriptors: compute, evaluate (grouped), and plot.

Parses real Cα coordinates from KLIFS pocket mol2 files and computes two
interpretable, training-free descriptors:

  * DFG-to-alphaC distance   (centroids of pocket positions 80-82 and 20-30)
  * hinge / activation-loop angle at the catalytic lysine (vertex pos 17,
    arms = hinge centroid 46-48 and activation-loop centroid 72-85)

It then (a) evaluates DFG-state classification from the two descriptors under
grouped leave-one-kinase-out (no leakage possible for the raw descriptors), and
(b) writes Figure 1: DFG-to-alphaC distance vs hinge angle, colored by DFG state.

Usage:
    python scripts/kinase_descriptors.py
"""
import csv
import os

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = "https://klifs.net/api_v2"
POCKETS = "klifs_cache/pockets"
OUTDIR = "research/kinase/data"
CAP_PER_CLASS = 30  # per kinase, per DFG state

KINASES = ["ABL1", "EGFR", "BRAF", "SRC", "KIT", "MAPK14", "MET", "KDR",
           "CDK2", "LCK", "MAPK1", "AKT1", "CSF1R", "DDR1", "FGFR1"]

# Okabe-Ito colorblind-safe pair (validated: CVD dE 29.2). DFG-in / DFG-out.
COL_IN, COL_OUT = "#0072B2", "#E69F00"


def _get(endpoint, params):
    return requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)


def kinase_id(name):
    d = _get("kinase_ID", {"kinase_name": name, "species": "Human"}).json()
    return d[0]["kinase_ID"] if d else None


def parse_pocket_ca(mol2_text):
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
    dfg = [ca[i] for i in (80, 81, 82) if i in ca]
    ac = [ca[i] for i in range(20, 31) if i in ca]
    if not dfg or not ac:
        return None
    return float(np.linalg.norm(np.mean(dfg, 0) - np.mean(ac, 0)))


def hinge_activation_angle(ca):
    hinge = [ca[i] for i in (46, 47, 48) if i in ca]
    actloop = [ca[i] for i in range(72, 86) if i in ca]
    lys = ca.get(17)
    if not hinge or not actloop or lys is None:
        return None
    a = np.mean(hinge, 0) - lys
    b = np.mean(actloop, 0) - lys
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def build_descriptors():
    os.makedirs(POCKETS, exist_ok=True)
    recs = []
    for name in KINASES:
        kid = kinase_id(name)
        if kid is None:
            continue
        sl = _get("structures_list", {"kinase_ID": [kid]}).json()
        ins = [s for s in sl if s.get("DFG") == "in"][:CAP_PER_CLASS]
        outs = [s for s in sl if "out" in str(s.get("DFG", "")).lower()][:CAP_PER_CLASS]
        for s in ins + outs:
            sid = s["structure_ID"]
            fp = os.path.join(POCKETS, f"{sid}.mol2")
            if os.path.exists(fp):
                txt = open(fp).read()
            else:
                txt = _get("structure_get_pocket", {"structure_ID": sid}).text
                open(fp, "w").write(txt)
            ca = parse_pocket_ca(txt)
            dist = dfg_achelix_distance(ca)
            ang = hinge_activation_angle(ca)
            if dist is None or ang is None:
                continue
            recs.append({"kinase": name, "structure_ID": sid,
                         "dfg": s["DFG"], "y": 0 if s["DFG"] == "in" else 1,
                         "distance": round(dist, 3), "hinge_angle": round(ang, 3)})
    return recs


def evaluate(recs):
    X = np.array([[r["distance"], r["hinge_angle"]] for r in recs])
    y = np.array([r["y"] for r in recs])
    groups = np.array([r["kinase"] for r in recs])
    logo = LeaveOneGroupOut()
    clf = LogisticRegression(max_iter=2000)
    p = cross_val_predict(clf, X, y, cv=logo, groups=groups)
    pr = cross_val_predict(clf, X, y, cv=logo, groups=groups, method="predict_proba")[:, 1]
    return {
        "n": len(y), "n_kinases": len(set(groups)),
        "grouped_acc": round(accuracy_score(y, p), 3),
        "grouped_auroc": round(roc_auc_score(y, pr), 3),
        "auroc_distance_only": round(roc_auc_score(y, X[:, 0]), 3),
        "auroc_angle_only": round(roc_auc_score(y, -X[:, 1]), 3),
    }


def plot(recs, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.array([r["distance"] for r in recs])
    a = np.array([r["hinge_angle"] for r in recs])
    y = np.array([r["y"] for r in recs])

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=150)
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e7e7e4", linewidth=0.7, zorder=0)
    for lbl, c, mask in [("DFG-in", COL_IN, y == 0), ("DFG-out", COL_OUT, y == 1)]:
        ax.scatter(d[mask], a[mask], s=42, c=c, edgecolors="white",
                   linewidths=0.6, alpha=0.9, label=f"{lbl} (n={int(mask.sum())})",
                   zorder=3)
    ax.set_xlabel("DFG-to-αC-helix distance (Å)")
    ax.set_ylabel("Hinge / activation-loop angle (°)")
    ax.set_title("Kinase conformation from interpretable geometric descriptors",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(frameon=False, loc="best")
    ax.text(0.98, 0.02,
            f"grouped LOKO AUROC = {metrics['grouped_auroc']}  (no leakage)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#555")
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "figure1_descriptors.png")
    fig.savefig(out, bbox_inches="tight")
    return out


def main():
    print("Building geometric descriptors from real KLIFS coordinates...")
    recs = build_descriptors()

    os.makedirs(OUTDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, "kinase_descriptors.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kinase", "structure_ID", "dfg", "y",
                                          "distance", "hinge_angle"])
        w.writeheader()
        w.writerows(recs)

    m = evaluate(recs)
    print(f"\nStructures: {m['n']}  |  kinases: {m['n_kinases']}")
    print(f"Single descriptor AUROC  : distance {m['auroc_distance_only']} · angle {m['auroc_angle_only']}")
    print(f"Both descriptors, grouped leave-one-kinase-out:")
    print(f"    accuracy = {m['grouped_acc']}   AUROC = {m['grouped_auroc']}")
    out = plot(recs, m)
    print(f"\nWrote: {csv_path}\nWrote: {out}")


if __name__ == "__main__":
    main()
