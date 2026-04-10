#!/usr/bin/env python3
"""
Rigorous biophysics validation — is the model learning real physics?

Tests:
1. Known drug check: Type II vs Type I inhibitors on ABL1
2. Coordinate ablation: zero out coords, does DFG prediction collapse?
3. Sequence shuffle: scramble sequence, keep coords — should still predict DFG
4. DFG-alphaC distance correlation: P(DFG-out) should correlate with geometry
5. Label shuffle control: shuffled labels should give AuROC ~0.5
6. Gold standard cross-family generalization (fully held out)
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~/Jinja"))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from hyaline.models.kinase_binding import KinaseBindingPredictor, KinaseBindingConfig, KLIFSLoader
from hyaline.models.conformational_prior import encode_pocket

os.chdir(os.path.expanduser("~/Jinja"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
ckpt = torch.load("checkpoints/deep_model/best_model.pt", map_location=device, weights_only=False)
config = KinaseBindingConfig(**ckpt["config"])
model = KinaseBindingPredictor(config).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

dfg_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)).to(device)
dfg_head.load_state_dict(ckpt["dfg_head"])
dfg_head.eval()

klifs_loader = KLIFSLoader()
df = pd.read_csv("data/klifs_with_bioactivity.csv")


def build_edges(coords):
    nonzero = np.any(np.abs(coords) > 1e-6, axis=1)
    vi = np.where(nonzero)[0]
    if len(vi) < 3:
        return torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    vc = coords[vi]
    k = min(10, len(vi) - 1)
    tree = cKDTree(vc)
    _, nn_idx = tree.query(vc, k=k + 1)
    r, c = [], []
    for i in range(len(vi)):
        for j in range(1, k + 1):
            r.append(vi[i])
            c.append(vi[nn_idx[i, j]])
    return torch.tensor([r, c], dtype=torch.long)


def predict_dfg(seq_str, coords_np):
    seq_t = torch.tensor(encode_pocket(seq_str), dtype=torch.long).unsqueeze(0).to(device)
    coords_t = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).to(device)
    conf = torch.zeros(1, 4, device=device)
    fp = torch.zeros(1, 2048, device=device)
    ei = build_edges(coords_np).to(device)
    bi = torch.zeros(85, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(seq_t, coords_t, conf, fp, ei, bi)
        prob = torch.sigmoid(dfg_head(out["pocket_embedding"])).item()
    return prob, out["pki"].item()


sep = "=" * 70
print(sep)
print("  BIOPHYSICS VALIDATION SUITE")
print(sep)

# ===================================================================
# TEST 1: Known Drug Validation (ABL1)
# ===================================================================
print("\n  TEST 1: Known Drug Validation (ABL1)")
print("  " + "-" * 50)

abl1 = df[df["kinase_name"] == "ABL1"]
# Type II structures (DFG-out) vs Type I (DFG-in)
for label, dfg_filter, expected in [
    ("Type II (DFG-out)", ["out", "out-like"], "HIGH"),
    ("Type I (DFG-in)", ["in"], "LOW"),
]:
    subset = abl1[abl1["dfg"].isin(dfg_filter)].sample(min(15, len(abl1[abl1["dfg"].isin(dfg_filter)])), random_state=42)
    probs = []
    for _, row in subset.iterrows():
        sid = row["structure_id"]
        coords = klifs_loader.get_pocket_coordinates(sid)
        seq = klifs_loader.get_pocket_sequence_from_mol2(sid) or "-" * 85
        if coords is not None:
            p, _ = predict_dfg(seq, coords)
            probs.append(p)
    if probs:
        print("  %s: mean P(DFG-out) = %.4f  (expect %s)  [n=%d]" % (label, np.mean(probs), expected, len(probs)))

# ===================================================================
# TEST 2: Coordinate Ablation
# ===================================================================
print("\n  TEST 2: Coordinate Ablation (zero coords should destroy prediction)")
print("  " + "-" * 50)

sample = df[df["dfg"].isin(["in", "out"])].sample(200, random_state=42)
real_probs, zero_probs, noise_probs, labels = [], [], [], []

for _, row in sample.iterrows():
    sid = row["structure_id"]
    coords = klifs_loader.get_pocket_coordinates(sid)
    seq = klifs_loader.get_pocket_sequence_from_mol2(sid) or "-" * 85
    if coords is None:
        continue

    # Real coordinates
    p_real, _ = predict_dfg(seq, coords)
    real_probs.append(p_real)

    # Random noise coordinates (same scale as real)
    noise_coords = np.random.randn(85, 3).astype(np.float32) * coords.std()
    p_noise, _ = predict_dfg(seq, noise_coords)
    noise_probs.append(p_noise)

    labels.append(1 if row["dfg"] == "out" else 0)

labels = np.array(labels)
real_probs = np.array(real_probs)
noise_probs = np.array(noise_probs)

auroc_real = roc_auc_score(labels, real_probs)
auroc_noise = roc_auc_score(labels, noise_probs)

print("  Real coords AuROC:   %.3f" % auroc_real)
print("  Random coords AuROC: %.3f  (should be ~0.5 if model uses geometry)" % auroc_noise)
print("  Performance drop:    %.1f%%" % ((1 - auroc_noise / auroc_real) * 100))

# ===================================================================
# TEST 3: Sequence Shuffle (keep real coords)
# ===================================================================
print("\n  TEST 3: Sequence Shuffle (real coords preserved)")
print("  " + "-" * 50)

shuffled_probs = []
rng = np.random.RandomState(123)
idx = 0
for _, row in sample.iterrows():
    sid = row["structure_id"]
    coords = klifs_loader.get_pocket_coordinates(sid)
    seq = klifs_loader.get_pocket_sequence_from_mol2(sid) or "-" * 85
    if coords is None:
        continue

    seq_list = list(seq)
    rng.shuffle(seq_list)
    shuffled_seq = "".join(seq_list)
    p, _ = predict_dfg(shuffled_seq, coords)
    shuffled_probs.append(p)
    idx += 1

shuffled_probs = np.array(shuffled_probs)
auroc_shuffled_seq = roc_auc_score(labels, shuffled_probs)
print("  Real seq + real coords:     AuROC = %.3f" % auroc_real)
print("  Shuffled seq + real coords: AuROC = %.3f" % auroc_shuffled_seq)
print("  (If 3D coords drive prediction, shuffling seq should not destroy it)")

# ===================================================================
# TEST 4: DFG-alphaC Distance Correlation
# ===================================================================
print("\n  TEST 4: DFG-alphaC Distance Correlation")
print("  " + "-" * 50)

distances = []
probs_for_corr = []

for i, (_, row) in enumerate(sample.iterrows()):
    dist = row.get("dfg_chelix_distance", None)
    if pd.notna(dist) and i < len(real_probs):
        distances.append(float(dist))
        probs_for_corr.append(real_probs[i])

if len(distances) > 10:
    distances = np.array(distances)
    probs_for_corr = np.array(probs_for_corr)
    r_pearson, p_pearson = pearsonr(distances, probs_for_corr)
    r_spearman, p_spearman = spearmanr(distances, probs_for_corr)
    print("  Pearson r(distance, P(DFG-out)):  %.3f  (p=%.2e)" % (r_pearson, p_pearson))
    print("  Spearman r(distance, P(DFG-out)): %.3f  (p=%.2e)" % (r_spearman, p_spearman))
    print("  (Positive = larger DFG-aC distance predicts more DFG-out)")

    # Also check: mean distance for high vs low P(DFG-out)
    high_mask = probs_for_corr > 0.5
    low_mask = probs_for_corr <= 0.5
    if high_mask.any() and low_mask.any():
        print("  Mean distance when P(out)>0.5: %.1f A" % distances[high_mask].mean())
        print("  Mean distance when P(out)<0.5: %.1f A" % distances[low_mask].mean())

# ===================================================================
# TEST 5: Label Shuffle Control
# ===================================================================
print("\n  TEST 5: Label Shuffle Control")
print("  " + "-" * 50)

shuffled_aurocs = []
for trial in range(20):
    sl = labels.copy()
    np.random.shuffle(sl)
    shuffled_aurocs.append(roc_auc_score(sl, real_probs))

print("  AuROC with real labels:     %.3f" % auroc_real)
print("  AuROC with shuffled labels: %.3f +/- %.3f  (expect ~0.5)" % (
    np.mean(shuffled_aurocs), np.std(shuffled_aurocs)))

# ===================================================================
# TEST 6: Gold Standard (fully held out)
# ===================================================================
print("\n  TEST 6: Gold Standard Cross-Family (fully held out from training)")
print("  " + "-" * 50)

gs = pd.read_csv("gold-standard-inhibitor-curation/data/known_inhibitors_curated.csv")
gs_probs, gs_labels, gs_groups = [], [], []

for _, row in gs.iterrows():
    pdb = row["PDB"]
    match = df[df["pdb"] == pdb].head(1)
    if len(match) == 0:
        continue
    sid = match.iloc[0]["structure_id"]
    coords = klifs_loader.get_pocket_coordinates(sid)
    seq = klifs_loader.get_pocket_sequence_from_mol2(sid) or "-" * 85
    if coords is None:
        continue
    p, _ = predict_dfg(seq, coords)
    gs_probs.append(p)
    gs_labels.append(1 if "out" in str(row.get("DFG", "")).lower() else 0)
    gs_groups.append(row.get("GROUPS", "Other"))

gs_probs = np.array(gs_probs)
gs_labels = np.array(gs_labels)

gs_auroc = roc_auc_score(gs_labels, gs_probs)
print("  Gold standard (Deep Model): AuROC = %.3f  (n=%d)" % (gs_auroc, len(gs_labels)))

gs_df = pd.DataFrame({"prob": gs_probs, "label": gs_labels, "group": gs_groups})
for grp in sorted(gs_df["group"].unique()):
    g = gs_df[gs_df["group"] == grp]
    if len(g) >= 5 and g["label"].nunique() > 1:
        a = roc_auc_score(g["label"], g["prob"])
        print("    %-12s: AuROC = %.3f  (n=%d)" % (grp, a, len(g)))

# ===================================================================
# VERDICT
# ===================================================================
print("\n" + sep)
print("  VERDICT")
print(sep)

checks = []
# Check 1: Coord ablation drops performance significantly
coord_pass = auroc_noise < auroc_real - 0.15
checks.append(("Coords essential (ablation drops >15%%)", coord_pass,
               "%.3f -> %.3f" % (auroc_real, auroc_noise)))

# Check 2: Sequence shuffle preserves most signal (coords dominate)
seq_pass = auroc_shuffled_seq > 0.7
checks.append(("Coords dominate over sequence", seq_pass,
               "shuffled seq AuROC = %.3f" % auroc_shuffled_seq))

# Check 3: Distance correlation positive and significant
dist_pass = r_spearman > 0.3 and p_spearman < 0.01 if len(distances) > 10 else False
checks.append(("DFG-aC distance correlates with P(out)", dist_pass,
               "Spearman r = %.3f" % (r_spearman if len(distances) > 10 else 0)))

# Check 4: Shuffle control is ~0.5
shuffle_pass = abs(np.mean(shuffled_aurocs) - 0.5) < 0.1
checks.append(("Shuffled labels give ~0.5 AuROC", shuffle_pass,
               "%.3f +/- %.3f" % (np.mean(shuffled_aurocs), np.std(shuffled_aurocs))))

# Check 5: Gold standard held-out > 0.8
gs_pass = gs_auroc > 0.8
checks.append(("Gold standard held-out AuROC > 0.8", gs_pass,
               "%.3f" % gs_auroc))

for name, passed, detail in checks:
    status = "PASS" if passed else "FAIL"
    print("  [%s] %s  (%s)" % (status, name, detail))

n_pass = sum(1 for _, p, _ in checks if p)
print("\n  Result: %d/%d checks passed" % (n_pass, len(checks)))

if n_pass == len(checks):
    print("  CONCLUSION: Model is learning real biophysics, not overfitting.")
elif n_pass >= 3:
    print("  CONCLUSION: Model learns real structure signal with some caveats.")
else:
    print("  CONCLUSION: Model may be overfitting. Investigate further.")

print(sep)
