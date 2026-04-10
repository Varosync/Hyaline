#!/usr/bin/env python3
"""
Train Deep Model v5 — Real ChEMBL pKi + DFG Classification + Clean Splits
"""

import sys, json, argparse, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, mean_absolute_error, balanced_accuracy_score
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyaline.models.kinase_binding import KinaseBindingPredictor, KinaseBindingConfig, KLIFSLoader
from hyaline.models.conformational_prior import encode_pocket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_knn_edges(coords, k=10):
    nonzero = np.any(np.abs(coords) > 1e-6, axis=1)
    vi = np.where(nonzero)[0]
    if len(vi) < 3:
        return torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    vc = coords[vi]
    ek = min(k, len(vi) - 1)
    tree = cKDTree(vc)
    _, nn = tree.query(vc, k=ek + 1)
    r, c = [], []
    for i in range(len(vi)):
        for j in range(1, ek + 1):
            r.append(vi[i])
            c.append(vi[nn[i, j]])
    return torch.tensor([r, c], dtype=torch.long)


class DeepDataset(Dataset):
    def __init__(self, df, pocket_cache):
        self.df = df.reset_index(drop=True)
        self.seqs, self.coords, self.edges = [], [], []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Graphs", leave=False):
            sid = row["structure_id"]
            if sid in pocket_cache:
                seq, co = pocket_cache[sid]
                self.seqs.append(torch.tensor(encode_pocket(seq), dtype=torch.long))
                self.coords.append(torch.tensor(co, dtype=torch.float32))
                self.edges.append(build_knn_edges(co))
            else:
                self.seqs.append(torch.zeros(85, dtype=torch.long))
                self.coords.append(torch.zeros(85, 3))
                self.edges.append(torch.tensor([[0, 1], [1, 0]], dtype=torch.long))

        self.fps = torch.zeros(len(df), 2048)
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            n = 0
            for i, (_, row) in enumerate(self.df.iterrows()):
                smi = row.get("ligand_smiles", None)
                if pd.notna(smi):
                    mol = Chem.MolFromSmiles(str(smi))
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        self.fps[i] = torch.tensor(list(fp), dtype=torch.float32)
                        n += 1
            logger.info("Parsed %d/%d Morgan FPs" % (n, len(df)))
        except ImportError:
            pass

        self.dfg_labels = torch.tensor(
            df["dfg"].isin(["out", "out-like"]).astype(int).values, dtype=torch.float32
        )
        self.has_real_pki = torch.tensor(df["pki_chembl"].notna().values, dtype=torch.bool)
        self.real_pki = torch.tensor(df["pki_chembl"].fillna(0).values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "seq": self.seqs[idx],
            "coords": self.coords[idx],
            "edges": self.edges[idx],
            "fp": self.fps[idx],
            "dfg": self.dfg_labels[idx],
            "has_pki": self.has_real_pki[idx],
            "pki": self.real_pki[idx],
        }


def collate(batch):
    B, N = len(batch), 85
    el = []
    for i, b in enumerate(batch):
        ei = b["edges"]
        if ei.size(1) > 0:
            el.append(ei + i * N)
        else:
            el.append(torch.tensor([[0, 1], [1, 0]], dtype=torch.long) + i * N)
    return {
        "seq": torch.stack([b["seq"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "edges": torch.cat(el, dim=1),
        "batch_idx": torch.arange(B).repeat_interleave(N),
        "fp": torch.stack([b["fp"] for b in batch]),
        "dfg": torch.stack([b["dfg"] for b in batch]),
        "has_pki": torch.stack([b["has_pki"] for b in batch]),
        "pki": torch.stack([b["pki"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Deep Model v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    sep = "=" * 70
    print("\n" + sep)
    print("  DEEP MODEL v5 -- Real ChEMBL pKi + Clean Splits")
    print(sep)
    print("  Device:", device)

    # Load clean splits
    train_df = pd.read_csv("data/splits/seed_42_train.csv")
    val_df = pd.read_csv("data/splits/seed_42_val.csv")

    # Merge real pKi
    real_pki = pd.read_csv("data/klifs_with_real_pki.csv")[["structure_id", "pki_chembl"]]
    train_df = train_df.merge(real_pki, on="structure_id", how="left")
    val_df = val_df.merge(real_pki, on="structure_id", how="left")

    train_df = train_df[train_df["dfg"].isin(["in", "out", "out-like"])].copy()
    val_df = val_df[val_df["dfg"].isin(["in", "out", "out-like"])].copy()

    n_train_pki = int(train_df["pki_chembl"].notna().sum())
    n_val_pki = int(val_df["pki_chembl"].notna().sum())
    print("  Train: %d (%d with real pKi)" % (len(train_df), n_train_pki))
    print("  Val: %d (%d with real pKi)" % (len(val_df), n_val_pki))

    # Load pocket cache
    loader = KLIFSLoader()
    pocket_cache = {}
    all_sids = set(train_df["structure_id"].tolist() + val_df["structure_id"].tolist())
    for sid in tqdm(all_sids, desc="Loading pockets"):
        co = loader.get_pocket_coordinates(sid)
        seq = loader.get_pocket_sequence_from_mol2(sid)
        if co is not None and seq is not None:
            pocket_cache[sid] = (seq, co)
    logger.info("Loaded %d pockets" % len(pocket_cache))

    train_df = train_df[train_df["structure_id"].isin(pocket_cache)].copy()
    val_df = val_df[val_df["structure_id"].isin(pocket_cache)].copy()

    train_ds = DeepDataset(train_df, pocket_cache)
    val_ds = DeepDataset(val_df, pocket_cache)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate,
    )

    config = KinaseBindingConfig(
        node_dim=64, hidden_dim=128, num_egnn_layers=4,
        fingerprint_dim=256, dropout=0.1,
    )
    model = KinaseBindingPredictor(config).to(device)
    dfg_head = nn.Sequential(
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in dfg_head.parameters())
    print("  Params: %s" % f"{total_params:,}")

    n_pos = int(train_df["dfg"].isin(["out", "out-like"]).sum())
    dfg_pw = torch.tensor(
        [(len(train_df) - n_pos) / max(n_pos, 1)], dtype=torch.float32
    ).to(device)
    dfg_crit = nn.BCEWithLogitsLoss(pos_weight=dfg_pw)
    pki_crit = nn.MSELoss()

    all_params_list = list(model.parameters()) + list(dfg_head.parameters())
    opt = torch.optim.AdamW(all_params_list, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    ckpt_dir = Path("checkpoints/deep_model")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Threshold annealing
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        thresh = 0.3 + progress * 0.7
        model.pocket_encoder.spiking_egnn.set_threshold(thresh)

        # Train
        model.train()
        dfg_head.train()
        loss_sum, n_samples = 0.0, 0

        for batch in tqdm(train_loader, desc="Epoch %d" % epoch, leave=False):
            seq = batch["seq"].to(device)
            coords = batch["coords"].to(device)
            conf = torch.zeros(seq.size(0), 4, device=device)  # ZEROED
            fp = batch["fp"].to(device)
            dfg_true = batch["dfg"].to(device)
            has_pki = batch["has_pki"].to(device)
            pki_true = batch["pki"].to(device)
            edges = batch["edges"].to(device)
            bi = batch["batch_idx"].to(device)

            opt.zero_grad()
            out = model(seq, coords, conf, fp, edges, bi)

            dfg_logits = dfg_head(out["pocket_embedding"]).squeeze(-1)
            loss_dfg = dfg_crit(dfg_logits, dfg_true)

            loss_pki = torch.tensor(0.0, device=device)
            if has_pki.any():
                loss_pki = pki_crit(out["pki"][has_pki], pki_true[has_pki])

            loss = loss_dfg + 0.5 * loss_pki
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params_list, 1.0)
            opt.step()

            bs = seq.size(0)
            loss_sum += loss.item() * bs
            n_samples += bs

        sched.step()

        # Validate
        model.eval()
        dfg_head.eval()
        v_pki, v_pki_true, v_dfg_scores, v_dfg_labels, v_syncs = [], [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                coords = batch["coords"].to(device)
                conf = torch.zeros(seq.size(0), 4, device=device)
                fp = batch["fp"].to(device)
                edges = batch["edges"].to(device)
                bi = batch["batch_idx"].to(device)

                out = model(seq, coords, conf, fp, edges, bi)
                dl = dfg_head(out["pocket_embedding"]).squeeze(-1)

                v_pki.extend(out["pki"].cpu().numpy())
                v_pki_true.extend(batch["pki"].numpy())
                v_dfg_scores.extend(torch.sigmoid(dl).cpu().numpy())
                v_dfg_labels.extend(batch["dfg"].numpy())
                v_syncs.append(out["sync_score"].mean().item())

        try:
            auroc = roc_auc_score(v_dfg_labels, v_dfg_scores)
        except ValueError:
            auroc = 0.5
        bal = balanced_accuracy_score(v_dfg_labels, [1 if s > 0.5 else 0 for s in v_dfg_scores])

        val_has_pki = val_df["pki_chembl"].notna().values
        if val_has_pki.any():
            vp = np.array(v_pki)
            vt = np.array(v_pki_true)
            pki_mae = mean_absolute_error(vt[val_has_pki], vp[val_has_pki])
        else:
            pki_mae = float("nan")

        sync_mean = float(np.mean(v_syncs))

        rec = {
            "epoch": epoch,
            "loss": float(loss_sum / max(n_samples, 1)),
            "dfg_auroc": float(auroc),
            "dfg_bal": float(bal),
            "pki_mae": float(pki_mae),
            "sync": sync_mean,
            "thresh": float(thresh),
        }
        history.append(rec)

        if epoch % 5 == 0 or epoch <= 3:
            print("  Epoch %d: loss=%.4f, dfg_auroc=%.3f, dfg_bal=%.3f, pki_mae=%.3f, sync=%.3f" % (
                epoch, rec["loss"], auroc, bal, pki_mae, sync_mean))

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save({
                "model": model.state_dict(),
                "dfg_head": dfg_head.state_dict(),
                "epoch": epoch,
                "val_auroc": auroc,
                "val_pki_mae": pki_mae,
                "config": config.__dict__,
                "seed": args.seed,
            }, ckpt_dir / "best_model.pt")
            if epoch % 5 == 0 or epoch <= 3:
                print("    -> New best (auroc=%.3f, pki_mae=%.3f)" % (auroc, pki_mae))

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Structure sensitivity test
    print("\n  Structure Sensitivity (real DFG pairs):")
    model.eval()
    dfg_head.eval()
    model.pocket_encoder.spiking_egnn.set_threshold(1.0)

    both = train_df.groupby("kinase_name").agg(
        has_in=pd.NamedAgg(column="dfg", aggfunc=lambda x: (x == "in").any()),
        has_out=pd.NamedAgg(column="dfg", aggfunc=lambda x: x.isin(["out", "out-like"]).any()),
    ).reset_index()
    both_kinases = both[(both.has_in) & (both.has_out)]["kinase_name"].tolist()[:10]

    deltas = []
    for kin in both_kinases:
        kdf = train_df[train_df["kinase_name"] == kin]
        in_r = kdf[kdf["dfg"] == "in"].head(1)
        out_r = kdf[kdf["dfg"].isin(["out", "out-like"])].head(1)
        if len(in_r) == 0 or len(out_r) == 0:
            continue
        ci = loader.get_pocket_coordinates(in_r.iloc[0]["structure_id"])
        co_c = loader.get_pocket_coordinates(out_r.iloc[0]["structure_id"])
        si = loader.get_pocket_sequence_from_mol2(in_r.iloc[0]["structure_id"]) or "-" * 85
        if ci is None or co_c is None:
            continue
        with torch.no_grad():
            si_t = torch.tensor(encode_pocket(si), dtype=torch.long).unsqueeze(0).to(device)
            ci_t = torch.tensor(ci, dtype=torch.float32).unsqueeze(0).to(device)
            co_t = torch.tensor(co_c, dtype=torch.float32).unsqueeze(0).to(device)
            z = torch.zeros(1, 4, device=device)
            fp = torch.zeros(1, 2048, device=device)
            ei_i = build_knn_edges(ci).to(device)
            ei_o = build_knn_edges(co_c).to(device)
            bi = torch.zeros(85, dtype=torch.long, device=device)
            o1 = model(si_t, ci_t, z, fp, ei_i, bi)
            o2 = model(si_t, co_t, z, fp, ei_o, bi)
            d1 = torch.sigmoid(dfg_head(o1["pocket_embedding"])).item()
            d2 = torch.sigmoid(dfg_head(o2["pocket_embedding"])).item()
        dd = abs(d1 - d2)
        deltas.append(dd)
        print("  %-15s DFG diff=%.3f (in=%.3f, out=%.3f)" % (kin, dd, d1, d2))

    if deltas:
        print("  Mean DFG diff: %.3f (pass: %s)" % (np.mean(deltas), np.mean(deltas) > 0.1))

    best = max(history, key=lambda r: r["dfg_auroc"])
    print("\n" + sep)
    print("  BEST: epoch %d, DFG auroc=%.3f, pKi MAE=%.3f" % (best["epoch"], best["dfg_auroc"], best["pki_mae"]))
    print(sep)

    try:
        import subprocess
        subprocess.run([
            "aws", "s3", "cp", str(ckpt_dir / "best_model.pt"),
            "s3://hyaline-kinase-data/models/deep_model/best_model_v5.pt",
        ], check=True, capture_output=True)
        print("  Uploaded to S3")
    except Exception as e:
        logger.warning("S3 upload failed: %s" % e)


if __name__ == "__main__":
    main()
