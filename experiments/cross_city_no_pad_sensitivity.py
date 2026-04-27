"""
cross_city_no_pad_sensitivity.py
=================================
Reviewer Q2 sensitivity check: how much does the index-repetition padding in
experiments/stgnn_shift_eval_v5_pems04.py (lines 87-91) bias cross-city MAE
when target N < source N?

For PEMS04 (307) → METR-LA (207): the existing eval pads `keep` with 100
copies of index 0. We re-evaluate the same PEMS04-trained checkpoints with
the same 5 subsets, but compute MAE only on the 207 unique target nodes,
discarding predictions on the duplicate slots. We compare this no-pad MAE
with the existing padded MAE for the same subset.

For PEMS04 (307) → PEMS-BAY (325): target > source, no padding kicks in;
included as a control.

Output: results/polish_v15/cross_city_no_pad_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.stgnn_backbone_bank import (
    BACKBONE_REGISTRY,
    _windows,
    load_adjacency,
    load_pems04,
    make_dataloader,
)


@torch.no_grad()
def evaluate_per_node(model, loader, adj, device):
    """Return predictions and targets concatenated, shape (B_total, N, pred_len)."""
    model.eval()
    preds, tgts = [], []
    for X_b, Y_b in loader:
        X_b = X_b.to(device)
        pred = model(X_b, adj).cpu()              # (B, N, pred_len)
        preds.append(pred)
        tgts.append(Y_b.permute(0, 2, 1))         # (B, N, pred_len)
    return torch.cat(preds, 0), torch.cat(tgts, 0)


def mae_on_indices(pred, tgt, indices, stats):
    """Denormalize using stats (per-node mean/std), then MAE over `indices`."""
    mean = torch.as_tensor(stats["mean"]).squeeze().float()    # (N,)
    std = torch.as_tensor(stats["std"]).squeeze().float()
    p = pred * std.unsqueeze(0).unsqueeze(-1) + mean.unsqueeze(0).unsqueeze(-1)
    t = tgt * std.unsqueeze(0).unsqueeze(-1) + mean.unsqueeze(0).unsqueeze(-1)
    p_i = p[:, indices, :]
    t_i = t[:, indices, :]
    return float((p_i - t_i).abs().mean())


def build_target_subsets_extended(target_key: str, target_dir: str, adj_path: str,
                                  source_num_nodes: int, source_stats: dict,
                                  n_subsets: int = 5, base_seed: int = 42,
                                  seq_len: int = 12, pred_len: int = 12):
    """Same as stgnn_shift_eval_v5_pems04.build_target_subsets, but also returns
    `unique_idx`: the positions in `keep` corresponding to unique (non-duplicate)
    target nodes. For target N >= source: unique_idx = range(source_num_nodes)
    (all positions are unique). For target N < source: only the first N_target
    positions are unique; the rest are index-0 duplicates.
    """
    from experiments.stgnn_backbone_bank import _load_raw
    prefix_map = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY"}
    if target_key not in prefix_map:
        raise ValueError(target_key)
    raw_full = _load_raw(target_dir, prefix=prefix_map[target_key])

    T, N_target = raw_full.shape
    train_end = int(T * 0.7)
    test_start = int(T * 0.8)
    train_raw = raw_full[:train_end]
    test_raw = raw_full[test_start:]

    target_mean = train_raw.mean(axis=0, keepdims=True)
    target_std = train_raw.std(axis=0, keepdims=True) + 1e-8

    def windows_with(arr, s_mean, s_std):
        norm = (arr - s_mean) / s_std
        X, Y = _windows(norm.astype(np.float32), seq_len, pred_len)
        return X.astype(np.float32), Y.astype(np.float32)

    with open(adj_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    adj_full = obj[2] if (isinstance(obj, (tuple, list)) and len(obj) == 3) else obj
    adj_full = np.asarray(adj_full).astype(np.float32)

    subsets = []
    for i in range(n_subsets):
        rng = np.random.RandomState(base_seed + i)
        if N_target >= source_num_nodes:
            keep = sorted(rng.choice(N_target, source_num_nodes, replace=False).tolist())
            unique_idx = list(range(source_num_nodes))
        else:
            keep = list(range(N_target)) + [0] * (source_num_nodes - N_target)
            unique_idx = list(range(N_target))   # only the first N_target slots are unique

        t_mean = target_mean[:, keep]
        t_std = target_std[:, keep]
        X_tgt, Y_tgt = windows_with(test_raw[:, keep], t_mean, t_std)

        adj_sub_np = adj_full[np.ix_(keep, keep)]
        adj_t = torch.FloatTensor(adj_sub_np)
        deg = adj_t.sum(dim=1).clamp(min=1e-8)
        D_inv_sqrt = deg.pow(-0.5)
        adj_norm = D_inv_sqrt.unsqueeze(1) * adj_t * D_inv_sqrt.unsqueeze(0)

        subsets.append({
            "subset_id": i,
            "keep_idx": keep,
            "unique_idx": unique_idx,
            "n_unique": len(unique_idx),
            "X_tgt": X_tgt, "Y_tgt": Y_tgt,
            "adj_norm": adj_norm,
            "tgt_stats": {"mean": t_mean, "std": t_std},
        })
    return subsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="baselines/stgnn_protocol_registry.json")
    ap.add_argument("--backbones", nargs="+", default=["dcrnn", "gwnet", "gru", "mtgnn", "stgcn"])
    ap.add_argument("--source_data_dir", default="data/pems04")
    ap.add_argument("--source_adj_path", default="data/pems04/adj_mx.pkl")
    ap.add_argument("--metr_la_dir", default="data/metr-la")
    ap.add_argument("--metr_la_adj", default="data/metr-la/adj_mx.pkl")
    ap.add_argument("--pems_bay_dir", default="data/pems-bay")
    ap.add_argument("--pems_bay_adj", default="data/pems-bay/adj_mx_bay.pkl")
    ap.add_argument("--output_dir", default="results/polish_v15")
    ap.add_argument("--n_subsets", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load PEMS04 source stats
    print("Loading PEMS04 source data + stats...")
    (_, _), (_, _), (X_te_src, Y_te_src), source_stats = load_pems04(args.source_data_dir)
    source_num_nodes = X_te_src.shape[2]
    print(f"  PEMS04 nodes={source_num_nodes}")

    adj_src = load_adjacency(args.source_adj_path, source_num_nodes, device)

    with open(args.registry) as f:
        registry = json.load(f)

    print(f"\nBuilding {args.n_subsets} METR-LA subsets (target<source: padded)...")
    metr_subs = build_target_subsets_extended(
        "metr-la", args.metr_la_dir, args.metr_la_adj,
        source_num_nodes, source_stats, n_subsets=args.n_subsets)
    print(f"  metr-la N_target={metr_subs[0]['n_unique']}, padded to {len(metr_subs[0]['keep_idx'])}")

    print(f"Building {args.n_subsets} PEMS-BAY subsets (target>source: subsample only)...")
    bay_subs = build_target_subsets_extended(
        "pems-bay", args.pems_bay_dir, args.pems_bay_adj,
        source_num_nodes, source_stats, n_subsets=args.n_subsets)
    print(f"  pems-bay N_unique={bay_subs[0]['n_unique']}")

    all_results = {}
    for bb in args.backbones:
        bkeys = sorted(k for k in registry if k.startswith(f"{bb}__pems04__"))
        if not bkeys:
            print(f"[WARN] no PEMS04 ckpts for {bb}, skip")
            continue
        print(f"\n=== {bb.upper()} ===")
        bb_results = {"per_seed": []}
        for key in bkeys:
            entry = registry[key]
            ckpt = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
            cfg = ckpt["config"]
            model_cls = BACKBONE_REGISTRY[bb]
            model = model_cls(num_nodes=source_num_nodes,
                              seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
                              hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)
            model.load_state_dict(ckpt["state_dict"]); model.eval()

            metr_padded, metr_nopad = [], []
            for sub in metr_subs:
                loader = make_dataloader(sub["X_tgt"], sub["Y_tgt"], 64, shuffle=False)
                pred, tgt = evaluate_per_node(model, loader, adj_src, device)
                # padded: MAE over ALL `keep` slots (including duplicates) — matches existing eval
                metr_padded.append(mae_on_indices(pred, tgt, list(range(len(sub["keep_idx"]))),
                                                  sub["tgt_stats"]))
                # no-pad: MAE over ONLY the unique target slots
                metr_nopad.append(mae_on_indices(pred, tgt, sub["unique_idx"], sub["tgt_stats"]))

            bay_padded, bay_nopad = [], []
            for sub in bay_subs:
                loader = make_dataloader(sub["X_tgt"], sub["Y_tgt"], 64, shuffle=False)
                pred, tgt = evaluate_per_node(model, loader, adj_src, device)
                bay_padded.append(mae_on_indices(pred, tgt, list(range(len(sub["keep_idx"]))),
                                                 sub["tgt_stats"]))
                bay_nopad.append(mae_on_indices(pred, tgt, sub["unique_idx"], sub["tgt_stats"]))

            seed_res = {
                "key": key,
                "metr_padded_mae_mean": float(np.mean(metr_padded)),
                "metr_padded_mae_std":  float(np.std(metr_padded)),
                "metr_nopad_mae_mean":  float(np.mean(metr_nopad)),
                "metr_nopad_mae_std":   float(np.std(metr_nopad)),
                "bay_padded_mae_mean":  float(np.mean(bay_padded)),
                "bay_nopad_mae_mean":   float(np.mean(bay_nopad)),
            }
            seed_res["metr_pad_minus_nopad"] = seed_res["metr_padded_mae_mean"] - seed_res["metr_nopad_mae_mean"]
            seed_res["bay_pad_minus_nopad"]  = seed_res["bay_padded_mae_mean"]  - seed_res["bay_nopad_mae_mean"]
            bb_results["per_seed"].append(seed_res)
            print(f"  {key}: metr padded={seed_res['metr_padded_mae_mean']:.3f}  "
                  f"nopad={seed_res['metr_nopad_mae_mean']:.3f}  "
                  f"Δ={seed_res['metr_pad_minus_nopad']:+.3f}")

        # aggregate over seeds
        keys = ["metr_padded_mae_mean", "metr_nopad_mae_mean",
                "bay_padded_mae_mean", "bay_nopad_mae_mean",
                "metr_pad_minus_nopad", "bay_pad_minus_nopad"]
        for k in keys:
            vals = [s[k] for s in bb_results["per_seed"]]
            bb_results[f"{k}__mean"] = float(np.mean(vals))
            bb_results[f"{k}__std"]  = float(np.std(vals))
        all_results[bb] = bb_results

    # ── summary print ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'Backbone':<10}  {'METR padded':>12}  {'METR nopad':>12}  {'Δ (mph)':>10}  ")
    print("-" * 78)
    for bb in args.backbones:
        if bb not in all_results:
            continue
        r = all_results[bb]
        print(f"{bb:<10}  "
              f"{r['metr_padded_mae_mean__mean']:>12.4f}  "
              f"{r['metr_nopad_mae_mean__mean']:>12.4f}  "
              f"{r['metr_pad_minus_nopad__mean']:>+10.4f}")
    print("=" * 78)

    out_path = os.path.join(args.output_dir, "cross_city_no_pad_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
