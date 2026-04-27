"""
Shift Evaluation v5 — PEMS04-source Cross-City Protocol (P18)

Dedicated script for the second city-pair experiment introduced in Reviewer Fix v5.
Loads frozen backbones trained on PEMS04 (from stgnn_backbone_bank.py --dataset pems04)
and evaluates them zero-shot on two targets: METR-LA and PEMS-BAY.

The main v4/v5 pipeline (stgnn_shift_eval_v2.py) remains the canonical source-of-truth
for METR-LA-source experiments; this script is intentionally small and side-channel so
that P18 additions cannot perturb any verified v4 result.

Method:
  * For each PEMS04-trained backbone checkpoint, build a 307-node model.
  * Target evaluation: randomly subsample the target city's sensors to 307 nodes
    (K=5 independent subsets; seeds 42..46), mirroring the METR-LA→PEMS-BAY protocol.
  * Report target-stat MAE (i.e. normalize with the target's own train mean/std,
    then denormalize back into target-native units) so values are comparable with
    the v4 main-table column.
  * The source-city adjacency is used at inference for all models (including DCRNN
    and STGCN) — this matches the v4 "source-only adjacency fairness control"
    (P16) and keeps all backbones on the same structural footing.

Pre-registered hypothesis P18:
  DCRNN achieves the lowest target-stat MAE among {DCRNN, GWNet, GWNet[Strong], GRU}
  on ≥1 of the two new directions (PEMS04→METR-LA, PEMS04→PEMS-BAY), confirming that
  the fixed-diffusion transfer advantage is not METR-LA-specific.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stgnn_backbone_bank import (
    BACKBONE_REGISTRY, BACKBONE_CFG_OVERRIDES, DATASET_REGISTRY,
    load_pems04, load_adjacency, load_metr_la,
    make_dataloader, evaluate, set_seed, _load_raw, _windows,
)


def build_target_subsets(target_key, target_dir, adj_path_target, source_num_nodes,
                         source_stats, n_subsets=5, seq_len=12, pred_len=12, base_seed=42):
    """Build K random subsets of size `source_num_nodes` from a target city for
    zero-shot evaluation of a model trained with `source_num_nodes` input channels.

    Both source-stat and target-stat normalizations are returned per subset, matching
    the protocol used by `load_pems_bay_subsets` in stgnn_shift_eval_v2.py.
    """
    if target_key == "metr-la":
        raw_full = _load_raw(target_dir, "METR-LA")  # (T, 207)
    elif target_key == "pems-bay":
        raw_full = _load_raw(target_dir, "PEMS-BAY")  # (T, 325)
    else:
        raise ValueError(f"Unsupported target dataset: {target_key}")

    T, N_target = raw_full.shape

    train_end = int(T * 0.7)
    test_start = int(T * 0.8)
    train_raw = raw_full[:train_end]
    test_raw = raw_full[test_start:]

    target_mean = train_raw.mean(axis=0, keepdims=True)          # (1, N_target)
    target_std = train_raw.std(axis=0, keepdims=True) + 1e-8

    src_global_mean = float(np.asarray(source_stats["mean"]).mean())
    src_global_std = float(np.asarray(source_stats["std"]).mean())

    def windows_with(arr, s_mean, s_std):
        norm = (arr - s_mean) / s_std
        X, Y = _windows(norm.astype(np.float32), seq_len, pred_len)
        return X.astype(np.float32), Y.astype(np.float32)

    # Target adjacency (full size)
    with open(adj_path_target, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    adj_full = obj[2] if (isinstance(obj, (tuple, list)) and len(obj) == 3) else obj
    adj_full = np.asarray(adj_full).astype(np.float32)

    subsets = []
    for i in range(n_subsets):
        rng = np.random.RandomState(base_seed + i)
        if N_target >= source_num_nodes:
            keep = sorted(rng.choice(N_target, source_num_nodes, replace=False).tolist())
        else:
            # Target is smaller than source model capacity: pad with repeated indices
            keep = list(range(N_target)) + [0] * (source_num_nodes - N_target)

        X_src, Y_src = windows_with(test_raw[:, keep], src_global_mean, src_global_std)

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
            "X_src": X_src, "Y_src": Y_src,
            "X_tgt": X_tgt, "Y_tgt": Y_tgt,
            "adj_norm": adj_norm,
            "tgt_stats": {"mean": t_mean, "std": t_std},
        })
    return subsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry",  default="baselines/stgnn_protocol_registry.json")
    ap.add_argument("--backbones", nargs="+",
                    default=["dcrnn", "gwnet", "gwnet_strong", "gru"])
    ap.add_argument("--source_data_dir", default="data/pems04")
    ap.add_argument("--source_adj_path", default="data/pems04/adj_mx.pkl")
    ap.add_argument("--metr_la_dir",     default="data/metr-la")
    ap.add_argument("--metr_la_adj",     default="data/metr-la/adj_mx.pkl")
    ap.add_argument("--pems_bay_dir",    default="data/pems-bay")
    ap.add_argument("--pems_bay_adj",    default="data/pems-bay/adj_mx_bay.pkl")
    ap.add_argument("--output_dir",      default="results/reviewer_fix_v5")
    ap.add_argument("--n_subsets",       type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load source (PEMS04) test set + stats (for source-stat normalized targets)
    print("Loading PEMS04 source data + stats...")
    (_, _), (_, _), (X_te_src, Y_te_src), source_stats = load_pems04(args.source_data_dir)
    source_num_nodes = X_te_src.shape[2]
    print(f"  PEMS04 nodes={source_num_nodes}  test_windows={len(X_te_src)}")

    adj_src = load_adjacency(args.source_adj_path, source_num_nodes, device)
    clean_loader = make_dataloader(X_te_src, Y_te_src, 64, shuffle=False)

    with open(args.registry) as f:
        registry = json.load(f)

    # Build target subsets for both directions
    print(f"Building {args.n_subsets} METR-LA subsets of size {source_num_nodes}...")
    metr_subsets = build_target_subsets(
        "metr-la", args.metr_la_dir, args.metr_la_adj,
        source_num_nodes, source_stats, n_subsets=args.n_subsets,
    )
    print(f"Building {args.n_subsets} PEMS-BAY subsets of size {source_num_nodes}...")
    bay_subsets = build_target_subsets(
        "pems-bay", args.pems_bay_dir, args.pems_bay_adj,
        source_num_nodes, source_stats, n_subsets=args.n_subsets,
    )

    all_results = {}
    for bb in args.backbones:
        bkeys = sorted([k for k in registry if k.startswith(f"{bb}__pems04__")])
        if not bkeys:
            print(f"[WARN] No PEMS04 checkpoints for {bb}. Skipping.")
            continue

        print(f"\n{'='*60}\nEvaluating {bb.upper()} (PEMS04 → METR-LA, PEMS04 → PEMS-BAY)\n{'='*60}")

        bb_results = {}
        for key in bkeys:
            entry = registry[key]
            ckpt = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
            cfg = ckpt["config"]

            model_cls = BACKBONE_REGISTRY[bb]
            model = model_cls(
                num_nodes=source_num_nodes,
                seq_len=cfg["seq_len"],
                pred_len=cfg["pred_len"],
                hidden_dim=cfg["hidden_dim"],
                dropout=cfg["dropout"],
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            seed = entry["seed"]
            seed_res = {"backbone": bb, "seed": seed}

            # In-distribution PEMS04 sanity check (denormalized to native flow units)
            src_clean = evaluate(model, clean_loader, adj_src, device, stats=source_stats)
            seed_res["pems04_indist_MAE"] = src_clean["MAE"]

            # Direction 1: PEMS04 → METR-LA (use source adj for fairness — same protocol
            # as the v4 source_only_adj_control for P16)
            metr_tgt_maes = []
            for sub in metr_subsets:
                tgt_loader = make_dataloader(sub["X_tgt"], sub["Y_tgt"], 64, shuffle=False)
                m = evaluate(model, tgt_loader, adj_src, device, stats=sub["tgt_stats"])
                metr_tgt_maes.append(m["MAE"])
            seed_res["pems04_to_metr_la"] = {
                "city_tgt_MAE_mean": float(np.mean(metr_tgt_maes)),
                "city_tgt_MAE_std":  float(np.std(metr_tgt_maes)),
                "n_subsets": len(metr_subsets),
            }

            # Direction 2: PEMS04 → PEMS-BAY
            bay_tgt_maes = []
            for sub in bay_subsets:
                tgt_loader = make_dataloader(sub["X_tgt"], sub["Y_tgt"], 64, shuffle=False)
                m = evaluate(model, tgt_loader, adj_src, device, stats=sub["tgt_stats"])
                bay_tgt_maes.append(m["MAE"])
            seed_res["pems04_to_pems_bay"] = {
                "city_tgt_MAE_mean": float(np.mean(bay_tgt_maes)),
                "city_tgt_MAE_std":  float(np.std(bay_tgt_maes)),
                "n_subsets": len(bay_subsets),
            }

            print(f"  [seed={seed}] PEMS04_indist={src_clean['MAE']:.4f}  "
                  f"→METR-LA={np.mean(metr_tgt_maes):.4f}±{np.std(metr_tgt_maes):.4f}  "
                  f"→PEMS-BAY={np.mean(bay_tgt_maes):.4f}±{np.std(bay_tgt_maes):.4f}")

            bb_results[key] = seed_res
        all_results[bb] = bb_results

    out_path = os.path.join(args.output_dir, "pems04_cross_city_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Summary table — ranking across backbones for each direction
    print("\n" + "="*78)
    print(f"{'Backbone':<14} {'PEMS04 indist':>14} {'→METR-LA':>16} {'→PEMS-BAY':>16}")
    print("-"*78)
    for bb in args.backbones:
        if bb not in all_results:
            continue
        br = all_results[bb]
        indist = [br[k]["pems04_indist_MAE"] for k in br]
        to_metr = [br[k]["pems04_to_metr_la"]["city_tgt_MAE_mean"] for k in br]
        to_bay = [br[k]["pems04_to_pems_bay"]["city_tgt_MAE_mean"] for k in br]
        print(f"{bb:<14} {np.mean(indist):>+10.4f}±{np.std(indist):.3f}  "
              f"{np.mean(to_metr):>+10.4f}±{np.std(to_metr):.3f}  "
              f"{np.mean(to_bay):>+10.4f}±{np.std(to_bay):.3f}")
    print("="*78)


if __name__ == "__main__":
    main()
