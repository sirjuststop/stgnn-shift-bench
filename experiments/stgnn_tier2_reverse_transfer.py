"""
Tier-2 Reverse Transfer: PEMS-BAY → METR-LA
Direction C: Bidirectional city transfer diagnostic.

Design: To match METR-LA's 207 nodes, we train on a fixed 207-node subsample
of PEMS-BAY (randomly chosen with seed=42 — same as the first forward-eval subset).
The trained 207-node models are then evaluated zero-shot on all 207 METR-LA nodes,
using the METR-LA adjacency matrix.

This is fully symmetric with forward transfer:
  Forward: 207 METR-LA nodes → eval on K×207-node PEMS-BAY subsets
  Reverse: 207 PEMS-BAY nodes → eval on 207 METR-LA nodes

Pre-registered hypotheses (P6-P8):
  P6: Reverse Δ_city gap is larger than forward (harder direction: BAY→LA)
  P7: DCRNN robustness ordering holds symmetrically (DCRNN best, GRU worst)
  P8: GWNet fixed-adj ablation also helps in reverse direction

Usage:
  cd /root/autodl-tmp/xile/benchmark_first_graph_gen
  python experiments/stgnn_tier2_reverse_transfer.py \\
    --output_dir results/tier2/ \\
    --registry baselines/stgnn_protocol_registry.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stgnn_backbone_bank import (
    BACKBONE_REGISTRY, _load_raw, load_adjacency, load_metr_la,
    make_dataloader, evaluate, train_backbone, update_registry, set_seed,
)


# ── PEMS-BAY 207-Node Subsample Loader ─────────────────────────────────────────

def load_pems_bay_subset_for_training(data_dir: str, adj_path: str,
                                       num_nodes_target: int = 207,
                                       subset_seed: int = 42,
                                       seq_len: int = 12, pred_len: int = 12):
    """Load a fixed 207-node subsample of PEMS-BAY for training.

    Reproducible: always the same subset (seed=42).
    Returns: (train, val, test) splits + stats + adj + node indices
    """
    data = _load_raw(data_dir, "PEMS-BAY")  # (T, 325)
    T, N_bay = data.shape
    print(f"  PEMS-BAY raw shape: {data.shape}")

    # Select 207 nodes (consistent with eval subset 0)
    rng = np.random.RandomState(subset_seed)
    keep = sorted(rng.choice(N_bay, num_nodes_target, replace=False).tolist())
    data_sub = data[:, keep]  # (T, 207)
    print(f"  Subsample: {num_nodes_target} nodes (seed={subset_seed}, first 5: {keep[:5]})")

    # Chronological 70/10/20% split
    train_end = int(T * 0.7)
    val_end   = int(T * 0.8)
    train_raw = data_sub[:train_end]
    val_raw   = data_sub[train_end:val_end]
    test_raw  = data_sub[val_end:]

    # Z-score normalization (fit on train only)
    mean = train_raw.mean(axis=0, keepdims=True).astype(np.float32)  # (1, 207)
    std  = train_raw.std(axis=0, keepdims=True).astype(np.float32) + 1e-8
    train_norm = (train_raw - mean) / std
    val_norm   = (val_raw - mean) / std
    test_norm  = (test_raw - mean) / std

    def make_windows(arr):
        X, Y = [], []
        for t in range(len(arr) - seq_len - pred_len + 1):
            X.append(arr[t:t + seq_len])
            Y.append(arr[t + seq_len:t + seq_len + pred_len])
        return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)

    X_tr, Y_tr = make_windows(train_norm)
    X_va, Y_va = make_windows(val_norm)
    X_te, Y_te = make_windows(test_norm)

    # Sub-adjacency
    with open(adj_path, "rb") as f:
        _, _, adj_bay_full = pickle.load(f, encoding="latin1")
    adj_sub = adj_bay_full[np.ix_(keep, keep)].astype(np.float32)

    stats = {"mean": mean, "std": std}
    return (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), stats, adj_sub, keep


# ── Reverse Transfer Evaluation ────────────────────────────────────────────────

@torch.no_grad()
def evaluate_reverse_transfer(model, metr_test_data, metr_stats, adj_metr, device,
                               n_subsets: int = 5, batch_size: int = 64):
    """Evaluate a PEMS-BAY-trained model on METR-LA zero-shot.

    Uses METR-LA's own statistics for denormalization (target-stat MAE).
    We use 1 METR-LA subset (all 207 nodes) — no subsampling needed since
    METR-LA has exactly 207 nodes.
    """
    X_te, Y_te = metr_test_data
    loader = make_dataloader(X_te, Y_te, batch_size, shuffle=False)

    # Source-stat MAE (no denormalization — in PEMS-BAY normalized space)
    src_metrics = evaluate(model, loader, adj_metr, device)

    # Target-stat MAE (denormalized with METR-LA's own train statistics)
    tgt_metrics = evaluate(model, loader, adj_metr, device, stats=metr_stats)

    return {
        "city_src_MAE": src_metrics["MAE"],  # in PEMS-BAY normalized space
        "city_tgt_MAE": tgt_metrics["MAE"],  # in METR-LA mph
        "n_metr_nodes": X_te.shape[2],
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tier-2 Reverse Transfer: PEMS-BAY → METR-LA"
    )
    parser.add_argument("--backbones",      nargs="+", default=["dcrnn", "gwnet", "gru"])
    parser.add_argument("--seeds",          nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--output_dir",     default="results/tier2/")
    parser.add_argument("--ckpt_dir",       default="baselines/stgnn_bay207/")
    parser.add_argument("--registry",       default="baselines/stgnn_protocol_registry.json")
    parser.add_argument("--metr_dir",       default="data/metr-la")
    parser.add_argument("--pems_bay_dir",   default="data/pems-bay")
    parser.add_argument("--adj_path_metr",  default="data/metr-la/adj_mx.pkl")
    parser.add_argument("--adj_path_bay",   default="data/pems-bay/adj_mx_bay.pkl")
    parser.add_argument("--subset_seed",    type=int, default=42,
                        help="Fixed seed for 207-node PEMS-BAY training subset")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device)

    # Frozen training config (same as Tier-0)
    cfg = {
        "epochs": 100, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.3,
        "seq_len": 12, "pred_len": 12, "hidden_dim": 64, "batch_size": 64,
        "early_stopping_patience": 15,
    }

    print("=" * 60)
    print("Tier-2 Reverse Transfer: PEMS-BAY(207) → METR-LA")
    print("=" * 60)
    print(f"Backbones: {args.backbones} | Seeds: {args.seeds}")

    # ── Load PEMS-BAY 207-node training subset ───────────────────────────────
    print("\nLoading PEMS-BAY 207-node training subset...")
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), bay_stats, adj_bay_sub, keep_nodes = \
        load_pems_bay_subset_for_training(
            args.pems_bay_dir, args.adj_path_bay,
            num_nodes_target=207, subset_seed=args.subset_seed,
        )
    num_nodes = 207
    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")

    # Convert sub-adjacency to normalized tensor
    adj_bay_t = torch.FloatTensor(adj_bay_sub)
    D = adj_bay_t.sum(dim=1).clamp(min=1e-8)
    D_inv_sqrt = D.pow(-0.5)
    adj_bay_norm = (D_inv_sqrt.unsqueeze(1) * adj_bay_t * D_inv_sqrt.unsqueeze(0)).to(device)

    train_loader = make_dataloader(X_tr, Y_tr, cfg["batch_size"], shuffle=True)
    val_loader   = make_dataloader(X_va, Y_va, cfg["batch_size"], shuffle=False)
    test_loader  = make_dataloader(X_te, Y_te, cfg["batch_size"], shuffle=False)

    # ── Load METR-LA for reverse evaluation ──────────────────────────────────
    print("\nLoading METR-LA for reverse evaluation...")
    (X_tr_m, _), (_, _), (X_te_m, Y_te_m), metr_stats = load_metr_la(args.metr_dir)
    adj_metr = load_adjacency(args.adj_path_metr, 207, device)
    print(f"  METR-LA test: {X_te_m.shape}")

    all_results = {}

    # ── Phase 1: Train backbone bank on PEMS-BAY 207-node subset ────────────
    print("\n" + "=" * 60)
    print("Phase 1: Training backbones on PEMS-BAY (207 nodes)")
    print("=" * 60)

    for backbone in args.backbones:
        per_seed_bay = []
        for seed in args.seeds:
            key = f"{backbone}__bay207__seed{seed}"
            ckpt_path = os.path.join(args.ckpt_dir, f"{key}.pt")

            if os.path.exists(ckpt_path):
                print(f"[SKIP] {key} — already exists")
                continue

            print(f"\nTraining {backbone.upper()} on PEMS-BAY[207] | seed={seed}")
            set_seed(seed)
            t0 = time.time()

            model, state_dict, test_metrics_bay = train_backbone(
                backbone, num_nodes, adj_bay_norm,
                train_loader, val_loader, test_loader,
                cfg, device, bay_stats,
            )

            elapsed = time.time() - t0
            print(f"  PEMS-BAY test: MAE={test_metrics_bay['MAE']:.4f}  "
                  f"RMSE={test_metrics_bay['RMSE']:.4f}  [{elapsed:.0f}s]")

            torch.save({
                "backbone": backbone,
                "seed": seed,
                "dataset": "pems-bay-207subset",
                "subset_seed": args.subset_seed,
                "keep_nodes": keep_nodes,
                "state_dict": state_dict,
                "test_metrics_bay": test_metrics_bay,
                "config": cfg,
                "num_nodes": num_nodes,
                "train_stats": {
                    "mean": bay_stats["mean"].tolist(),
                    "std": bay_stats["std"].tolist(),
                },
            }, ckpt_path)
            print(f"  Checkpoint frozen: {ckpt_path}")

            update_registry(args.registry, key, {
                "key": key,
                "backbone": backbone,
                "dataset": "pems-bay-207subset",
                "seed": seed,
                "checkpoint": ckpt_path,
                "test_metrics": test_metrics_bay,
                "status": "frozen",
            })
            per_seed_bay.append(test_metrics_bay)

        if per_seed_bay:
            maes = [r["MAE"] for r in per_seed_bay]
            print(f"[{backbone.upper()}] PEMS-BAY[207] in-dist MAE: "
                  f"{np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # ── Phase 2: Reverse Transfer Evaluation ────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Reverse Transfer Evaluation (PEMS-BAY[207] → METR-LA)")
    print("=" * 60)

    # Load registry (may have been updated in Phase 1)
    with open(args.registry) as f:
        registry = json.load(f)

    for backbone in args.backbones:
        bkeys = sorted([k for k in registry if k.startswith(f"{backbone}__bay207__")])
        if not bkeys:
            print(f"[WARN] No bay207 checkpoints for {backbone}. Did Phase 1 complete?")
            continue

        print(f"\nEvaluating {backbone.upper()} → METR-LA")
        bb_results = {}

        for key in bkeys:
            entry = registry[key]
            seed = entry["seed"]
            ckpt = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
            cfg_loaded = ckpt["config"]

            model_cls = BACKBONE_REGISTRY[backbone]
            model = model_cls(
                num_nodes=207,
                seq_len=cfg_loaded["seq_len"],
                pred_len=cfg_loaded["pred_len"],
                hidden_dim=cfg_loaded["hidden_dim"],
                dropout=cfg_loaded["dropout"],
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            # In-dist on PEMS-BAY (from checkpoint)
            bay_in_dist_mae = entry["test_metrics"]["MAE"]

            # Reverse transfer: eval on METR-LA
            rev = evaluate_reverse_transfer(
                model, (X_te_m, Y_te_m), metr_stats, adj_metr, device
            )

            print(f"  seed={seed}: PEMS-BAY(in-dist)={bay_in_dist_mae:.4f}  "
                  f"METR-LA(src-norm)={rev['city_src_MAE']:.4f}  "
                  f"METR-LA(tgt-mph)={rev['city_tgt_MAE']:.4f}")

            bb_results[key] = {
                "backbone": backbone,
                "seed": seed,
                "pems_bay_in_dist_MAE": bay_in_dist_mae,
                "reverse_transfer": rev,
            }

        all_results[backbone] = bb_results

    # ── Save Results ─────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "tier2_reverse_transfer.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REVERSE TRANSFER SUMMARY: PEMS-BAY[207] → METR-LA (target-stat MAE, mph)")
    print("=" * 70)
    print(f"{'Backbone':<10} {'In-dist(BAY)':<16} {'METR-LA(mph)':<16} {'Direction'}")
    print("-" * 70)
    for bb in args.backbones:
        if bb not in all_results:
            continue
        br = all_results[bb]
        in_dists = [br[k]["pems_bay_in_dist_MAE"] for k in br]
        rev_maes  = [br[k]["reverse_transfer"]["city_tgt_MAE"] for k in br]
        if in_dists and rev_maes:
            print(f"{bb:<10} {np.mean(in_dists):>6.2f}±{np.std(in_dists):.2f}        "
                  f"{np.mean(rev_maes):>6.2f}±{np.std(rev_maes):.2f}        "
                  f"BAY→LA")
    print("=" * 70)

    # Compare with forward transfer from Tier-1 if available
    forward_results_path = "results/tier1/shift_eval_v2_results.json"
    if os.path.exists(forward_results_path):
        print("\nComparison with forward transfer (METR-LA → PEMS-BAY, from Tier-1):")
        with open(forward_results_path) as f:
            fwd = json.load(f)
        print(f"{'Backbone':<10} {'LA→BAY(mph)':<16} {'BAY→LA(mph)':<16}")
        print("-" * 45)
        for bb in args.backbones:
            # Forward
            if bb in fwd:
                fwd_tgt = [fwd[bb][k]["city"]["city_tgt_MAE_mean"]
                           for k in fwd[bb] if "city" in fwd[bb][k]]
                fwd_mean = np.mean(fwd_tgt) if fwd_tgt else float("nan")
            else:
                fwd_mean = float("nan")
            # Reverse
            if bb in all_results:
                rev_tgt = [all_results[bb][k]["reverse_transfer"]["city_tgt_MAE"]
                           for k in all_results[bb]]
                rev_mean = np.mean(rev_tgt) if rev_tgt else float("nan")
            else:
                rev_mean = float("nan")
            print(f"{bb:<10} {fwd_mean:>8.2f}mph         {rev_mean:>8.2f}mph")
        print("-" * 45)

    print("\n=== Tier-2 Reverse Transfer Complete ===")


if __name__ == "__main__":
    main()
