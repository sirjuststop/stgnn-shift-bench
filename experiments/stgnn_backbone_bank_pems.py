"""
Backbone Bank Builder — PEMS-BAY Training
Direction C Tier-2: Bidirectional city transfer diagnostic.

Trains DCRNN, GWNet, GRU (and optionally STGCN) on PEMS-BAY with seeds 3,4,5.
Checkpoints saved to baselines/stgnn_pems_bay/.
Registry updated with pems-bay entries.

Imports model definitions and utilities from stgnn_backbone_bank.py (no code duplication).

Usage:
  cd <project_root>
  python experiments/stgnn_backbone_bank_pems.py \\
    --backbones dcrnn gwnet gru \\
    --seeds 3 4 5 \\
    --output_dir baselines/stgnn_pems_bay/ \\
    --registry baselines/stgnn_protocol_registry.json \\
    --data_dir data/ \\
    --adj_path data/pems-bay/adj_mx_bay.pkl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import shared utilities from the existing backbone bank
sys.path.insert(0, str(Path(__file__).parent))
from stgnn_backbone_bank import (
    BACKBONE_REGISTRY, _load_raw, load_adjacency,
    make_dataloader, train_backbone, update_registry, set_seed,
)


# ── PEMS-BAY Data Loader ───────────────────────────────────────────────────────

def load_pems_bay(data_dir: str, seq_len: int = 12, pred_len: int = 12):
    """Load PEMS-BAY (325 sensors) and return train/val/test splits."""
    data = _load_raw(data_dir, "PEMS-BAY")  # (T, 325)
    T, N = data.shape
    print(f"  PEMS-BAY raw shape: {data.shape}")

    # Chronological 70/10/20% split (same as METR-LA protocol)
    train_end = int(T * 0.7)
    val_end = int(T * 0.8)
    train_raw = data[:train_end]
    val_raw = data[train_end:val_end]
    test_raw = data[val_end:]

    # Z-score normalization (fit on train only)
    mean = train_raw.mean(axis=0, keepdims=True)  # (1, 325)
    std = train_raw.std(axis=0, keepdims=True) + 1e-8
    train_norm = (train_raw - mean) / std
    val_norm = (val_raw - mean) / std
    test_norm = (test_raw - mean) / std

    def make_windows(arr):
        X, Y = [], []
        for t in range(len(arr) - seq_len - pred_len + 1):
            X.append(arr[t:t + seq_len])
            Y.append(arr[t + seq_len:t + seq_len + pred_len])
        return np.stack(X), np.stack(Y)

    X_tr, Y_tr = make_windows(train_norm)
    X_va, Y_va = make_windows(val_norm)
    X_te, Y_te = make_windows(test_norm)

    stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build backbone bank trained on PEMS-BAY (Tier-2 bidirectional transfer)"
    )
    parser.add_argument("--backbones", nargs="+",
                        default=["dcrnn", "gwnet", "gru"],
                        choices=list(BACKBONE_REGISTRY.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--output_dir", default="baselines/stgnn_pems_bay/")
    parser.add_argument("--registry", default="baselines/stgnn_protocol_registry.json")
    parser.add_argument("--data_dir", default="data/pems-bay/")
    parser.add_argument("--adj_path", default="data/pems-bay/adj_mx_bay.pkl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Frozen training config (same as Tier-0 METR-LA protocol)
    cfg = {
        "epochs": 100, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.3,
        "seq_len": 12, "pred_len": 12, "hidden_dim": 64, "batch_size": 64,
        "early_stopping_patience": 15,
    }

    print("=" * 60)
    print("PEMS-BAY Backbone Bank Builder — Tier-2 Bidirectional Transfer")
    print("=" * 60)
    print(f"Backbones: {args.backbones}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print()

    print("Loading PEMS-BAY dataset...")
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), stats = load_pems_bay(
        args.data_dir, cfg["seq_len"], cfg["pred_len"]
    )
    num_nodes = X_tr.shape[2]  # 325 for PEMS-BAY
    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    print(f"  Num nodes: {num_nodes}")

    print("Loading PEMS-BAY adjacency matrix...")
    adj = load_adjacency(args.adj_path, num_nodes, device)
    print(f"  Adjacency shape: {adj.shape}")

    train_loader = make_dataloader(X_tr, Y_tr, cfg["batch_size"], shuffle=True)
    val_loader   = make_dataloader(X_va, Y_va, cfg["batch_size"], shuffle=False)
    test_loader  = make_dataloader(X_te, Y_te, cfg["batch_size"], shuffle=False)

    all_results = {}

    for backbone in args.backbones:
        per_seed_results = []
        for seed in args.seeds:
            key = f"{backbone}__pems-bay__seed{seed}"
            ckpt_path = os.path.join(args.output_dir, f"{key}.pt")

            if os.path.exists(ckpt_path):
                print(f"[SKIP] {key} already exists — backbone bank is frozen.")
                # Load metrics from registry for reporting
                if os.path.exists(args.registry):
                    with open(args.registry) as f:
                        reg = json.load(f)
                    if key in reg:
                        per_seed_results.append(reg[key]["test_metrics"])
                continue

            print(f"\n{'='*60}")
            print(f"Training {backbone.upper()} on PEMS-BAY | seed={seed}")
            print(f"{'='*60}")
            set_seed(seed)
            t0 = time.time()

            model, state_dict, test_metrics = train_backbone(
                backbone, num_nodes, adj,
                train_loader, val_loader, test_loader,
                cfg, device, stats
            )

            elapsed = time.time() - t0
            print(f"  Test: MAE={test_metrics['MAE']:.4f}  RMSE={test_metrics['RMSE']:.4f}"
                  f"  MAPE={test_metrics['MAPE']:.2f}%  [{elapsed:.0f}s]")

            # Save frozen checkpoint
            torch.save({
                "backbone": backbone,
                "seed": seed,
                "dataset": "pems-bay",
                "state_dict": state_dict,
                "test_metrics": test_metrics,
                "config": cfg,
                "num_nodes": num_nodes,
                "train_stats": {
                    "mean": stats["mean"].tolist(),
                    "std": stats["std"].tolist(),
                },
            }, ckpt_path)
            print(f"  Checkpoint frozen: {ckpt_path}")

            per_seed_results.append(test_metrics)
            update_registry(args.registry, key, {
                "key": key,
                "backbone": backbone,
                "dataset": "pems-bay",
                "seed": seed,
                "checkpoint": ckpt_path,
                "test_metrics": test_metrics,
                "status": "frozen",
            })

        if per_seed_results:
            maes = [r["MAE"] for r in per_seed_results]
            all_results[backbone] = {
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "per_seed": per_seed_results,
            }
            print(f"\n[{backbone.upper()}] PEMS-BAY test MAE: "
                  f"{np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("PEMS-BAY Backbone Bank — COMPLETE")
    print("=" * 60)
    for bb, res in all_results.items():
        print(f"  {bb.upper():8s} MAE = {res['mae_mean']:.4f} ± {res['mae_std']:.4f}")

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "pems_bay_bank_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print(f"Registry: {args.registry}")


if __name__ == "__main__":
    main()
