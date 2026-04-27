"""
mask_aware_dropout_sweep.py
============================
Sweep mask-aware GWNet across multiple dropout ratios + one structured pattern,
to address reviewer Q4 (stability beyond the two reported operating points).

For each ratio in {0.05, 0.10, 0.20, 0.30, 0.40}:
  - Standard ΔSensor: GWNet on zero-masked input
  - Mask-aware ΔSensor: GWNet with adjacency rows/cols zeroed for dropped sensors
  - Recovery% = (1 - mask_aware/standard) * 100

Plus one structured pattern: hub-dropout 30% (drops top-30% nodes by degree
in the METR-LA distance adjacency).

3 model seeds × 5 mask seeds per cell. METR-LA only.
Output: results/polish_v15/mask_aware_ratio_sweep.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reviewer_clarification import evaluate_mask_aware_gwnet
from experiments.stgnn_backbone_bank import (
    BACKBONE_REGISTRY,
    load_adjacency,
    load_metr_la,
)

DROPOUT_RATIOS = [0.05, 0.10, 0.20, 0.30, 0.40]
N_MASK_SEEDS = 5
BASE_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hub_dropout_indices(adj_norm: torch.Tensor, ratio: float) -> list[int]:
    """Return the top-`ratio` fraction of nodes by row-sum degree in the
    symmetric-normalized adjacency (i.e., highest-connectivity nodes).
    Note: adj_norm is already D^{-1/2} A D^{-1/2}; row-sum is ~constant for
    perfectly symmetric normalization, so we use raw connection count via
    a count-of-non-zeros heuristic.
    """
    deg = (adj_norm > 1e-8).float().sum(dim=1)
    n = int(adj_norm.shape[0] * ratio)
    return torch.topk(deg, n).indices.cpu().tolist()


def evaluate_mask_aware_with_indices(model, X_te, Y_te, adj, device,
                                     drop_idx: list[int]):
    """Single-mask version of evaluate_mask_aware_gwnet with a fixed drop set."""
    from experiments.stgnn_backbone_bank import make_dataloader
    from experiments.reviewer_clarification import _collect_preds_targets

    keep_idx = [j for j in range(X_te.shape[2]) if j not in drop_idx]
    X_masked = X_te.copy()
    X_masked[:, :, drop_idx] = 0.0

    masked_loader = make_dataloader(X_masked, Y_te, 64, shuffle=False)
    clean_loader = make_dataloader(X_te, Y_te, 64, shuffle=False)

    p_clean, t_clean = _collect_preds_targets(model, clean_loader, adj, device)
    p_std, t_std = _collect_preds_targets(model, masked_loader, adj, device)
    mae_clean = (p_clean[:, keep_idx, :] - t_clean[:, keep_idx, :]).abs().mean().item()
    mae_standard = (p_std[:, keep_idx, :] - t_std[:, keep_idx, :]).abs().mean().item()
    delta_standard = mae_standard - mae_clean

    orig_get_adj = model.get_adaptive_adj
    _di = drop_idx

    def masked_adj_fn(topology_dropout_p=0.0, _d=_di):
        a = orig_get_adj(topology_dropout_p).clone()
        a[_d, :] = 0.0
        a[:, _d] = 0.0
        row_sum = a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return a / row_sum

    model.get_adaptive_adj = masked_adj_fn
    p_ma, _ = _collect_preds_targets(model, masked_loader, adj, device)
    mae_ma = (p_ma[:, keep_idx, :] - t_clean[:, keep_idx, :]).abs().mean().item()
    delta_aware = mae_ma - mae_clean
    model.get_adaptive_adj = orig_get_adj  # restore

    return delta_standard, delta_aware


def main():
    print(f"DEVICE = {DEVICE}")
    registry_path = REPO_ROOT / "baselines" / "stgnn_protocol_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    print("Loading METR-LA test split + adjacency...")
    (_, _), (_, _), (X_te, Y_te), _ = load_metr_la(str(REPO_ROOT / "data" / "metr-la"))
    N = X_te.shape[2]
    adj = load_adjacency(str(REPO_ROOT / "data" / "metr-la" / "adj_mx.pkl"), N, DEVICE)

    gw_keys = sorted(k for k in registry if k.startswith("gwnet__metr-la__seed"))
    print(f"GWNet checkpoints: {gw_keys}")

    results = {"ratio_sweep": {}, "structured": {}}

    # ── Random-mask sweep ────────────────────────────────────────────────────
    for ratio in DROPOUT_RATIOS:
        print(f"\n=== ratio = {ratio:.2f} ===")
        per_seed_std, per_seed_aware, per_seed_recovery = [], [], []
        for key in gw_keys:
            entry = registry[key]
            ckpt = torch.load(entry["checkpoint"], map_location=DEVICE, weights_only=False)
            cfg = ckpt["config"]
            model = BACKBONE_REGISTRY["gwnet"](
                num_nodes=N, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
                hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
            model.load_state_dict(ckpt["state_dict"]); model.eval()

            ma = evaluate_mask_aware_gwnet(
                model, X_te, Y_te, adj, DEVICE,
                dropout_ratio=ratio, n_mask_seeds=N_MASK_SEEDS, base_seed=BASE_SEED)
            per_seed_std.append(ma["delta_standard_mean"])
            per_seed_aware.append(ma["delta_aware_mean"])
            per_seed_recovery.append(ma["recovery_pct"])
            print(f"  {key}: std={ma['delta_standard_mean']:+.4f}  "
                  f"aware={ma['delta_aware_mean']:+.4f}  rec={ma['recovery_pct']:.1f}%")

        cell = {
            "ratio": ratio,
            "delta_standard_mean": float(np.mean(per_seed_std)),
            "delta_standard_std":  float(np.std(per_seed_std)),
            "delta_aware_mean":    float(np.mean(per_seed_aware)),
            "delta_aware_std":     float(np.std(per_seed_aware)),
            "recovery_mean":       float(np.mean(per_seed_recovery)),
            "recovery_std":        float(np.std(per_seed_recovery)),
        }
        results["ratio_sweep"][f"r{int(ratio*100):02d}"] = cell

    # ── Structured (hub) dropout, ratio = 0.30 ───────────────────────────────
    print(f"\n=== structured: hub-dropout 0.30 ===")
    drop_idx = hub_dropout_indices(adj, 0.30)
    per_seed_std, per_seed_aware = [], []
    for key in gw_keys:
        entry = registry[key]
        ckpt = torch.load(entry["checkpoint"], map_location=DEVICE, weights_only=False)
        cfg = ckpt["config"]
        model = BACKBONE_REGISTRY["gwnet"](
            num_nodes=N, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        d_std, d_aw = evaluate_mask_aware_with_indices(model, X_te, Y_te, adj, DEVICE,
                                                       drop_idx)
        per_seed_std.append(d_std); per_seed_aware.append(d_aw)
        print(f"  {key}: std={d_std:+.4f}  aware={d_aw:+.4f}")
    rec_mean = (1.0 - np.mean(per_seed_aware) / max(np.mean(per_seed_std), 1e-12)) * 100.0
    results["structured"]["hub_30"] = {
        "ratio": 0.30,
        "drop_n": len(drop_idx),
        "delta_standard_mean": float(np.mean(per_seed_std)),
        "delta_standard_std":  float(np.std(per_seed_std)),
        "delta_aware_mean":    float(np.mean(per_seed_aware)),
        "delta_aware_std":     float(np.std(per_seed_aware)),
        "recovery_mean":       float(rec_mean),
    }

    # ── Output ────────────────────────────────────────────────────────────────
    out_path = REPO_ROOT / "results" / "polish_v15" / "mask_aware_ratio_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print compact summary
    print(f"\n{'='*65}")
    print(f"{'cell':>10}  {'std Δ':>10}  {'aware Δ':>10}  {'recovery':>10}")
    print(f"{'='*65}")
    for k, v in results["ratio_sweep"].items():
        print(f"{k:>10}  {v['delta_standard_mean']:>+10.4f}  "
              f"{v['delta_aware_mean']:>+10.4f}  {v['recovery_mean']:>9.1f}%")
    h = results["structured"]["hub_30"]
    print(f"{'hub_30':>10}  {h['delta_standard_mean']:>+10.4f}  "
          f"{h['delta_aware_mean']:>+10.4f}  {h['recovery_mean']:>9.1f}%")


if __name__ == "__main__":
    main()
