"""
Density Sweep — GWNet Adaptive Adjacency Top-k Sparsification (Direction C)

Hypothesis (P20): denser adaptive connectivity is associated with higher dropout
fragility under a fixed trained checkpoint.

Method:
  For each frozen GWNet[adaptive] checkpoint (3 seeds), compute the full softmax
  adjacency A = softmax(E1 @ E2^T), then sparsify it to top-k entries per row
  (renormalized) at inference time.  No weights are updated.

  For each k in K_LIST:
    - evaluate in-distribution MAE on METR-LA test set
    - evaluate sensor-dropout Δ (30%, 5 random masks, same seeds as tier-1)

  k=full reproduces original GWNet[adaptive] numbers (sanity check).
  GWNet[fixed] (already in registry) serves as the k=0 anchor point.

  MTGNNLight (k=10 cosine-sim, also in registry) is an independent corroboration
  point listed in the output for reference but NOT part of the sweep.

Framing: this is *association evidence*, not causal isolation.  Top-k filtering
also changes effective receptive field of learned embeddings; the paper says
"associated with higher dropout fragility" and explicitly acknowledges this.

Output: results/density_sweep/density_sweep_results.json
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch

# Allow imports from experiments/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stgnn_backbone_bank import (
    BACKBONE_REGISTRY, load_metr_la, load_adjacency,
    make_dataloader, evaluate, set_seed,
)
from stgnn_shift_eval_v2 import evaluate_sensor_zeromask


# ── Config ─────────────────────────────────────────────────────────────────────

K_LIST = [5, 10, 15, 20, 30]   # plus "full" added automatically
N_MASKS = 5                     # masks per k-value (same seeds as tier-1: 42..46)
DROPOUT_RATIO = 0.30
TOLERANCE = 1e-2                # MAE tolerance for k=full sanity check


def topk_row_normalize(adj: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k entries per row, zero the rest, renormalize rows to sum to 1.

    Args:
        adj: (N, N) dense adjacency (already softmax-normalized or any non-negative)
        k:   number of neighbours to retain per row (inclusive of self if present)

    Returns:
        (N, N) sparse, row-normalized adjacency
    """
    N = adj.shape[0]
    k = min(k, N)
    # topk returns values and indices; scatter back
    vals, idx = adj.topk(k, dim=-1)        # (N, k)
    sparse = torch.zeros_like(adj)
    sparse.scatter_(-1, idx, vals)
    # Renormalize: each row sums to the original top-k mass; rescale to 1
    row_sum = sparse.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sparse = sparse / row_sum
    # Verify row sums ≈ 1
    assert (sparse.sum(dim=-1) - 1.0).abs().max().item() < 1e-5, \
        "Row renormalization failed: rows do not sum to 1"
    return sparse


def patch_gwnet_topk(model, k: int):
    """Monkey-patch model.get_adaptive_adj to return top-k sparsified adjacency.

    This modifies the instance method only, leaving the class untouched.
    The patch is read-only with respect to model parameters (no grad, no update).
    """
    def get_adaptive_adj_topk(self_m, topology_dropout_p=0.0):
        with torch.no_grad():
            adj_full = torch.softmax(self_m.node_emb1 @ self_m.node_emb2.T, dim=-1)
        return topk_row_normalize(adj_full, k)

    model.get_adaptive_adj = types.MethodType(get_adaptive_adj_topk, model)


def restore_gwnet_original(model):
    """Remove the monkey-patch, restoring the original get_adaptive_adj."""
    if "get_adaptive_adj" in model.__dict__:
        del model.__dict__["get_adaptive_adj"]


def run_density_sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load METR-LA data
    data_dir = args.data_dir
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), metr_stats = load_metr_la(data_dir)
    N = X_te.shape[2]
    print(f"METR-LA test set: {X_te.shape}, N={N}")

    adj_metr = load_adjacency(args.adj_path, N, device)

    clean_loader = make_dataloader(X_te, Y_te, batch_size=64, shuffle=False)

    # Load protocol registry to find GWNet checkpoints
    with open(args.registry, "r") as f:
        registry = json.load(f)

    gwnet_keys = sorted([
        k for k in registry
        if k.startswith("gwnet__metr-la__") and "strong" not in k
    ])
    if not gwnet_keys:
        raise RuntimeError("No gwnet__metr-la__* entries found in registry")
    print(f"GWNet checkpoints: {gwnet_keys}")

    k_values = K_LIST + ["full"]
    results = {}

    for key in gwnet_keys:
        entry = registry[key]
        seed  = entry["seed"]
        ckpt  = torch.load(entry["checkpoint"], map_location=device,
                           weights_only=False)
        cfg   = ckpt["config"]

        model = BACKBONE_REGISTRY["gwnet"](
            num_nodes=N,
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Confirm no gradients are computed
        for p in model.parameters():
            p.requires_grad_(False)

        key_res = {"seed": seed, "backbone": "gwnet", "k_results": {}}

        for k in k_values:
            k_label = "full" if k == "full" else str(k)
            k_int   = N if k == "full" else k

            if k == "full":
                restore_gwnet_original(model)
            else:
                patch_gwnet_topk(model, k_int)

            # In-distribution MAE
            id_metrics = evaluate(model, clean_loader, adj_metr, device,
                                  stats=metr_stats)
            id_mae = id_metrics["MAE"]

            # Sensor-dropout Δ — averaged over N_MASKS random masks
            delta_list = []
            for mask_seed in range(42, 42 + N_MASKS):
                sd = evaluate_sensor_zeromask(
                    model, X_te, Y_te, adj_metr, device,
                    dropout_ratio=DROPOUT_RATIO, seed=mask_seed,
                )
                delta_list.append(sd["delta_sensor"])

            delta_mean = float(np.mean(delta_list))
            delta_std  = float(np.std(delta_list))
            rel_fragility = delta_mean / id_mae if id_mae > 0 else float("nan")

            key_res["k_results"][k_label] = {
                "k":              k_int,
                "id_mae":         id_mae,
                "delta_sensor_mean": delta_mean,
                "delta_sensor_std":  delta_std,
                "rel_fragility":  rel_fragility,
            }
            print(f"  seed={seed} k={k_label:>4}  "
                  f"ID_MAE={id_mae:.4f}  "
                  f"Δsensor={delta_mean:+.4f}±{delta_std:.4f}  "
                  f"rel={rel_fragility:.4f}")

        results[key] = key_res

    # ── Sanity check: k=full must reproduce registry values ──────────────────
    print("\n── Sanity check: k=full vs registry ──────────────────")
    any_failed = False
    for key in gwnet_keys:
        registered_mae = registry[key]["test_metrics"]["MAE"]
        sweep_full_mae = results[key]["k_results"]["full"]["id_mae"]
        diff = abs(sweep_full_mae - registered_mae)
        status = "OK" if diff < TOLERANCE else "FAIL"
        print(f"  {key}: registry={registered_mae:.4f}  sweep_full={sweep_full_mae:.4f}  "
              f"diff={diff:.5f}  [{status}]")
        if diff >= TOLERANCE:
            any_failed = True
    if any_failed:
        print("WARNING: k=full sanity check failed. Results may be unreliable.")
    else:
        print("Sanity check PASSED.")

    # ── Aggregate across seeds ────────────────────────────────────────────────
    print("\n── Aggregate (mean ± std across seeds) ──────────────")
    print(f"{'k':>6}  {'ID_MAE':>10}  {'Δsensor':>10}  {'rel_frag':>10}")
    print("-" * 45)
    agg = {}
    for k_label in [str(k) for k in K_LIST] + ["full"]:
        id_maes   = [results[key]["k_results"][k_label]["id_mae"]
                     for key in gwnet_keys if k_label in results[key]["k_results"]]
        deltas    = [results[key]["k_results"][k_label]["delta_sensor_mean"]
                     for key in gwnet_keys if k_label in results[key]["k_results"]]
        rel_frags = [results[key]["k_results"][k_label]["rel_fragility"]
                     for key in gwnet_keys if k_label in results[key]["k_results"]]
        agg[k_label] = {
            "id_mae_mean":        float(np.mean(id_maes)),
            "id_mae_std":         float(np.std(id_maes)),
            "delta_sensor_mean":  float(np.mean(deltas)),
            "delta_sensor_std":   float(np.std(deltas)),
            "rel_fragility_mean": float(np.mean(rel_frags)),
            "rel_fragility_std":  float(np.std(rel_frags)),
        }
        print(f"  k={k_label:>4}  "
              f"ID={np.mean(id_maes):.4f}±{np.std(id_maes):.4f}  "
              f"Δ={np.mean(deltas):+.4f}±{np.std(deltas):.4f}  "
              f"rel={np.mean(rel_frags):.4f}")

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "per_seed": results,
        "aggregate": agg,
        "config": {
            "k_list": K_LIST,
            "n_masks": N_MASKS,
            "dropout_ratio": DROPOUT_RATIO,
            "n_seeds": len(gwnet_keys),
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {args.output}")

    # ── Brief trend report ────────────────────────────────────────────────────
    print("\n── Trend check (is Δsensor broadly increasing with k?) ──")
    delta_means = [agg[str(k)]["delta_sensor_mean"] for k in K_LIST] + \
                  [agg["full"]["delta_sensor_mean"]]
    k_labels    = [str(k) for k in K_LIST] + ["full"]
    increasing_pairs = sum(
        1 for i in range(len(delta_means) - 1) if delta_means[i + 1] > delta_means[i]
    )
    total_pairs = len(delta_means) - 1
    print(f"  Δsensor values by k: " +
          "  ".join(f"{k}:{v:+.4f}" for k, v in zip(k_labels, delta_means)))
    print(f"  Increasing pairs: {increasing_pairs}/{total_pairs} "
          f"({'broadly monotonic' if increasing_pairs >= total_pairs - 1 else 'non-monotonic'})")


def main():
    parser = argparse.ArgumentParser(description="GWNet adaptive-adj density sweep")
    parser.add_argument("--data_dir",  default="data/metr-la")
    parser.add_argument("--adj_path",  default="data/metr-la/adj_mx.pkl")
    parser.add_argument("--registry",  default="baselines/stgnn_protocol_registry.json")
    parser.add_argument("--output",    default="results/density_sweep/density_sweep_results.json")
    args = parser.parse_args()

    run_density_sweep(args)


if __name__ == "__main__":
    main()
