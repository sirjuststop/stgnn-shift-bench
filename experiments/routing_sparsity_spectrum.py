"""
routing_sparsity_spectrum.py
============================
Compute adjacency sparsity / entropy / max-mass for each model's learned
adaptive graph, then correlate with the known sensor-dropout fragility Δ.

Goal: test whether "routing density ↔ fragility" is a monotonic law that
covers all 8 models, upgrading the 1-mechanism / 3-phenomena asymmetry.

Runs on CPU only — just loads checkpoints, computes adjacency, stats.

Per model, we compute for the final learned adjacency A (averaged over 3 seeds):
  - row_entropy     = -Σ p log p averaged over rows (higher = more diffuse)
  - normalized_entropy = row_entropy / log(N)                (∈[0,1])
  - max_mass_mean   = mean over rows of max row-wise probability
  - effective_k     = exp(row_entropy)                       (fractional neighbor count)
  - above_uniform   = fraction of entries > 1/N

For GWNet:      A = softmax(ReLU(E1 E2^T))      dense
For AGCRN:      A = softmax(ReLU(E E^T))        dense (single embedding)
For MTGNNLight: A = row_norm(ReLU(cos_sim) ⊙ topk_mask)   sparse top-k
For MegaCRN:    A is meta-graph via memory prototypes — separate metric
For DCRNN/STGCN: fixed graph (distance-based) — reference baseline
For GRU/STAEformer: no explicit graph
"""

import json
import sys
from pathlib import Path
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATE_TAG = date.today().strftime("%Y-%m-%d")


def compute_adj_stats(A):
    """A: (N, N) tensor (should already be row-normalized and non-negative).
    Returns dict of scalar stats.
    """
    A = A.detach().cpu().float()
    N = A.shape[0]
    # ensure non-negative & row-normalized
    A = torch.clamp(A, min=0.0)
    row_sum = A.sum(-1, keepdim=True).clamp(min=1e-12)
    P = A / row_sum
    eps = 1e-12
    row_entropy = -(P * torch.log(P.clamp(min=eps))).sum(-1)    # (N,)
    normalized_entropy = row_entropy / np.log(max(N, 2))
    max_mass = P.max(-1).values
    effective_k = torch.exp(row_entropy)                         # fractional neighbors
    above_uniform = (P > (1.0 / N)).float().sum(-1)              # per-row
    # density of raw adjacency (fraction of non-zero entries)
    density = (A > 1e-8).float().mean().item()
    return {
        "num_nodes": N,
        "row_entropy_mean":      float(row_entropy.mean()),
        "row_entropy_std":       float(row_entropy.std()),
        "normalized_entropy_mean": float(normalized_entropy.mean()),
        "max_mass_mean":         float(max_mass.mean()),
        "max_mass_std":          float(max_mass.std()),
        "effective_k_mean":      float(effective_k.mean()),
        "above_uniform_mean":    float(above_uniform.mean()),
        "density":               density,
    }


def adj_from_gwnet(sd):
    E1 = sd["node_emb1"]
    E2 = sd["node_emb2"]
    A = torch.softmax(F.relu(E1 @ E2.t()), dim=-1)
    return A


def adj_from_agcrn(sd):
    E = sd["node_embeddings"]
    A = torch.softmax(F.relu(E @ E.t()), dim=-1)
    return A


def adj_from_mtgnn(sd, top_k=10):
    E = sd["node_emb"]
    e = F.normalize(E, p=2, dim=-1)
    sim = e @ e.t()
    N = sim.shape[0]
    k = min(top_k, N)
    _, topk_idx = sim.topk(k, dim=-1)
    mask = torch.zeros_like(sim).scatter_(-1, topk_idx, 1.0)
    adj = F.relu(sim) * mask
    adj = adj / adj.sum(-1, keepdim=True).clamp(min=1e-8)
    return adj


def adj_from_megacrn(sd):
    """MegaCRN uses a meta-graph through the memory module:
         We1, We2 project nodes into memory-prototype space.
       The effective adjacency can be approximated as softmax(We1 We2^T / sqrt(d)).
    """
    E1 = sd["memory.We1"]
    E2 = sd["memory.We2"]
    A = torch.softmax(E1 @ E2.t() / np.sqrt(E1.shape[1]), dim=-1)
    return A


def adj_from_fixed(path):
    """Load DCRNN/STGCN fixed adjacency for reference."""
    import pickle
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding='latin1')
    # obj is (sensor_ids, id_to_idx, adj_mx)
    if isinstance(obj, (tuple, list)) and len(obj) >= 3:
        A = obj[2]
    else:
        A = obj
    A = torch.tensor(np.asarray(A), dtype=torch.float32)
    # symmetric row-normalize
    D = A.sum(-1).clamp(min=1e-8)
    return A / D.unsqueeze(-1)


# fragility reference (from results in project_context.md / official_adaptive_results)
# Δ_sensor averaged across 3 seeds (30% random zero-mask, normalized)
FRAGILITY_REFERENCE = {
    "dcrnn":      +0.021,   # from memory
    "mtgnn":      +0.051,
    "stgcn":      +0.079,
    "gwnet":      +0.092,
    "gru":        +0.044,
    "agcrn":      +0.011,   # from official baselines (lower than DCRNN per claim-freeze memo)
    "megacrn":    -0.005,   # the anomalous negative delta
    "staeformer": +0.023,
}


MODEL_CONFIG = {
    "gwnet":   {"ckpt_dir": "baselines/stgnn", "pattern": "gwnet__metr-la__seed{seed}.pt",
                "adj_fn": "gwnet", "family": "dense-bilinear"},
    "mtgnn":   {"ckpt_dir": "baselines/stgnn", "pattern": "mtgnn__metr-la__seed{seed}.pt",
                "adj_fn": "mtgnn", "family": "sparse-topk"},
    "agcrn":   {"ckpt_dir": "baselines/stgnn", "pattern": "agcrn__metr-la__seed{seed}.pt",
                "adj_fn": "agcrn", "family": "dense-symmetric"},
    "megacrn": {"ckpt_dir": "baselines/stgnn_official_fix", "pattern": "megacrn__metr-la__seed{seed}.pt",
                "adj_fn": "megacrn", "family": "memory-meta"},
}


ADJ_FN_MAP = {
    "gwnet":   adj_from_gwnet,
    "mtgnn":   adj_from_mtgnn,
    "agcrn":   adj_from_agcrn,
    "megacrn": adj_from_megacrn,
}


def main():
    out_dir = REPO_ROOT / "results" / "routing_sparsity_spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model = {}
    for model_name, cfg in MODEL_CONFIG.items():
        print(f"\n=== {model_name.upper()} ({cfg['family']}) ===")
        per_seed = []
        for seed in [0, 1, 2]:
            path = REPO_ROOT / cfg["ckpt_dir"] / cfg["pattern"].format(seed=seed)
            if not path.exists():
                print(f"  seed{seed}: missing {path}")
                continue
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            A = ADJ_FN_MAP[cfg["adj_fn"]](ckpt["state_dict"])
            stats = compute_adj_stats(A)
            print(f"  seed{seed}: norm_H={stats['normalized_entropy_mean']:.4f}  "
                  f"max_mass={stats['max_mass_mean']:.4f}  "
                  f"eff_k={stats['effective_k_mean']:.2f}  "
                  f"above_1/N={stats['above_uniform_mean']:.2f}")
            per_seed.append(stats)
        if per_seed:
            # aggregate across seeds
            agg = {
                "family": cfg["family"],
                "normalized_entropy_mean": float(np.mean([s["normalized_entropy_mean"] for s in per_seed])),
                "normalized_entropy_std":  float(np.std([s["normalized_entropy_mean"] for s in per_seed])),
                "max_mass_mean":           float(np.mean([s["max_mass_mean"] for s in per_seed])),
                "effective_k_mean":        float(np.mean([s["effective_k_mean"] for s in per_seed])),
                "above_uniform_mean":      float(np.mean([s["above_uniform_mean"] for s in per_seed])),
                "density_mean":            float(np.mean([s["density"] for s in per_seed])),
                "per_seed":                per_seed,
            }
            agg["fragility_delta_sensor"] = FRAGILITY_REFERENCE.get(model_name)
            per_model[model_name] = agg

    # ---- Reference: DCRNN/STGCN fixed adjacency ----
    print(f"\n=== FIXED ADJACENCY (reference) ===")
    fixed_path = REPO_ROOT / "data" / "metr-la" / "adj_mx.pkl"
    if fixed_path.exists():
        A_fixed = adj_from_fixed(fixed_path)
        fixed_stats = compute_adj_stats(A_fixed)
        fixed_stats["family"] = "fixed-distance"
        fixed_stats["fragility_delta_sensor_dcrnn"] = FRAGILITY_REFERENCE["dcrnn"]
        fixed_stats["fragility_delta_sensor_stgcn"] = FRAGILITY_REFERENCE["stgcn"]
        print(f"  norm_H={fixed_stats['normalized_entropy_mean']:.4f}  "
              f"max_mass={fixed_stats['max_mass_mean']:.4f}  "
              f"eff_k={fixed_stats['effective_k_mean']:.2f}")
        per_model["fixed_adj"] = fixed_stats
    else:
        print(f"  (fixed adj file not found at {fixed_path})")

    # ---- Write output + correlation ----
    out_path = out_dir / f"ROUTING_SPARSITY_SPECTRUM_{DATE_TAG}.json"
    with open(out_path, "w") as f:
        json.dump(per_model, f, indent=2)
    print(f"\nWritten: {out_path}")

    # correlation: normalized entropy (density proxy) vs Δ_sensor across adaptive models
    adaptive_models = ["gwnet", "mtgnn", "agcrn", "megacrn"]
    H = [per_model[m]["normalized_entropy_mean"] for m in adaptive_models if m in per_model]
    D = [per_model[m]["fragility_delta_sensor"]  for m in adaptive_models if m in per_model]
    if len(H) == len(D) and len(H) >= 3:
        rho = np.corrcoef(H, D)[0, 1]
        print(f"\nPearson(norm_entropy, Δ_sensor) across adaptive models = {rho:+.4f}  (N={len(H)})")
        print("  (positive = more diffuse routing → higher fragility)")
        # also for max_mass
        M = [per_model[m]["max_mass_mean"] for m in adaptive_models if m in per_model]
        rho2 = np.corrcoef(M, D)[0, 1]
        print(f"Pearson(max_mass, Δ_sensor)     across adaptive models = {rho2:+.4f}")

    # pretty table
    print(f"\n{'='*90}\nROUTING-SPARSITY SPECTRUM TABLE\n{'='*90}")
    print(f"{'model':>12}  {'family':>18}  {'norm_H':>7}  {'max_mass':>9}  {'eff_k':>7}  {'Δ_sensor':>9}")
    print("-"*90)
    for m, s in per_model.items():
        if m == "fixed_adj":
            deltas = f"dcrnn{s.get('fragility_delta_sensor_dcrnn','-'):+.3f} / stgcn{s.get('fragility_delta_sensor_stgcn','-'):+.3f}"
            print(f"{m:>12}  {s['family']:>18}  {s['normalized_entropy_mean']:>7.4f}  "
                  f"{s['max_mass_mean']:>9.4f}  {s['effective_k_mean']:>7.2f}  {deltas}")
        else:
            d = s.get("fragility_delta_sensor")
            print(f"{m:>12}  {s['family']:>18}  {s['normalized_entropy_mean']:>7.4f}  "
                  f"{s['max_mass_mean']:>9.4f}  {s['effective_k_mean']:>7.2f}  "
                  f"{d:+.4f}" if d is not None else "n/a")

    print("\nRouting-sparsity spectrum done.")


if __name__ == "__main__":
    main()
