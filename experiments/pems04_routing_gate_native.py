"""
pems04_routing_gate_native.py
==============================
Two-city replication of the routing-density law on PEMS04.

Unlike pems04_routing_gate_fast.py (which used METR-LA-trained zero-shot models on
PEMS04 — the wrong test, dominated by distribution shift), this script uses
**PEMS04-trained** native checkpoints for the four adaptive backbones:

    AGCRN, MegaCRN, MTGNN, GWNet

For each backbone:
  1. max_mass(A) is computed from each PEMS04 checkpoint via the canonical
     adjacency extractors in experiments/routing_sparsity_spectrum.py
     (NO hard-coded values).
  2. ΔSensor is computed on PEMS04 test data with the PEMS04-trained adjacency
     and PEMS04 train-statistics scaler, 3 seeds × 5 mask seeds × 30% zero-mask
     (same protocol as pems04_routing_gate_fast.py, but native).
  3. Both metrics are averaged across 3 seeds.

Pearson ρ(max_mass, ΔSensor) is then computed across the 4 models.

Gate thresholds (paper §5.3 update path):
  ρ ≥ +0.85  → promote to "two-city law" in main text
  0 ≤ ρ < 0.85 → footnote + appendix table only
  ρ < 0      → §6 limitations caveat

Output: results/polish_v15/pems04_native_routing_gate.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.official_adaptive_common import (
    StandardScaler,
    build_source_dataset,
    build_time_in_day,
    make_torch_loader,
    make_windows,
    scalar_source_scaler,
)
from experiments.official_adaptive_models import build_model_from_config
from experiments.routing_sparsity_spectrum import (
    adj_from_agcrn,
    adj_from_gwnet,
    adj_from_megacrn,
    adj_from_mtgnn,
    compute_adj_stats,
)
from experiments.stgnn_backbone_bank import BACKBONE_REGISTRY, load_adjacency

# ── Protocol constants ────────────────────────────────────────────────────────
N_NODES       = 307            # PEMS04
SEEDS         = [0, 1, 2]
DROPOUT_RATIO = 0.30
MASK_SEEDS    = [42, 43, 44, 45, 46]
BATCH_SIZE    = 64
SEQ_LEN       = 12
PRED_LEN      = 12
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATTERNS = {
    "agcrn":   "baselines/stgnn/agcrn__pems04__seed{seed}.pt",
    "megacrn": "baselines/stgnn/megacrn__pems04__seed{seed}.pt",
    "mtgnn":   "baselines/stgnn_pems04_extra/mtgnn__pems04__seed{seed}.pt",
    "gwnet":   "baselines/stgnn/gwnet__pems04__seed{seed}.pt",
}

ADJ_FN = {
    "agcrn":   adj_from_agcrn,
    "megacrn": adj_from_megacrn,
    "mtgnn":   adj_from_mtgnn,
    "gwnet":   adj_from_gwnet,
}


# ── max_mass computation ──────────────────────────────────────────────────────
def compute_max_mass(model_name: str, seed: int) -> float:
    """Compute max_mass(A) from a PEMS04 checkpoint."""
    path = REPO_ROOT / CKPT_PATTERNS[model_name].format(seed=seed)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    A = ADJ_FN[model_name](ckpt["state_dict"])
    stats = compute_adj_stats(A)
    return float(stats["max_mass_mean"])


# ── Per-model forward + arrays ────────────────────────────────────────────────
def make_per_node_scaler(ckpt) -> StandardScaler:
    """Reconstruct the StandardScaler from saved scaler_mean / scaler_std (AGCRN/MegaCRN)."""
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    std  = np.asarray(ckpt["scaler_std"],  dtype=np.float32)
    if mean.ndim == 1:
        mean = mean[None, :]
        std  = std[None, :]
    return StandardScaler(mean, std)


def build_eval_arrays(model_name: str, ckpt, test_df_norm: np.ndarray, test_index):
    """Return (x, y) windows with the input layout each model expects.
    test_df_norm: (T, N) already normalized with model-specific scaler
    """
    if model_name == "agcrn":
        return make_windows(test_df_norm[..., None], SEQ_LEN, PRED_LEN)
    if model_name == "megacrn":
        tod = build_time_in_day(test_index, N_NODES)
        data = np.concatenate([test_df_norm[..., None], tod], axis=-1)
        x, y_full = make_windows(data, SEQ_LEN, PRED_LEN)
        return x, y_full[..., :1]
    # GWNet / MTGNN: single channel
    return make_windows(test_df_norm[..., None], SEQ_LEN, PRED_LEN)


def build_model_and_fwd(model_name: str, seed: int, adj: torch.Tensor):
    """Load checkpoint, return (forward_callable, scaler, source_used).
    forward_callable(bx, by) returns predictions in normalized space, shape (B, T, N, 1).
    """
    path = REPO_ROOT / CKPT_PATTERNS[model_name].format(seed=seed)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    if model_name == "agcrn":
        scaler = make_per_node_scaler(ckpt)
        model = build_model_from_config("agcrn", ckpt["model_config"], DEVICE)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        def fwd(bx, by):
            return model(bx, None)
        return fwd, scaler

    if model_name == "megacrn":
        scaler = make_per_node_scaler(ckpt)
        model = build_model_from_config("megacrn", ckpt["model_config"], DEVICE)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        def fwd(bx, by):
            B, T, N, _ = by.shape
            y_cov = torch.zeros(B, T, N, 1, device=bx.device)
            pred, *_ = model(bx[..., :1], y_cov, None, None)
            return pred
        return fwd, scaler

    if model_name == "mtgnn":
        # PEMS04 train-statistics scaler (not in ckpt for backbone-bank models)
        scaler = _pems04_per_node_scaler()
        cfg = ckpt["config"]
        model = BACKBONE_REGISTRY["mtgnn"](
            num_nodes=N_NODES, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        def fwd(bx, by):
            bx_in = bx[..., 0]
            pred = model(bx_in, adj)
            return pred.permute(0, 2, 1).unsqueeze(-1)
        return fwd, scaler

    if model_name == "gwnet":
        scaler = _pems04_per_node_scaler()
        cfg = ckpt["config"]
        model = BACKBONE_REGISTRY["gwnet"](
            num_nodes=N_NODES, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        def fwd(bx, by):
            bx_in = bx[..., 0]
            pred = model(bx_in, adj)
            return pred.permute(0, 2, 1).unsqueeze(-1)
        return fwd, scaler

    raise ValueError(f"unknown model {model_name}")


_PEMS04_PER_NODE_SCALER = None
def _pems04_per_node_scaler() -> StandardScaler:
    global _PEMS04_PER_NODE_SCALER
    if _PEMS04_PER_NODE_SCALER is None:
        src = build_source_dataset("pems04", scaler_mode="per_node")
        _PEMS04_PER_NODE_SCALER = src["scaler"]
    return _PEMS04_PER_NODE_SCALER


# ── ΔSensor computation ───────────────────────────────────────────────────────
@torch.no_grad()
def delta_sensor_seed(model_name: str, seed: int, adj: torch.Tensor, test_df) -> float:
    """ΔSensor for one (model, seed): mean over 5 mask seeds."""
    fwd, scaler = build_model_and_fwd(model_name, seed, adj)
    # Build (T, N) normalized PEMS04 test
    values = test_df.values.astype(np.float32)
    values_norm = scaler.transform_numpy(values)
    x_all, y_all = build_eval_arrays(model_name, None, values_norm, test_df.index)
    x_all = torch.FloatTensor(x_all)
    y_all = torch.FloatTensor(y_all)

    deltas = []
    for ms in MASK_SEEDS:
        d = _delta_sensor_one_mask(fwd, x_all, y_all, ms)
        deltas.append(d)
    return float(np.mean(deltas)), float(np.std(deltas)), deltas


@torch.no_grad()
def _delta_sensor_one_mask(fwd, x_all: torch.Tensor, y_all: torch.Tensor, mask_seed: int) -> float:
    N = x_all.shape[2]
    rng = np.random.RandomState(mask_seed)
    drop = rng.choice(N, int(N * DROPOUT_RATIO), replace=False)
    keep = [i for i in range(N) if i not in drop]

    x_mask = x_all.clone()
    x_mask[:, :, drop, 0] = 0.0

    def run(x_arr):
        loader = make_torch_loader(x_arr.numpy(), y_all.numpy(), BATCH_SIZE, shuffle=False)
        preds, tgts = [], []
        for bx, by in loader:
            bx = bx.to(DEVICE); by = by.to(DEVICE)
            out = fwd(bx, by)
            preds.append(out.cpu()); tgts.append(by.cpu())
        return torch.cat(preds), torch.cat(tgts)

    pred_clean, tgt = run(x_all)
    pred_mask, _   = run(x_mask)
    p_cl = pred_clean.squeeze(-1)[:, :, keep]
    p_mk = pred_mask.squeeze(-1)[:,  :, keep]
    t    = tgt.squeeze(-1)[:, :, keep]
    return float((p_mk - t).abs().mean() - (p_cl - t).abs().mean())


# ── Driver ────────────────────────────────────────────────────────────────────
def main():
    print(f"DEVICE = {DEVICE}")
    print("Loading PEMS04 test split + adjacency...")
    src = build_source_dataset("pems04", scaler_mode="per_node")
    test_df = src["test"]
    print(f"  test_df: {test_df.shape}")
    adj = load_adjacency(str(REPO_ROOT / "data/pems04/adj_mx.pkl"), N_NODES, DEVICE)
    print(f"  adj: {tuple(adj.shape)}")

    per_model = {}
    for model_name in ["agcrn", "megacrn", "mtgnn", "gwnet"]:
        print(f"\n=== {model_name.upper()} ===")
        # max_mass per seed → mean ± std
        mm_per_seed = []
        for seed in SEEDS:
            try:
                mm = compute_max_mass(model_name, seed)
                mm_per_seed.append(mm)
                print(f"  seed{seed}: max_mass = {mm:.5f}")
            except FileNotFoundError as e:
                print(f"  seed{seed}: missing checkpoint {e}")
        if not mm_per_seed:
            continue
        mm_mean = float(np.mean(mm_per_seed))
        mm_std  = float(np.std(mm_per_seed))

        # ΔSensor per seed → mean ± std (5 masks each)
        ds_per_seed = []
        ds_per_seed_full = []
        for seed in SEEDS:
            try:
                d_mean, d_std, deltas = delta_sensor_seed(model_name, seed, adj, test_df)
                ds_per_seed.append(d_mean)
                ds_per_seed_full.append({"seed": seed, "mean": d_mean, "std": d_std,
                                         "deltas_per_mask": deltas})
                print(f"  seed{seed}: ΔSensor = {d_mean:+.5f} ± {d_std:.5f}")
            except FileNotFoundError as e:
                print(f"  seed{seed}: missing checkpoint {e}")
        if not ds_per_seed:
            continue
        ds_mean = float(np.mean(ds_per_seed))
        ds_std  = float(np.std(ds_per_seed))

        per_model[model_name] = {
            "max_mass_per_seed": mm_per_seed,
            "max_mass_mean":     mm_mean,
            "max_mass_std":      mm_std,
            "delta_sensor_per_seed": ds_per_seed_full,
            "delta_sensor_mean": ds_mean,
            "delta_sensor_std":  ds_std,
        }

    # ── Pearson ρ ────────────────────────────────────────────────────────────
    order   = [m for m in ["agcrn", "megacrn", "mtgnn", "gwnet"] if m in per_model]
    masses  = np.array([per_model[m]["max_mass_mean"]    for m in order])
    deltas  = np.array([per_model[m]["delta_sensor_mean"] for m in order])
    if len(order) >= 2 and np.std(masses) > 0 and np.std(deltas) > 0:
        rho = float(np.corrcoef(masses, deltas)[0, 1])
    else:
        rho = float("nan")

    if rho >= 0.85:
        verdict = "PROMOTE: two-city law (main text)"
    elif rho >= 0.0:
        verdict = "PARTIAL: footnote + appendix table"
    else:
        verdict = "FAIL: §6 limitations caveat"

    print(f"\n{'='*60}")
    print(f"Pearson ρ(max_mass, ΔSensor) on PEMS04 = {rho:+.4f}")
    print(f"Verdict: {verdict}")
    print(f"\nPer-model:")
    for m in order:
        r = per_model[m]
        print(f"  {m:<10}  max_mass={r['max_mass_mean']:.4f}±{r['max_mass_std']:.4f}  "
              f"ΔSensor={r['delta_sensor_mean']:+.5f}±{r['delta_sensor_std']:.5f}")

    out = {
        "pearson_rho_pems04_native": rho,
        "gate_verdict":               verdict,
        "model_order":                order,
        "max_masses":                 masses.tolist(),
        "delta_sensors":              deltas.tolist(),
        "per_model":                  per_model,
        "protocol": {
            "dataset":       "PEMS04 (native checkpoints)",
            "n_nodes":       N_NODES,
            "n_seeds":       len(SEEDS),
            "n_mask_seeds":  len(MASK_SEEDS),
            "dropout_ratio": DROPOUT_RATIO,
            "device":        str(DEVICE),
            "note":          "Full PEMS04 test split (no window subsample), 3 model seeds, 5 mask seeds. "
                             "max_mass extracted per checkpoint via routing_sparsity_spectrum extractors. "
                             "ΔSensor uses PEMS04 train-statistics scaler and PEMS04 adjacency.",
        },
    }

    out_path = REPO_ROOT / "results" / "polish_v15" / "pems04_native_routing_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
