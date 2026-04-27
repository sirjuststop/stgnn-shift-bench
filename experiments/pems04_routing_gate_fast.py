"""
pems04_routing_gate_fast.py
============================
Fast gate check: does max_mass(A) still rank ΔSensor on PEMS04?
Uses CPU, 256-window subsample, 1 seed per model, 5 masks.
Full eval (pems04_routing_density_check.py) should confirm later.

Gate: Pearson ρ(max_mass, ΔSensor_PEMS04) ≥ +0.90
→ if PASSED: add one sentence to §5.2 with approximate ρ; run full eval for exact value
→ if FAILED: add one sentence to §6 limitations

Output: results/pems04_eval/routing_gate_fast.json
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
    load_registry,
    load_traffic_frame,
    make_torch_loader,
    make_windows,
    build_time_in_day,
    split_traffic_frame,
)
from experiments.official_adaptive_models import build_model_from_config
from experiments.stgnn_backbone_bank import BACKBONE_REGISTRY, load_adjacency

DROPOUT_RATIO = 0.30
MASK_SEEDS    = [42, 43, 44, 45, 46]
N_NODES       = 207
BATCH_SIZE    = 64
N_WINDOWS     = 256   # subsample for speed
SEQ_LEN       = 12
PRED_LEN      = 12
SUBSET_SEED   = 42    # single PEMS04 subset

MAX_MASS = {"agcrn": 0.008, "megacrn": 0.008, "mtgnn": 0.16, "gwnet": 0.23}
DEVICE   = torch.device("cpu")  # GPU occupied by training jobs


def global_scaler(src_scaler, n_nodes):
    g = float(src_scaler.mean.mean())
    s = float(src_scaler.std.mean())
    return StandardScaler(
        np.full((1, n_nodes), g, dtype=np.float32),
        np.full((1, n_nodes), s, dtype=np.float32))


def pems04_subset():
    df = load_traffic_frame("pems04")
    _, _, test_df = split_traffic_frame(df)
    rng = np.random.RandomState(SUBSET_SEED)
    keep = sorted(rng.choice(test_df.shape[1], N_NODES, replace=False).tolist())
    return test_df.iloc[:, keep]


@torch.no_grad()
def delta_sensor(fwd_fn, x_all, y_all, mask_seed):
    """ΔSensor on a random subsample of N_WINDOWS windows."""
    rng0 = np.random.RandomState(mask_seed + 100)
    idx = rng0.choice(len(x_all), N_WINDOWS, replace=False)
    x_sub = x_all[idx]
    y_sub = y_all[idx]

    N = x_sub.shape[2]
    rng = np.random.RandomState(mask_seed)
    drop = rng.choice(N, int(N * DROPOUT_RATIO), replace=False)
    keep = [i for i in range(N) if i not in drop]

    x_mask = x_sub.copy()
    x_mask[:, :, drop, 0] = 0.0

    def mae_kept(pred, tgt):
        # pred: (B, pred_len, N, 1), tgt: (B, pred_len, N, 1)
        p = pred.squeeze(-1)  # (B, pred_len, N)
        t = tgt.squeeze(-1)
        return float((p[:, :, keep] - t[:, :, keep]).abs().mean())

    def run_loader(x_arr, y_arr):
        loader = make_torch_loader(x_arr, y_arr, BATCH_SIZE, shuffle=False)
        preds, tgts = [], []
        for bx, by in loader:
            out = fwd_fn(bx, by)
            preds.append(out.cpu())
            tgts.append(by.cpu())
        return torch.cat(preds), torch.cat(tgts)

    pred_cl, tgt = run_loader(x_sub,    y_sub)
    pred_mk, _   = run_loader(x_mask,   y_sub)
    return mae_kept(pred_mk, tgt) - mae_kept(pred_cl, tgt)


def eval_model(name, fwd_fn_factory, x_all, y_all):
    deltas = [delta_sensor(fwd_fn_factory(), x_all, y_all, ms) for ms in MASK_SEEDS]
    m = float(np.mean(deltas))
    print(f"  {name}: ΔSensor = {m:+.5f}  (masks: {[f'{d:+.4f}' for d in deltas]})")
    return m


def main():
    print("Loading METR-LA source...")
    src = build_source_dataset("metr-la", scaler_mode="per_node")
    adj = load_adjacency(str(REPO_ROOT / "data/metr-la/adj_mx.pkl"), N_NODES, DEVICE)
    scaler = global_scaler(src["scaler"], N_NODES)

    print("Loading PEMS04 subset...")
    test_df = pems04_subset()
    print(f"  Subset: {test_df.shape} → {N_WINDOWS} windows sampled per mask")

    results = {}

    # ── AGCRN ────────────────────────────────────────────────────────────────
    print("\nAGCRN")
    registry = load_registry(REPO_ROOT /
        "results/official_adaptive_relation/official_adaptive_unified_registry.json")
    entry = registry["agcrn__metr-la__seed0"]
    ckpt = torch.load(REPO_ROOT / entry["checkpoint"], map_location=DEVICE, weights_only=False)
    model_a = build_model_from_config("agcrn", ckpt["model_config"], DEVICE)
    model_a.load_state_dict(ckpt["state_dict"]); model_a.eval()

    v = scaler.transform_numpy(test_df.values.astype(np.float32))
    x_a, y_a = make_windows(v[..., None], SEQ_LEN, PRED_LEN)

    def agcrn_fwd():
        def f(bx, by):
            return model_a(bx, None)
        return f

    results["agcrn"] = eval_model("agcrn", agcrn_fwd, x_a, y_a)

    # ── MegaCRN ──────────────────────────────────────────────────────────────
    print("\nMegaCRN")
    entry = registry["megacrn__metr-la__seed0"]
    ckpt = torch.load(REPO_ROOT / entry["checkpoint"], map_location=DEVICE, weights_only=False)
    model_m = build_model_from_config("megacrn", ckpt["model_config"], DEVICE)
    model_m.load_state_dict(ckpt["state_dict"]); model_m.eval()

    v_norm = scaler.transform_numpy(test_df.values.astype(np.float32))[..., None]
    tod    = build_time_in_day(test_df.index, N_NODES)
    data_m = np.concatenate([v_norm, tod], axis=-1)
    x_m, y_full = make_windows(data_m, SEQ_LEN, PRED_LEN)
    y_m = y_full[..., :1]  # traffic channel only for target

    def megacrn_fwd():
        def f(bx, by):
            B, T, N, _ = by.shape
            y_cov = torch.zeros(B, T, N, 1, device=bx.device)
            pred, *_ = model_m(bx[..., :1], y_cov, None, None)
            return pred
        return f

    results["megacrn"] = eval_model("megacrn", megacrn_fwd, x_m, y_m)

    # ── MTGNNLight ───────────────────────────────────────────────────────────
    print("\nMTGNNLight")
    with open(REPO_ROOT / "baselines/stgnn_protocol_registry.json") as f:
        breg = json.load(f)
    entry_mt = breg["mtgnn__metr-la__seed0"]
    ckpt = torch.load(REPO_ROOT / entry_mt["checkpoint"], map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model_mt = BACKBONE_REGISTRY["mtgnn"](
        num_nodes=N_NODES, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
    model_mt.load_state_dict(ckpt["state_dict"]); model_mt.eval()

    def mtgnn_fwd():
        def f(bx, by):
            bx_in = bx[..., 0]
            pred = model_mt(bx_in, adj)
            return pred.permute(0, 2, 1).unsqueeze(-1)
        return f

    results["mtgnn"] = eval_model("mtgnn", mtgnn_fwd, x_a, y_a)  # same arrays as agcrn

    # ── GWNet ────────────────────────────────────────────────────────────────
    print("\nGWNet")
    entry_gw = breg["gwnet__metr-la__seed0"]
    ckpt = torch.load(REPO_ROOT / entry_gw["checkpoint"], map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model_gw = BACKBONE_REGISTRY["gwnet"](
        num_nodes=N_NODES, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
    model_gw.load_state_dict(ckpt["state_dict"]); model_gw.eval()

    def gwnet_fwd():
        def f(bx, by):
            bx_in = bx[..., 0]
            pred = model_gw(bx_in, adj)
            return pred.permute(0, 2, 1).unsqueeze(-1)
        return f

    results["gwnet"] = eval_model("gwnet", gwnet_fwd, x_a, y_a)

    # ── Pearson ρ ─────────────────────────────────────────────────────────────
    order = ["agcrn", "megacrn", "mtgnn", "gwnet"]
    masses = np.array([MAX_MASS[m] for m in order])
    deltas = np.array([results[m] for m in order])
    rho = float(np.corrcoef(masses, deltas)[0, 1])

    print(f"\n{'='*55}")
    print(f"Pearson ρ (max_mass vs ΔSensor on PEMS04): {rho:+.4f}")
    print(f"Gate (ρ ≥ 0.90): {'PASSED ✓' if rho >= 0.90 else 'FAILED ✗'}")
    print(f"\nRanking check:")
    for mn in order:
        print(f"  {mn:<10}  max_mass={MAX_MASS[mn]:.3f}  ΔSensor={results[mn]:+.5f}")

    out = {
        "pearson_rho_pems04": rho, "gate_passed": rho >= 0.90,
        "model_order": order, "max_masses": masses.tolist(),
        "delta_sensors_pems04": deltas.tolist(),
        "per_model": {m: {"delta": results[m], "max_mass": MAX_MASS[m]} for m in order},
        "note": f"Fast gate check: 1 seed, 1 PEMS04 subset (seed={SUBSET_SEED}), "
                f"5 masks, {N_WINDOWS} windows subsampled each mask, CPU only",
    }

    out_path = REPO_ROOT / "results" / "pems04_eval" / "routing_gate_fast.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
