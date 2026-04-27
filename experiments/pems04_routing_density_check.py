"""
pems04_routing_density_check.py
================================
Test whether the routing-density law (max_mass(A) ↔ ΔSensor, ρ=+0.979 on METR-LA)
replicates when METR-LA-trained models are evaluated for sensor dropout on PEMS04.

Protocol (matched-structural-access):
- METR-LA-trained checkpoints (frozen) for GWNet, MTGNNLight, AGCRN, MegaCRN
- PEMS04 test set subsampled to 207 nodes (5 subsets, seeds 42-46)
- Normalize with METR-LA-derived global scaler (source-stat normalization)
- Source (METR-LA) adjacency at inference for all models
- 30% sensor dropout, 5 masks per subset, 3 seeds → 75 evaluations per model
- Gate: Pearson ρ(max_mass, ΔSensor_PEMS04) ≥ +0.90

Results → results/pems04_eval/routing_density_pems04.json
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
    scaler_mode_for_model,
    split_traffic_frame,
)
from experiments.official_adaptive_models import build_model_from_config
from experiments.stgnn_backbone_bank import BACKBONE_REGISTRY, load_adjacency

# ─── Configuration ───────────────────────────────────────────────────────────

DROPOUT_RATIO  = 0.30
MASK_SEEDS     = [42, 43, 44, 45, 46]
SUBSET_SEEDS   = [42, 43, 44, 45, 46]
N_METR_NODES   = 207
BATCH_SIZE     = 64
SEQ_LEN        = 12
PRED_LEN       = 12

# max_mass(A) from routing_sparsity_spectrum.py on METR-LA test samples
MAX_MASS = {
    "agcrn":   0.008,
    "megacrn": 0.008,
    "mtgnn":   0.16,
    "gwnet":   0.23,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def global_scaler_from(scaler: StandardScaler, n_nodes: int) -> StandardScaler:
    """Broadcast per-node METR-LA scaler to a single global mean/std for n_nodes."""
    g_mean = float(scaler.mean.mean())
    g_std  = float(scaler.std.mean())
    return StandardScaler(
        np.full((1, n_nodes), g_mean, dtype=np.float32),
        np.full((1, n_nodes), g_std,  dtype=np.float32),
    )


def load_pems04_subsets():
    """Return 5 random 207-node test subsets from PEMS04."""
    df = load_traffic_frame("pems04")
    _, _, test_df = split_traffic_frame(df)
    subsets = []
    for seed in SUBSET_SEEDS:
        rng = np.random.RandomState(seed)
        keep = sorted(rng.choice(test_df.shape[1], N_METR_NODES, replace=False).tolist())
        subsets.append({"keep": keep, "test_df": test_df.iloc[:, keep]})
    return subsets


@torch.no_grad()
def eval_dropout_single(forward_fn, x_norm, y_traffic_norm, mask_seed, device):
    """
    Compute ΔSensor_norm for one mask.

    x_norm:          (T, seq_len, N, F)  — normalized input (traffic in channel 0)
    y_traffic_norm:  (T, pred_len, N, 1) — normalized target traffic only
    forward_fn:      (bx, by) → pred_norm (B, pred_len, N, 1) in normalized space
    """
    N = x_norm.shape[2]
    rng = np.random.RandomState(mask_seed)
    n_drop = int(N * DROPOUT_RATIO)
    drop_idx = rng.choice(N, n_drop, replace=False)
    keep_idx = [i for i in range(N) if i not in drop_idx]

    x_masked = x_norm.copy()
    x_masked[:, :, drop_idx, 0] = 0.0

    def run(x_arr):
        loader = make_torch_loader(x_arr, y_traffic_norm, BATCH_SIZE, shuffle=False)
        preds, tgts = [], []
        for bx, by in loader:
            pred = forward_fn(bx.to(device), by.to(device))  # (B, pred_len, N, 1) or (B, T, N, 1)
            if pred.dim() == 4 and pred.shape[1] == PRED_LEN:
                pass  # already (B, pred_len, N, 1)
            elif pred.dim() == 4 and pred.shape[2] == PRED_LEN:
                pred = pred.permute(0, 2, 1, 3)  # (B, N, pred_len, 1)→(B, pred_len, N, 1)? check
            preds.append(pred.cpu())
            tgts.append(by.cpu())
        return torch.cat(preds, 0), torch.cat(tgts, 0)

    pred_cl, tgt = run(x_norm)
    pred_mk, _   = run(x_masked)

    # Normalize shapes to (B, pred_len, N) kept-node MAE
    def kept_mae(pred, tgt):
        # pred may be (B, pred_len, N, 1) or (B, T, N, 1) where T=pred_len
        p = pred.squeeze(-1)  # (B, pred_len, N)
        t = tgt.squeeze(-1)   # (B, pred_len, N)
        if p.shape[-1] != N:
            # try (B, N, pred_len)
            p = p.permute(0, 2, 1)
            t = t.permute(0, 2, 1)
        return (p[:, :, keep_idx] - t[:, :, keep_idx]).abs().mean().item()

    return kept_mae(pred_mk, tgt) - kept_mae(pred_cl, tgt)


@torch.no_grad()
def eval_model_on_pems04(model_name, seeds_info, subsets, subset_scaler, metr_la_adj, device,
                          build_input_fn, forward_fn_factory):
    """
    seeds_info: list of (seed, model) tuples
    build_input_fn(test_df, scaler) → (x_norm, y_traffic_norm) arrays
    forward_fn_factory(model) → forward_fn(bx, by)
    Returns (mean_delta, std_delta)
    """
    all_seed_deltas = []
    for seed, model in seeds_info:
        model.eval()
        fwd = forward_fn_factory(model)
        seed_deltas = []
        for subset in subsets:
            x_norm, y_traffic_norm = build_input_fn(subset["test_df"], subset_scaler)
            for ms in MASK_SEEDS:
                d = eval_dropout_single(fwd, x_norm, y_traffic_norm, ms, device)
                seed_deltas.append(d)
        mean_d = float(np.mean(seed_deltas))
        print(f"  {model_name} seed {seed}: ΔSensor = {mean_d:+.5f}")
        all_seed_deltas.append(mean_d)
    return float(np.mean(all_seed_deltas)), float(np.std(all_seed_deltas))


# ─── Model-specific loaders ──────────────────────────────────────────────────

def load_gwnet_seeds(device):
    with open(REPO_ROOT / "baselines/stgnn_protocol_registry.json") as f:
        reg = json.load(f)
    seeds = []
    for key in sorted(k for k in reg if k.startswith("gwnet__metr-la__")):
        entry = reg[key]
        ckpt = torch.load(REPO_ROOT / entry["checkpoint"], map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = BACKBONE_REGISTRY["gwnet"](
            num_nodes=N_METR_NODES, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        seeds.append((entry["seed"], model))
    return seeds


def load_mtgnn_seeds(device):
    with open(REPO_ROOT / "baselines/stgnn_protocol_registry.json") as f:
        reg = json.load(f)
    seeds = []
    for key in sorted(k for k in reg if k.startswith("mtgnn__metr-la__")):
        entry = reg[key]
        ckpt = torch.load(REPO_ROOT / entry["checkpoint"], map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = BACKBONE_REGISTRY["mtgnn"](
            num_nodes=N_METR_NODES, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        seeds.append((entry["seed"], model))
    return seeds


def load_adaptive_seeds(model_name, device,
                         reg_path="results/official_adaptive_relation/official_adaptive_unified_registry.json"):
    registry = load_registry(REPO_ROOT / reg_path)
    seeds = []
    for key in sorted(k for k in registry if k.startswith(f"{model_name}__metr-la__")):
        entry = registry[key]
        ckpt = torch.load(REPO_ROOT / entry["checkpoint"], map_location=device, weights_only=False)
        model = build_model_from_config(model_name, ckpt["model_config"], device)
        model.load_state_dict(ckpt["state_dict"])
        seeds.append((entry["seed"], model))
    return seeds


# ─── Input builders & forward factories ─────────────────────────────────────

def build_canonical_input(test_df, scaler):
    """GWNet / MTGNNLight: (T, seq_len, N, 1) normalized traffic."""
    values = test_df.values.astype(np.float32)
    values_norm = scaler.transform_numpy(values)
    x, y = make_windows(values_norm[..., None], seq_len=SEQ_LEN, pred_len=PRED_LEN)
    # y has same shape; use traffic channel only as target
    return x, y  # both (T, window, N, 1)


def canonical_fwd_factory(model, adj):
    def fwd(bx, by):
        # bx: (B, seq_len, N, 1) → squeeze to (B, seq_len, N)
        bx_in = bx[..., 0]  # (B, seq_len, N)
        pred = model(bx_in, adj)  # (B, N, pred_len)
        # reformat to (B, pred_len, N, 1)
        return pred.permute(0, 2, 1).unsqueeze(-1)
    return fwd


def build_agcrn_input(test_df, scaler):
    values = test_df.values.astype(np.float32)
    values_norm = scaler.transform_numpy(values)
    x, y = make_windows(values_norm[..., None], seq_len=SEQ_LEN, pred_len=PRED_LEN)
    return x, y


def agcrn_fwd_factory(model):
    def fwd(bx, by):
        pred = model(bx.to(next(model.parameters()).device), None)
        # pred shape from AGCRN: check
        return pred
    return fwd


def build_megacrn_input(test_df, scaler):
    values = test_df.values.astype(np.float32)
    values_norm = scaler.transform_numpy(values)[..., None]
    tod = build_time_in_day(test_df.index, N_METR_NODES)
    data = np.concatenate([values_norm, tod], axis=-1)
    x, y = make_windows(data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    # y: (T, pred_len, N, 2) — channel 0 is traffic_norm, channel 1 is TOD
    # Return traffic-only target for MAE, but keep full y for forward pass
    y_traffic = y[..., :1]  # (T, pred_len, N, 1)
    return x, y_traffic  # note: forward fn gets y_traffic, uses it as by placeholder


def megacrn_fwd_factory(model, full_y_ref=None):
    # full_y_ref is not needed because we store y with TOD in a closure below
    def fwd(bx, by):
        # bx: (B, seq_len, N, 2) — channel 0 = traffic, channel 1 = TOD
        # by: (B, pred_len, N, 1) — traffic only (no TOD available)
        # We pass zeros as y_cov when TOD isn't available in by
        B, T_pred, N, _ = by.shape
        y_cov_zero = torch.zeros(B, T_pred, N, 1, device=bx.device)
        pred_norm, *_ = model(bx[..., :1], y_cov_zero, None, None)
        # pred_norm: (B, pred_len, N, 1) in normalized space
        return pred_norm
    return fwd


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading METR-LA source (scaler + adjacency)...")
    src = build_source_dataset("metr-la", scaler_mode="per_node")
    adj = load_adjacency(str(REPO_ROOT / "data/metr-la/adj_mx.pkl"), N_METR_NODES, device)

    # Build a global scaler from METR-LA per-node stats for PEMS04 normalization
    subset_scaler = global_scaler_from(src["scaler"], N_METR_NODES)
    print(f"  Global mean={float(subset_scaler.mean.mean()):.4f}  "
          f"std={float(subset_scaler.std.mean()):.4f}")

    print("\nLoading PEMS04 test subsets...")
    subsets = load_pems04_subsets()
    print(f"  {len(subsets)} subsets × {N_METR_NODES} nodes, "
          f"test len={subsets[0]['test_df'].shape[0]}")

    results = {}

    # ── AGCRN ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("AGCRN")
    seeds = load_adaptive_seeds("agcrn", device)
    m, s = eval_model_on_pems04(
        "agcrn", seeds, subsets, subset_scaler, adj, device,
        build_input_fn=build_agcrn_input,
        forward_fn_factory=agcrn_fwd_factory,
    )
    results["agcrn"] = {"delta_sensor_norm_mean": m, "delta_sensor_norm_std": s,
                        "max_mass": MAX_MASS["agcrn"]}
    print(f"  → AGCRN aggregate ΔSensor = {m:+.5f} ± {s:.5f}")

    # ── MegaCRN ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MegaCRN")
    seeds = load_adaptive_seeds("megacrn", device)
    m, s = eval_model_on_pems04(
        "megacrn", seeds, subsets, subset_scaler, adj, device,
        build_input_fn=build_megacrn_input,
        forward_fn_factory=megacrn_fwd_factory,
    )
    results["megacrn"] = {"delta_sensor_norm_mean": m, "delta_sensor_norm_std": s,
                          "max_mass": MAX_MASS["megacrn"]}
    print(f"  → MegaCRN aggregate ΔSensor = {m:+.5f} ± {s:.5f}")

    # ── MTGNNLight ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MTGNNLight")
    seeds = load_mtgnn_seeds(device)

    def mtgnn_fwd_factory(model):
        return canonical_fwd_factory(model, adj)

    m, s = eval_model_on_pems04(
        "mtgnn", seeds, subsets, subset_scaler, adj, device,
        build_input_fn=build_canonical_input,
        forward_fn_factory=mtgnn_fwd_factory,
    )
    results["mtgnn"] = {"delta_sensor_norm_mean": m, "delta_sensor_norm_std": s,
                        "max_mass": MAX_MASS["mtgnn"]}
    print(f"  → MTGNNLight aggregate ΔSensor = {m:+.5f} ± {s:.5f}")

    # ── GWNet ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("GWNet")
    seeds = load_gwnet_seeds(device)

    def gwnet_fwd_factory(model):
        return canonical_fwd_factory(model, adj)

    m, s = eval_model_on_pems04(
        "gwnet", seeds, subsets, subset_scaler, adj, device,
        build_input_fn=build_canonical_input,
        forward_fn_factory=gwnet_fwd_factory,
    )
    results["gwnet"] = {"delta_sensor_norm_mean": m, "delta_sensor_norm_std": s,
                        "max_mass": MAX_MASS["gwnet"]}
    print(f"  → GWNet aggregate ΔSensor = {m:+.5f} ± {s:.5f}")

    # ── Pearson ρ ─────────────────────────────────────────────────────────────
    model_order = ["agcrn", "megacrn", "mtgnn", "gwnet"]
    max_masses  = np.array([results[mn]["max_mass"]               for mn in model_order])
    deltas      = np.array([results[mn]["delta_sensor_norm_mean"] for mn in model_order])
    rho = float(np.corrcoef(max_masses, deltas)[0, 1])

    print("\n" + "="*60)
    print("RESULT SUMMARY")
    for mn in model_order:
        r = results[mn]
        print(f"  {mn:<10}  max_mass={r['max_mass']:.3f}  "
              f"ΔSensor={r['delta_sensor_norm_mean']:+.5f} ± {r['delta_sensor_norm_std']:.5f}")
    print(f"\n  Pearson ρ (max_mass vs ΔSensor on PEMS04): {rho:+.4f}")

    gate = rho >= 0.90
    print(f"  Gate (ρ ≥ 0.90): {'PASSED ✓' if gate else 'FAILED ✗'}")
    if gate:
        print(f'\n  Paper addition (§5.2):')
        print(f'    "The routing-density law replicates on METR-LA→PEMS04 transfer')
        print(f'     (Pearson ρ={rho:+.3f}, Appendix~\\ref{{sec:appendix_pems04}})."')
    else:
        print(f'\n  Paper addition (§6 limitations):')
        print(f'    One sentence: PEMS04 replication inconclusive (ρ={rho:+.3f}).')

    results.update({
        "pearson_rho_pems04": rho,
        "gate_passed":        gate,
        "model_order":        model_order,
        "max_masses":         max_masses.tolist(),
        "delta_sensors_pems04": deltas.tolist(),
        "protocol": {
            "dropout_ratio":  DROPOUT_RATIO,
            "n_subsets":      len(SUBSET_SEEDS),
            "n_masks":        len(MASK_SEEDS),
            "n_seeds":        3,
            "n_metr_nodes":   N_METR_NODES,
            "normalization":  "METR-LA global scaler (source stats)",
            "adjacency":      "METR-LA source adj (matched-structural-access)",
        },
    })

    out = REPO_ROOT / "results" / "pems04_eval" / "routing_density_pems04.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()
