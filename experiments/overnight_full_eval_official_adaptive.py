"""
overnight_full_eval_official_adaptive.py
=========================================
Complete overnight evaluation pass for AGCRN / MegaCRN / STAEformer.
Produces paper-facing, unit-clean results for all 9 checkpoints.

Evaluation axes
---------------
A. Clean evaluation  — masked MAE in mph (paper-facing), normalized MAE (internal)
B. Sensor dropout    — severity curve 10/20/30/40/50%; 5 mask seeds; delta in both
                       normalized-kept and denormalized-masked spaces
C. Cross-city        — source-stat normalised + target-stat masked-mph; 5 subsets
D. Temporal subgroup — weekday / weekend MAE (normalized + abs-masked); per-horizon gap

All results are written to results/official_adaptive_relation/
with date-stamped filenames.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.official_adaptive_common import (
    StandardScaler,
    build_agcrn_arrays,
    build_megacrn_arrays,
    build_staeformer_arrays,
    build_source_dataset,
    compute_masked_metrics,
    fit_train_scaler,
    load_registry,
    load_traffic_frame,
    make_torch_loader,
    sample_target_subsets,
    scaler_mode_for_model,
    set_seed,
    split_traffic_frame,
)
from experiments.official_adaptive_models import build_model_from_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATE_TAG       = date.today().strftime("%Y-%m-%d")
MASK_SEEDS     = [42, 43, 44, 45, 46]
SEVERITY_PTS   = [0.10, 0.20, 0.30, 0.40, 0.50]
N_SUBSETS      = 5
SUBSET_SEEDS   = 42           # base seed; sample_target_subsets increments to 42..46
BATCH_SIZE     = 64
PRED_LEN       = 12           # horizons 1..12
SEQ_LEN        = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_arrays(model_name, frame, scaler):
    if model_name == "agcrn":
        return build_agcrn_arrays(frame, scaler, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    if model_name == "megacrn":
        return build_megacrn_arrays(frame, scaler, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    if model_name == "staeformer":
        return build_staeformer_arrays(frame, scaler, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    raise KeyError(model_name)


@torch.no_grad()
def forward_batch(model_name, model, bx, by, device):
    bx, by = bx.to(device), by.to(device)
    if model_name == "agcrn":
        return model(bx, None)
    if model_name == "megacrn":
        pred_norm, *_ = model(bx[..., :1], by[..., 1:], None, None)
        return pred_norm
    if model_name == "staeformer":
        return model(bx)
    raise KeyError(model_name)


@torch.no_grad()
def collect_preds(model_name, model, loader, scaler, device):
    """Return (pred_norm, target_norm, pred_raw, target_raw) concatenated tensors."""
    pn_list, tn_list, pr_list, tr_list = [], [], [], []
    for bx, by in loader:
        pred_norm = forward_batch(model_name, model, bx, by, device)
        # resolve targets
        if model_name == "staeformer":
            target_raw  = by.to(device)
            target_norm = scaler.transform_torch(target_raw)
        elif model_name == "megacrn":
            target_norm = by[..., :1].to(device)
            target_raw  = scaler.inverse_transform_torch(target_norm)
        else:  # agcrn
            target_norm = by.to(device)
            target_raw  = scaler.inverse_transform_torch(target_norm)
        pred_raw = scaler.inverse_transform_torch(pred_norm)
        pn_list.append(pred_norm.cpu()); tn_list.append(target_norm.cpu())
        pr_list.append(pred_raw.cpu());  tr_list.append(target_raw.cpu())
    return (torch.cat(pn_list), torch.cat(tn_list),
            torch.cat(pr_list), torch.cat(tr_list))


def norm_mae(pred, tgt):
    return float((pred - tgt).abs().mean())


def masked_mae_np(pred_raw, target_raw, mask_value=0.0):
    """Compute masked MAE from numpy arrays."""
    mask = target_raw > mask_value
    if mask.sum() == 0:
        return float("nan")
    return float(np.abs(pred_raw[mask] - target_raw[mask]).mean())


# ---------------------------------------------------------------------------
# A. Clean evaluation
# ---------------------------------------------------------------------------
def eval_clean(model_name, model, x_test, y_test, scaler, device):
    loader = make_torch_loader(x_test, y_test, BATCH_SIZE, shuffle=False)
    pn, tn, pr, tr = collect_preds(model_name, model, loader, scaler, device)

    clean_norm_mae      = norm_mae(pn, tn)
    clean_denorm_mae    = norm_mae(pr, tr)
    masked_m            = compute_masked_metrics(pr, tr)
    clean_denorm_masked = masked_m["MAE"]
    clean_denorm_rmse_masked = masked_m["RMSE"]

    return {
        "clean_norm_MAE":            clean_norm_mae,         # internal only
        "clean_denorm_MAE":          clean_denorm_mae,       # inflated by zeros
        "clean_denorm_MAE_masked":   clean_denorm_masked,    # *** PAPER-FACING ***
        "clean_denorm_RMSE_masked":  clean_denorm_rmse_masked,
        "pn": pn, "tn": tn, "pr": pr, "tr": tr,             # tensors for reuse
    }


# ---------------------------------------------------------------------------
# B. Sensor dropout
# ---------------------------------------------------------------------------
def eval_sensor_dropout_single(model_name, model, x_test, y_test, scaler,
                                 device, dropout_ratio, mask_seed,
                                 clean_norm_mae_ref=None):
    """One (dropout_ratio, mask_seed) evaluation. Returns dict."""
    rng = np.random.RandomState(mask_seed)
    n_nodes = x_test.shape[2]
    n_drop  = int(n_nodes * dropout_ratio)
    drop_idx = sorted(rng.choice(n_nodes, n_drop, replace=False).tolist())
    keep_idx = [i for i in range(n_nodes) if i not in drop_idx]

    x_masked = x_test.copy()
    x_masked[:, :, drop_idx, 0] = 0.0

    # clean (re-run for kept-node-only metric)
    cl_loader = make_torch_loader(x_test, y_test, BATCH_SIZE, shuffle=False)
    pn_cl, tn_cl, pr_cl, tr_cl = collect_preds(model_name, model, cl_loader, scaler, device)

    # masked
    mk_loader = make_torch_loader(x_masked, y_test, BATCH_SIZE, shuffle=False)
    pn_mk, tn_mk, pr_mk, tr_mk = collect_preds(model_name, model, mk_loader, scaler, device)

    # kept-node normalised MAE
    clean_kept_norm  = norm_mae(pn_cl[:, :, keep_idx, :], tn_cl[:, :, keep_idx, :])
    masked_kept_norm = norm_mae(pn_mk[:, :, keep_idx, :], tn_mk[:, :, keep_idx, :])
    delta_norm       = masked_kept_norm - clean_kept_norm

    # kept-node masked-denorm MAE
    pr_cl_np = pr_cl[:, :, keep_idx, :].numpy()
    tr_cl_np = tr_cl[:, :, keep_idx, :].numpy()
    pr_mk_np = pr_mk[:, :, keep_idx, :].numpy()
    tr_mk_np = tr_mk[:, :, keep_idx, :].numpy()
    clean_kept_masked_mph  = masked_mae_np(pr_cl_np, tr_cl_np)
    masked_kept_masked_mph = masked_mae_np(pr_mk_np, tr_mk_np)
    delta_masked_mph       = masked_kept_masked_mph - clean_kept_masked_mph

    # relative fragility (vs clean masked mph for comparability)
    rel_frag_norm   = delta_norm / max(abs(clean_kept_norm), 1e-8)
    rel_frag_masked = delta_masked_mph / max(abs(clean_kept_masked_mph), 1e-8)

    return {
        "dropout_ratio":          dropout_ratio,
        "mask_seed":              mask_seed,
        "n_drop":                 n_drop,
        "n_keep":                 len(keep_idx),
        "clean_kept_norm_MAE":    clean_kept_norm,
        "masked_kept_norm_MAE":   masked_kept_norm,
        "delta_norm":             delta_norm,
        "clean_kept_masked_mph":  clean_kept_masked_mph,
        "masked_kept_masked_mph": masked_kept_masked_mph,
        "delta_masked_mph":       delta_masked_mph,
        "rel_frag_norm":          rel_frag_norm,
        "rel_frag_masked_mph":    rel_frag_masked,
    }


def eval_sensor_severity(model_name, model, x_test, y_test, scaler, device):
    """Full severity curve: all SEVERITY_PTS × MASK_SEEDS."""
    results = {}
    for sev in SEVERITY_PTS:
        per_mask = []
        for ms in MASK_SEEDS:
            r = eval_sensor_dropout_single(
                model_name, model, x_test, y_test, scaler, device, sev, ms)
            per_mask.append(r)
        delta_norms  = [r["delta_norm"]       for r in per_mask]
        delta_mphs   = [r["delta_masked_mph"] for r in per_mask]
        results[sev] = {
            "per_mask":           per_mask,
            "delta_norm_mean":    float(np.mean(delta_norms)),
            "delta_norm_std":     float(np.std(delta_norms)),
            "delta_masked_mph_mean": float(np.mean(delta_mphs)),
            "delta_masked_mph_std":  float(np.std(delta_mphs)),
            "clean_kept_norm_mean":  float(np.mean([r["clean_kept_norm_MAE"] for r in per_mask])),
            "clean_kept_masked_mph_mean": float(np.mean([r["clean_kept_masked_mph"] for r in per_mask])),
        }
        print(f"      sev={int(sev*100):2d}%  "
              f"Δnorm={results[sev]['delta_norm_mean']:+.5f}±{results[sev]['delta_norm_std']:.5f}  "
              f"Δmph={results[sev]['delta_masked_mph_mean']:+.4f}±{results[sev]['delta_masked_mph_std']:.4f}")
    return results


# ---------------------------------------------------------------------------
# C. Cross-city transfer
# ---------------------------------------------------------------------------
def eval_cross_city(model_name, model, source_scaler, subsets, device):
    from experiments.official_adaptive_common import scalar_source_scaler

    subset_scaler = scalar_source_scaler(source_scaler, 207)
    src_maes, tgt_maes_masked, tgt_maes_raw = [], [], []

    for subset in subsets:
        tgt_scaler = fit_train_scaler(subset["train"], mode=scaler_mode_for_model(model_name))

        # --- source-stat evaluation ---
        x_src, y_src = build_arrays(model_name, subset["test"], subset_scaler)
        ld_src = make_torch_loader(x_src, y_src, BATCH_SIZE, shuffle=False)
        pn_s, tn_s, *_ = collect_preds(model_name, model, ld_src, subset_scaler, device)
        src_maes.append(norm_mae(pn_s, tn_s))

        # --- target-stat evaluation ---
        x_tgt, y_tgt = build_arrays(model_name, subset["test"], tgt_scaler)
        ld_tgt = make_torch_loader(x_tgt, y_tgt, BATCH_SIZE, shuffle=False)
        pn_t, tn_t, pr_t, tr_t = collect_preds(model_name, model, ld_tgt, tgt_scaler, device)
        tgt_maes_raw.append(norm_mae(pr_t, tr_t))
        tgt_maes_masked.append(compute_masked_metrics(pr_t, tr_t)["MAE"])

    return {
        "city_src_MAE_mean":        float(np.mean(src_maes)),
        "city_src_MAE_std":         float(np.std(src_maes)),
        "city_src_MAE_per_subset":  [float(v) for v in src_maes],
        "city_tgt_MAE_raw_mean":    float(np.mean(tgt_maes_raw)),
        "city_tgt_MAE_raw_std":     float(np.std(tgt_maes_raw)),
        "city_tgt_MAE_masked_mean": float(np.mean(tgt_maes_masked)),
        "city_tgt_MAE_masked_std":  float(np.std(tgt_maes_masked)),
        "city_tgt_MAE_masked_per_subset": [float(v) for v in tgt_maes_masked],
        "n_subsets":                N_SUBSETS,
    }


# ---------------------------------------------------------------------------
# D. Temporal subgroup
# ---------------------------------------------------------------------------
def build_temporal_loaders(model_name, test_df, scaler):
    """Return (wd_loader, we_loader, wd_count, we_count) for test_df."""
    # compute weekday/weekend mask on test_df rows
    dow = test_df.index.dayofweek   # Mon=0, Sun=6
    is_weekend = (dow == 5) | (dow == 6)

    x_arr, y_arr = build_arrays(model_name, test_df, scaler)
    limit = len(test_df) - SEQ_LEN - PRED_LEN + 1
    wd_idx = [t for t in range(limit) if not is_weekend[t]]
    we_idx = [t for t in range(limit) if     is_weekend[t]]

    ld_wd = make_torch_loader(x_arr[wd_idx], y_arr[wd_idx], BATCH_SIZE) if wd_idx else None
    ld_we = make_torch_loader(x_arr[we_idx], y_arr[we_idx], BATCH_SIZE) if we_idx else None
    return ld_wd, ld_we, len(wd_idx), len(we_idx)


def eval_temporal(model_name, model, test_df, scaler, device):
    ld_wd, ld_we, n_wd, n_we = build_temporal_loaders(model_name, test_df, scaler)
    if ld_wd is None or ld_we is None:
        return {"error": "insufficient windows"}

    pn_wd, tn_wd, pr_wd, tr_wd = collect_preds(model_name, model, ld_wd, scaler, device)
    pn_we, tn_we, pr_we, tr_we = collect_preds(model_name, model, ld_we, scaler, device)

    wd_norm_mae = norm_mae(pn_wd, tn_wd)
    we_norm_mae = norm_mae(pn_we, tn_we)

    wd_masked_mph = compute_masked_metrics(pr_wd, tr_wd)["MAE"]
    we_masked_mph = compute_masked_metrics(pr_we, tr_we)["MAE"]

    # Per-horizon temporal gap (normalized space)
    horizon_gaps = []
    for h in range(PRED_LEN):   # h = 0..11 → horizon 1..12
        wd_h = norm_mae(pn_wd[:, h, :, :], tn_wd[:, h, :, :])
        we_h = norm_mae(pn_we[:, h, :, :], tn_we[:, h, :, :])
        horizon_gaps.append({
            "horizon": h + 1,
            "weekday_norm_MAE": wd_h,
            "weekend_norm_MAE": we_h,
            "delta_norm":       we_h - wd_h,
        })

    return {
        "weekday_windows":        n_wd,
        "weekend_windows":        n_we,
        "weekday_norm_MAE":       wd_norm_mae,
        "weekend_norm_MAE":       we_norm_mae,
        "delta_temporal_norm":    we_norm_mae - wd_norm_mae,
        "weekday_masked_mph":     wd_masked_mph,
        "weekend_masked_mph":     we_masked_mph,
        "delta_temporal_masked_mph": we_masked_mph - wd_masked_mph,
        "per_horizon":            horizon_gaps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["agcrn", "megacrn", "staeformer"])
    ap.add_argument("--registry",
                    default="results/official_adaptive_relation/official_adaptive_unified_registry.json")
    ap.add_argument("--source_dataset", default="metr-la")
    ap.add_argument("--target_dataset", default="pems-bay")
    ap.add_argument("--output_dir", default="results/official_adaptive_relation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag    = DATE_TAG

    registry = load_registry(args.registry)
    subsets  = sample_target_subsets(args.target_dataset, 207, n_subsets=N_SUBSETS,
                                     base_seed=SUBSET_SEEDS)

    all_results  = {}   # full per-seed tree
    summary_rows = []   # paper-table rows

    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'='*70}")

        # build source dataset once per model (scaler mode may differ)
        scaler_mode = scaler_mode_for_model(model_name)
        source = build_source_dataset(args.source_dataset, scaler_mode=scaler_mode)
        test_df = source["test"]
        scaler  = source["scaler"]
        x_test, y_test = build_arrays(model_name, test_df, scaler)

        model_results = {}
        keys = sorted(k for k in registry
                      if k.startswith(f"{model_name}__{args.source_dataset}__"))

        for key in keys:
            entry = registry[key]
            seed  = entry["seed"]
            print(f"\n  --- seed {seed} ---")
            t0 = time.time()

            ckpt  = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
            model = build_model_from_config(model_name, ckpt["model_config"], device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            ckpt_scaler = StandardScaler(
                np.array(ckpt["scaler_mean"], dtype=np.float32),
                np.array(ckpt["scaler_std"],  dtype=np.float32))

            # A. Clean
            print("    [A] Clean eval ...", flush=True)
            clean = eval_clean(model_name, model, x_test, y_test, ckpt_scaler, device)
            print(f"        masked MAE = {clean['clean_denorm_MAE_masked']:.4f} mph  "
                  f"(paper-facing)   norm MAE = {clean['clean_norm_MAE']:.5f}")
            # drop tensors from JSON output (keep separately)
            clean_tensors = {k: clean.pop(k) for k in ["pn","tn","pr","tr"]}

            # B. Sensor dropout severity
            print("    [B] Sensor severity curve ...", flush=True)
            sensor = eval_sensor_severity(model_name, model, x_test, y_test,
                                          ckpt_scaler, device)

            # C. Cross-city
            print("    [C] Cross-city ...", flush=True)
            city = eval_cross_city(model_name, model, ckpt_scaler, subsets, device)
            print(f"        city_src_MAE = {city['city_src_MAE_mean']:.5f}  "
                  f"city_tgt_masked = {city['city_tgt_MAE_masked_mean']:.4f} mph")

            # D. Temporal
            print("    [D] Temporal subgroup ...", flush=True)
            temporal = eval_temporal(model_name, model, test_df, ckpt_scaler, device)
            print(f"        WD={temporal['weekday_norm_MAE']:.5f}  "
                  f"WE={temporal['weekend_norm_MAE']:.5f}  "
                  f"Δnorm={temporal['delta_temporal_norm']:+.5f}  "
                  f"Δmph={temporal['delta_temporal_masked_mph']:+.4f}")

            elapsed = time.time() - t0
            print(f"    seed {seed} done in {elapsed:.1f}s")

            model_results[key] = {
                "model": model_name, "seed": seed,
                "checkpoint": entry["checkpoint"],
                "clean": clean,
                "sensor_severity": {str(sev): v for sev, v in sensor.items()},
                "city": city,
                "temporal": temporal,
            }

        all_results[model_name] = model_results

        # Build per-model summary row (3-seed aggregate)
        def agg(vals):
            return float(np.mean(vals)), float(np.std(vals))

        seeds_data = list(model_results.values())
        clean_masked_maes  = [s["clean"]["clean_denorm_MAE_masked"] for s in seeds_data]
        s30_delta_norm     = [s["sensor_severity"]["0.3"]["delta_norm_mean"] for s in seeds_data]
        s30_delta_mph      = [s["sensor_severity"]["0.3"]["delta_masked_mph_mean"] for s in seeds_data]
        city_src           = [s["city"]["city_src_MAE_mean"] for s in seeds_data]
        city_tgt_masked    = [s["city"]["city_tgt_MAE_masked_mean"] for s in seeds_data]
        temp_delta_norm    = [s["temporal"]["delta_temporal_norm"] for s in seeds_data]
        temp_delta_mph     = [s["temporal"]["delta_temporal_masked_mph"] for s in seeds_data]
        wd_norm_maes       = [s["temporal"]["weekday_norm_MAE"] for s in seeds_data]
        we_norm_maes       = [s["temporal"]["weekend_norm_MAE"] for s in seeds_data]

        mu, sd = agg(clean_masked_maes)
        summary_rows.append({
            "model":                    model_name,
            "clean_masked_MAE_mean_mph": mu,
            "clean_masked_MAE_std_mph":  sd,
            "sensor_30pct_delta_norm_mean":  float(np.mean(s30_delta_norm)),
            "sensor_30pct_delta_norm_std":   float(np.std(s30_delta_norm)),
            "sensor_30pct_delta_masked_mph_mean": float(np.mean(s30_delta_mph)),
            "sensor_30pct_delta_masked_mph_std":  float(np.std(s30_delta_mph)),
            "city_src_MAE_mean":        float(np.mean(city_src)),
            "city_src_MAE_std":         float(np.std(city_src)),
            "city_tgt_MAE_masked_mean_mph": float(np.mean(city_tgt_masked)),
            "temporal_delta_norm_mean": float(np.mean(temp_delta_norm)),
            "temporal_delta_norm_std":  float(np.std(temp_delta_norm)),
            "temporal_delta_masked_mph_mean": float(np.mean(temp_delta_mph)),
            "temporal_delta_masked_mph_std":  float(np.std(temp_delta_mph)),
            "weekday_norm_MAE_mean":    float(np.mean(wd_norm_maes)),
            "weekend_norm_MAE_mean":    float(np.mean(we_norm_maes)),
        })

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    # 1. Full result JSON
    full_path = out / f"OFFICIAL_ADAPTIVE_FULL_EVAL_{tag}.json"
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWritten: {full_path}")

    # 2. Summary JSON
    summ_path = out / f"FULL_OVERNIGHT_EVAL_SUMMARY_{tag}.json"
    with open(summ_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Written: {summ_path}")

    # 3. Sensor severity curves JSON
    sev_export = {}
    for model_name, seeds in all_results.items():
        sev_export[model_name] = {}
        for key, res in seeds.items():
            seed = res["seed"]
            sev_export[model_name][f"seed{seed}"] = {}
            for sev_key, sev_val in res["sensor_severity"].items():
                sev_export[model_name][f"seed{seed}"][sev_key] = {
                    "delta_norm_mean":    sev_val["delta_norm_mean"],
                    "delta_norm_std":     sev_val["delta_norm_std"],
                    "delta_masked_mph_mean": sev_val["delta_masked_mph_mean"],
                    "delta_masked_mph_std":  sev_val["delta_masked_mph_std"],
                }
    sev_path = out / f"SENSOR_SEVERITY_CURVES_{tag}.json"
    with open(sev_path, "w") as f:
        json.dump(sev_export, f, indent=2)
    print(f"Written: {sev_path}")

    # 4. City rerun summary JSON
    city_export = {}
    for model_name, seeds in all_results.items():
        city_export[model_name] = {}
        for key, res in seeds.items():
            city_export[model_name][f"seed{res['seed']}"] = res["city"]
    city_path = out / f"CITY_RERUN_SUMMARY_{tag}.json"
    with open(city_path, "w") as f:
        json.dump(city_export, f, indent=2)
    print(f"Written: {city_path}")

    # 5. Temporal rerun summary JSON
    temp_export = {}
    for model_name, seeds in all_results.items():
        temp_export[model_name] = {}
        for key, res in seeds.items():
            temp_export[model_name][f"seed{res['seed']}"] = res["temporal"]
    temp_path = out / f"TEMPORAL_RERUN_SUMMARY_{tag}.json"
    with open(temp_path, "w") as f:
        json.dump(temp_export, f, indent=2)
    print(f"Written: {temp_path}")

    # 6. CSV summary
    csv_rows = []
    for row in summary_rows:
        csv_rows.append({
            "model":          row["model"],
            "clean_masked_MAE_mph": f"{row['clean_masked_MAE_mean_mph']:.4f}±{row['clean_masked_MAE_std_mph']:.4f}",
            "sensor30_delta_norm":  f"{row['sensor_30pct_delta_norm_mean']:+.4f}±{row['sensor_30pct_delta_norm_std']:.4f}",
            "sensor30_delta_mph":   f"{row['sensor_30pct_delta_masked_mph_mean']:+.4f}±{row['sensor_30pct_delta_masked_mph_std']:.4f}",
            "city_src_MAE":         f"{row['city_src_MAE_mean']:.4f}±{row['city_src_MAE_std']:.4f}",
            "city_tgt_mph":         f"{row['city_tgt_MAE_masked_mean_mph']:.4f}",
            "temporal_delta_norm":  f"{row['temporal_delta_norm_mean']:+.4f}±{row['temporal_delta_norm_std']:.4f}",
            "temporal_delta_mph":   f"{row['temporal_delta_masked_mph_mean']:+.4f}±{row['temporal_delta_masked_mph_std']:.4f}",
        })
    csv_path = out / f"FULL_OVERNIGHT_EVAL_SUMMARY_{tag}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Written: {csv_path}")

    # 7. Print final table
    print(f"\n{'='*100}")
    print("FINAL SUMMARY TABLE (paper-facing units)")
    print(f"{'='*100}")
    print(f"{'Model':>12} | {'Clean masked MAE':>18} | {'Sensor30 Δnorm':>15} | "
          f"{'Sensor30 Δmph':>14} | {'City src':>9} | {'Temp Δnorm':>11} | {'Temp Δmph':>10}")
    print("-"*100)
    for r in summary_rows:
        print(f"  {r['model']:>10} | "
              f"  {r['clean_masked_MAE_mean_mph']:.4f}±{r['clean_masked_MAE_std_mph']:.4f} | "
              f"  {r['sensor_30pct_delta_norm_mean']:+.4f}±{r['sensor_30pct_delta_norm_std']:.4f} | "
              f"  {r['sensor_30pct_delta_masked_mph_mean']:+.4f}±{r['sensor_30pct_delta_masked_mph_std']:.4f} | "
              f"  {r['city_src_MAE_mean']:.4f}±{r['city_src_MAE_std']:.4f} | "
              f"  {r['temporal_delta_norm_mean']:+.4f}±{r['temporal_delta_norm_std']:.4f} | "
              f"  {r['temporal_delta_masked_mph_mean']:+.4f}±{r['temporal_delta_masked_mph_std']:.4f}")

    print(f"\nAll outputs written to {out}/")
    print("Overnight eval complete.")


if __name__ == "__main__":
    main()
