"""
temporal_covariate_ablation.py
==============================
Inference-time ablation for MegaCRN and STAEformer weekday/weekend gap.

For each (model, seed):
  1. Clean:      baseline weekday/weekend delta (reference).
  2. Randomize:  replace tod (and dow for STAEformer) with uniform noise.
  3. Zero:       replace tod/dow with zeros.
  4. Swap:       for weekday samples use a weekend covariate; for weekend samples
                 use a weekday covariate. Directly tests causal direction.

If "frozen temporal covariate" drives the large MegaCRN/STAEformer weekend gap,
we expect:
  - Randomize / Zero  → gap collapses (gap ≈ 0)
  - Swap              → gap sign flips (weekday MAE rises above weekend MAE,
                        or at least the two converge toward each other)

All runs use frozen checkpoints; no retraining.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import date

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.official_adaptive_common import (
    StandardScaler,
    build_megacrn_arrays,
    build_staeformer_arrays,
    build_source_dataset,
    compute_masked_metrics,
    load_registry,
    make_torch_loader,
    scaler_mode_for_model,
    set_seed,
)
from experiments.official_adaptive_models import build_model_from_config


BATCH_SIZE = 64
SEQ_LEN = 12
PRED_LEN = 12
DATE_TAG = date.today().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Channel layout reminder
#   MegaCRN    x: [..., C=2]  (value, tod).  Model forward uses by[..., 1:]
#              as TARGET-HORIZON tod. So covariate perturbation must target y[...,1].
#   STAEformer x: [..., C=3]  (value, tod, dow).  Model forward uses bx directly;
#              y is (N,12,N_nodes,1). Covariate perturbation targets x[...,1:].
# ---------------------------------------------------------------------------

def build_arrays(model_name, frame, scaler):
    if model_name == "megacrn":
        return build_megacrn_arrays(frame, scaler, SEQ_LEN, PRED_LEN)
    if model_name == "staeformer":
        return build_staeformer_arrays(frame, scaler, SEQ_LEN, PRED_LEN)
    raise KeyError(model_name)


def covariate_target_tensor(model_name):
    """Which tensor's channel is consumed by the model? ('x' or 'y')."""
    if model_name == "megacrn":
        return "y"
    if model_name == "staeformer":
        return "x"
    raise KeyError(model_name)


def covariate_channels(model_name):
    if model_name == "megacrn":
        return [1]                           # tod only
    if model_name == "staeformer":
        return [1, 2]                        # tod, dow
    raise KeyError(model_name)


def apply_covariate_perturbation(arr, model_name, mode, rng):
    """Perturb the covariate channels of `arr` in-place-safe (returns a copy)."""
    arr = arr.copy()
    if mode == "clean":
        return arr
    chans = covariate_channels(model_name)
    if mode == "zero":
        for c in chans:
            arr[..., c] = 0.0
    elif mode == "randomize":
        # tod must stay strictly < 1.0 (STAEformer casts (tod*288).long() as index).
        # dow must stay in {0,..,6} (Embedding(7,_) index).
        for c in chans:
            if model_name == "staeformer" and c == 2:
                arr[..., c] = rng.randint(0, 7, size=arr[..., c].shape).astype(np.float32)
            else:
                arr[..., c] = rng.uniform(0.0, 0.9999, size=arr[..., c].shape).astype(np.float32)
    else:
        raise ValueError(mode)
    return arr


def apply_covariate_swap_pair(arr, wd_idx, we_idx, model_name):
    """Swap covariate channels across weekday/weekend groups.

    Weekday samples receive covariates drawn (wrap-around) from weekend samples,
    and vice versa. Single pass using the original values to avoid double-overwrite.
    """
    arr_orig = arr.copy()
    arr_new = arr.copy()
    chans = covariate_channels(model_name)
    if len(wd_idx) == 0 or len(we_idx) == 0:
        return arr_new
    reps_wd = (len(wd_idx) + len(we_idx) - 1) // len(we_idx)
    donors_for_wd = np.tile(we_idx, reps_wd)[: len(wd_idx)]
    reps_we = (len(we_idx) + len(wd_idx) - 1) // len(wd_idx)
    donors_for_we = np.tile(wd_idx, reps_we)[: len(we_idx)]
    for c in chans:
        arr_new[wd_idx, :, :, c] = arr_orig[donors_for_wd, :, :, c]
        arr_new[we_idx, :, :, c] = arr_orig[donors_for_we, :, :, c]
    return arr_new


def perturb_for_model(x_arr, y_arr, model_name, mode, rng, wd_idx=None, we_idx=None):
    """Return (x_new, y_new). Only the channel the model reads is perturbed."""
    target = covariate_target_tensor(model_name)
    if mode in ("clean",):
        return x_arr, y_arr
    if mode == "swap":
        if target == "x":
            return apply_covariate_swap_pair(x_arr, wd_idx, we_idx, model_name), y_arr
        return x_arr, apply_covariate_swap_pair(y_arr, wd_idx, we_idx, model_name)
    # zero / randomize
    if target == "x":
        return apply_covariate_perturbation(x_arr, model_name, mode, rng), y_arr
    return x_arr, apply_covariate_perturbation(y_arr, model_name, mode, rng)


@torch.no_grad()
def forward_batch(model_name, model, bx, by, device):
    bx = bx.to(device); by = by.to(device)
    if model_name == "megacrn":
        pred_norm, *_ = model(bx[..., :1], by[..., 1:], None, None)
        return pred_norm
    if model_name == "staeformer":
        return model(bx)
    raise KeyError(model_name)


@torch.no_grad()
def collect_preds(model_name, model, loader, scaler, device):
    pn_list, tn_list, pr_list, tr_list = [], [], [], []
    for bx, by in loader:
        pred_norm = forward_batch(model_name, model, bx, by, device)
        if model_name == "staeformer":
            target_raw = by.to(device)
            target_norm = scaler.transform_torch(target_raw)
        elif model_name == "megacrn":
            target_norm = by[..., :1].to(device)
            target_raw = scaler.inverse_transform_torch(target_norm)
        else:
            raise KeyError(model_name)
        pred_raw = scaler.inverse_transform_torch(pred_norm)
        pn_list.append(pred_norm.cpu()); tn_list.append(target_norm.cpu())
        pr_list.append(pred_raw.cpu());  tr_list.append(target_raw.cpu())
    return (torch.cat(pn_list), torch.cat(tn_list),
            torch.cat(pr_list), torch.cat(tr_list))


def norm_mae(pred, tgt):
    return float((pred - tgt).abs().mean())


def split_weekday_weekend(test_df):
    dow = test_df.index.dayofweek
    is_weekend = (dow == 5) | (dow == 6)
    limit = len(test_df) - SEQ_LEN - PRED_LEN + 1
    wd_idx = np.array([t for t in range(limit) if not is_weekend[t]], dtype=np.int64)
    we_idx = np.array([t for t in range(limit) if     is_weekend[t]], dtype=np.int64)
    return wd_idx, we_idx


def eval_mode(model_name, model, x_arr, y_arr, wd_idx, we_idx, scaler, device):
    """Return dict with weekday / weekend normalized + masked-mph MAE + delta."""
    ld_wd = make_torch_loader(x_arr[wd_idx], y_arr[wd_idx], BATCH_SIZE)
    ld_we = make_torch_loader(x_arr[we_idx], y_arr[we_idx], BATCH_SIZE)
    pn_wd, tn_wd, pr_wd, tr_wd = collect_preds(model_name, model, ld_wd, scaler, device)
    pn_we, tn_we, pr_we, tr_we = collect_preds(model_name, model, ld_we, scaler, device)
    wd_norm = norm_mae(pn_wd, tn_wd)
    we_norm = norm_mae(pn_we, tn_we)
    wd_masked = compute_masked_metrics(pr_wd, tr_wd)["MAE"]
    we_masked = compute_masked_metrics(pr_we, tr_we)["MAE"]
    return {
        "weekday_norm_MAE":  wd_norm,
        "weekend_norm_MAE":  we_norm,
        "delta_norm":        we_norm - wd_norm,
        "weekday_masked_mph": wd_masked,
        "weekend_masked_mph": we_masked,
        "delta_masked_mph":   we_masked - wd_masked,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["megacrn", "staeformer"])
    ap.add_argument("--registry",
                    default="results/official_adaptive_relation/official_adaptive_unified_registry.json")
    ap.add_argument("--source_dataset", default="metr-la")
    ap.add_argument("--output_dir", default="results/temporal_covariate_ablation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rng_seed", type=int, default=1234)
    args = ap.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    registry = load_registry(args.registry)

    results = {}
    for model_name in args.models:
        print(f"\n{'='*70}\n  MODEL: {model_name.upper()}\n{'='*70}")
        scaler_mode = scaler_mode_for_model(model_name)
        source = build_source_dataset(args.source_dataset, scaler_mode=scaler_mode)
        test_df = source["test"]
        x_clean, y_clean = build_arrays(model_name, test_df, source["scaler"])
        wd_idx, we_idx = split_weekday_weekend(test_df)
        print(f"  wd windows: {len(wd_idx)}   we windows: {len(we_idx)}")

        model_results = {}
        keys = sorted(k for k in registry
                      if k.startswith(f"{model_name}__{args.source_dataset}__"))
        for key in keys:
            entry = registry[key]; seed = entry["seed"]
            print(f"\n  -- seed {seed} --")
            t0 = time.time()
            ckpt_path = REPO_ROOT / entry["checkpoint"]
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_model_from_config(model_name, ckpt["model_config"], device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            ckpt_scaler = StandardScaler(
                np.array(ckpt["scaler_mean"], dtype=np.float32),
                np.array(ckpt["scaler_std"],  dtype=np.float32))

            # rebuild arrays with the checkpoint scaler (in case of slight drift)
            x_arr, y_arr = build_arrays(model_name, test_df, ckpt_scaler)
            rng = np.random.RandomState(args.rng_seed + seed)

            per_mode = {}
            for mode in ("clean", "zero", "randomize", "swap"):
                x_mod, y_mod = perturb_for_model(
                    x_arr, y_arr, model_name, mode, rng,
                    wd_idx=wd_idx, we_idx=we_idx)
                per_mode[mode] = eval_mode(
                    model_name, model, x_mod, y_mod, wd_idx, we_idx, ckpt_scaler, device)
                m = per_mode[mode]
                print(f"    [{mode:>9}]  wd={m['weekday_norm_MAE']:.5f}  "
                      f"we={m['weekend_norm_MAE']:.5f}  "
                      f"Δnorm={m['delta_norm']:+.5f}  "
                      f"Δmph={m['delta_masked_mph']:+.4f}")

            elapsed = time.time() - t0
            print(f"    seed {seed} done in {elapsed:.1f}s")
            model_results[key] = {
                "model": model_name, "seed": seed,
                "checkpoint": entry["checkpoint"],
                "per_mode": per_mode,
            }
        results[model_name] = model_results
        # Incremental save after each model
        partial_path = out / f"TEMPORAL_COV_ABLATION_PARTIAL_{DATE_TAG}.json"
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [partial save] {partial_path}")

    # ---- aggregate across seeds ----
    summary = {}
    for model_name, seeds in results.items():
        summary[model_name] = {}
        modes = ["clean", "zero", "randomize", "swap"]
        for mode in modes:
            deltas_norm = [s["per_mode"][mode]["delta_norm"]        for s in seeds.values()]
            deltas_mph  = [s["per_mode"][mode]["delta_masked_mph"]  for s in seeds.values()]
            wd_norms    = [s["per_mode"][mode]["weekday_norm_MAE"]  for s in seeds.values()]
            we_norms    = [s["per_mode"][mode]["weekend_norm_MAE"]  for s in seeds.values()]
            summary[model_name][mode] = {
                "delta_norm_mean": float(np.mean(deltas_norm)),
                "delta_norm_std":  float(np.std(deltas_norm)),
                "delta_masked_mph_mean": float(np.mean(deltas_mph)),
                "delta_masked_mph_std":  float(np.std(deltas_mph)),
                "weekday_norm_MAE_mean": float(np.mean(wd_norms)),
                "weekend_norm_MAE_mean": float(np.mean(we_norms)),
            }

    full_path = out / f"TEMPORAL_COV_ABLATION_FULL_{DATE_TAG}.json"
    summ_path = out / f"TEMPORAL_COV_ABLATION_SUMMARY_{DATE_TAG}.json"
    with open(full_path, "w") as f: json.dump(results, f, indent=2)
    with open(summ_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"\nWritten: {full_path}")
    print(f"Written: {summ_path}")

    # quick print table
    print(f"\n{'='*90}\nAGGREGATE: Δnorm (we - wd) per mode, per model (mean ± std, 3 seeds)\n{'='*90}")
    for model_name, modes in summary.items():
        print(f"\n  {model_name.upper()}")
        for mode_name, stats in modes.items():
            print(f"    {mode_name:>10}:  "
                  f"Δnorm = {stats['delta_norm_mean']:+.5f} ± {stats['delta_norm_std']:.5f}   "
                  f"Δmph = {stats['delta_masked_mph_mean']:+.4f} ± {stats['delta_masked_mph_std']:.4f}   "
                  f"wd={stats['weekday_norm_MAE_mean']:.5f}   we={stats['weekend_norm_MAE_mean']:.5f}")
    print("\nTemporal covariate ablation complete.")


if __name__ == "__main__":
    main()
