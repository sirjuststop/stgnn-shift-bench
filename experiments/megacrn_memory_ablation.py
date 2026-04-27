"""
megacrn_memory_ablation.py
===========================
Test the hypothesis that MegaCRN's negative Δ_sensor at moderate dropout ratios
comes from its memory-bank (meta-graph + prototype attention) acting as an
implicit regularizer.

Procedure: for each seed, load the MegaCRN checkpoint, then for each of
  - full:        standard memory
  - memzero:     set memory['Memory'] = 0  →  prototype attention h_att = 0
                 and g1 = g2 = uniform 1/N adjacency  (full ablation)
  - graphonly:   replace h_att with zeros via forward patch (disables attention,
                 keeps meta-graph adjacency)
  - attnonly:    replace g1 = g2 = uniform adjacency (disables meta-graph,
                 keeps prototype attention)
run the sensor-dropout severity sweep at {10,20,30,40,50}% with 3 mask seeds.

If the U-shape collapses under memzero / graphonly (i.e. delta becomes
monotone-positive like AGCRN), the regularization hypothesis is supported.
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
    build_source_dataset,
    load_registry,
    make_torch_loader,
    scaler_mode_for_model,
)
from experiments.official_adaptive_models import build_model_from_config


BATCH_SIZE = 64
SEQ_LEN = 12
PRED_LEN = 12
DATE_TAG = date.today().strftime("%Y-%m-%d")
SEVERITY_PTS = [0.10, 0.20, 0.30, 0.40, 0.50]
MASK_SEEDS = [42, 43, 44]


def norm_mae(pred, tgt):
    return float((pred - tgt).abs().mean())


def install_ablation(model, mode):
    """Patch the model's forward for the given ablation mode. Returns an undo fn."""
    orig_forward = model.forward
    orig_memory_data = model.memory["Memory"].data.clone()
    orig_query_memory = model.query_memory

    def forward_memzero(x, y_cov, labels=None, batches_seen=None):
        # Set all memory rows to 0 for this call; restore afterwards via undo.
        return orig_forward(x, y_cov, labels, batches_seen)

    if mode == "full":
        return lambda: None
    if mode == "memzero":
        with torch.no_grad():
            model.memory["Memory"].data.zero_()
        def undo():
            with torch.no_grad():
                model.memory["Memory"].data.copy_(orig_memory_data)
        return undo
    if mode == "graphonly":
        # Keep adjacency (uses Memory) intact, but force h_att=0 by replacing query_memory.
        def query_memory_zero(h_t):
            B, N, _ = h_t.shape
            d = model.mem_dim
            h_att_zero = torch.zeros(B, N, d, device=h_t.device)
            # pos/neg/query still need to return something shape-correct to avoid
            # downstream crashes; they're used only in training losses.
            query = torch.zeros(B, N, d, device=h_t.device)
            pos = torch.zeros(B, N, d, device=h_t.device)
            neg = torch.zeros(B, N, d, device=h_t.device)
            return h_att_zero, query, pos, neg
        model.query_memory = query_memory_zero
        def undo():
            model.query_memory = orig_query_memory
        return undo
    if mode == "attnonly":
        import torch.nn.functional as F
        def forward_attnonly(x, y_cov, labels=None, batches_seen=None):
            # Replace g1/g2 with identity (keep attention intact)
            N = model.num_nodes
            I = torch.eye(N, device=x.device) / 1.0  # each row has mass on itself
            supports = [I, I]
            init_state = model.encoder.init_hidden(x.shape[0])
            h_en, state_en = model.encoder(x, init_state, supports)
            h_t = h_en[:, -1, :, :]
            h_att, query, pos, neg = model.query_memory(h_t)
            h_t = torch.cat([h_t, h_att], dim=-1)
            ht_list = [h_t] * model.num_layers
            go = torch.zeros((x.shape[0], model.num_nodes, model.output_dim), device=x.device)
            out = []
            for t in range(model.horizon):
                h_de, ht_list = model.decoder(
                    torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
                go = model.proj(h_de)
                out.append(go)
            return torch.stack(out, dim=1), h_att, query, pos, neg
        model.forward = forward_attnonly
        def undo():
            model.forward = orig_forward
        return undo
    raise ValueError(mode)


@torch.no_grad()
def collect_preds_megacrn(model, loader, scaler, device):
    pn, tn, pr, tr = [], [], [], []
    for bx, by in loader:
        bx = bx.to(device); by = by.to(device)
        pred_norm, *_ = model(bx[..., :1], by[..., 1:], None, None)
        target_norm = by[..., :1]
        pred_raw = scaler.inverse_transform_torch(pred_norm)
        target_raw = scaler.inverse_transform_torch(target_norm)
        pn.append(pred_norm.cpu()); tn.append(target_norm.cpu())
        pr.append(pred_raw.cpu());  tr.append(target_raw.cpu())
    return torch.cat(pn), torch.cat(tn), torch.cat(pr), torch.cat(tr)


def eval_dropout(model, x_test, y_test, scaler, device, ratio, mask_seed):
    rng = np.random.RandomState(mask_seed)
    n_nodes = x_test.shape[2]
    n_drop = int(n_nodes * ratio)
    drop_idx = sorted(rng.choice(n_nodes, n_drop, replace=False).tolist())
    keep_idx = [i for i in range(n_nodes) if i not in drop_idx]

    x_masked = x_test.copy()
    x_masked[:, :, drop_idx, 0] = 0.0

    cl_loader = make_torch_loader(x_test, y_test, BATCH_SIZE)
    mk_loader = make_torch_loader(x_masked, y_test, BATCH_SIZE)
    pn_cl, tn_cl, *_ = collect_preds_megacrn(model, cl_loader, scaler, device)
    pn_mk, tn_mk, *_ = collect_preds_megacrn(model, mk_loader, scaler, device)

    clean_kept = norm_mae(pn_cl[:, :, keep_idx, :], tn_cl[:, :, keep_idx, :])
    masked_kept = norm_mae(pn_mk[:, :, keep_idx, :], tn_mk[:, :, keep_idx, :])
    return clean_kept, masked_kept, masked_kept - clean_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry",
        default="results/official_adaptive_relation/official_adaptive_unified_registry.json")
    ap.add_argument("--source_dataset", default="metr-la")
    ap.add_argument("--output_dir", default="results/megacrn_memory_ablation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--modes", nargs="+",
                    default=["full", "memzero", "graphonly", "attnonly"])
    args = ap.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    registry = load_registry(args.registry)

    scaler_mode = scaler_mode_for_model("megacrn")
    source = build_source_dataset(args.source_dataset, scaler_mode=scaler_mode)
    test_df = source["test"]

    keys = sorted(k for k in registry
                  if k.startswith(f"megacrn__{args.source_dataset}__"))
    print(f"Found {len(keys)} MegaCRN checkpoints")

    results = {}
    for key in keys:
        entry = registry[key]; seed = entry["seed"]
        print(f"\n=== seed {seed} ===")
        t0 = time.time()
        ckpt = torch.load(REPO_ROOT / entry["checkpoint"], map_location=device, weights_only=False)
        model = build_model_from_config("megacrn", ckpt["model_config"], device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        ckpt_scaler = StandardScaler(
            np.array(ckpt["scaler_mean"], dtype=np.float32),
            np.array(ckpt["scaler_std"],  dtype=np.float32))
        x_test, y_test = build_megacrn_arrays(test_df, ckpt_scaler)

        seed_results = {}
        for mode in args.modes:
            undo = install_ablation(model, mode)
            try:
                per_ratio = {}
                for r in SEVERITY_PTS:
                    deltas = []
                    for ms in MASK_SEEDS:
                        _, _, d = eval_dropout(model, x_test, y_test,
                                               ckpt_scaler, device, r, ms)
                        deltas.append(d)
                    per_ratio[str(r)] = {
                        "delta_mean": float(np.mean(deltas)),
                        "delta_std":  float(np.std(deltas)),
                    }
                seed_results[mode] = per_ratio
                print(f"  [{mode:>10}]  " +
                      "  ".join(f"r={r}:{per_ratio[str(r)]['delta_mean']:+.4f}"
                                for r in SEVERITY_PTS))
            finally:
                undo()
        results[key] = seed_results
        print(f"  seed {seed} done in {time.time()-t0:.1f}s")

    # ---- aggregate ----
    summary = {}
    for mode in args.modes:
        summary[mode] = {}
        for r in SEVERITY_PTS:
            deltas = [results[k][mode][str(r)]["delta_mean"] for k in results]
            summary[mode][str(r)] = {
                "delta_mean_over_seeds": float(np.mean(deltas)),
                "delta_std_over_seeds":  float(np.std(deltas)),
            }

    full_path = out / f"MEGACRN_MEM_ABLATION_FULL_{DATE_TAG}.json"
    summ_path = out / f"MEGACRN_MEM_ABLATION_SUMMARY_{DATE_TAG}.json"
    with open(full_path, "w") as f: json.dump(results, f, indent=2)
    with open(summ_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"\nWritten: {full_path}")
    print(f"Written: {summ_path}")

    print(f"\n{'='*80}\nSUMMARY (delta mean across 3 seeds × 3 masks)\n{'='*80}")
    print(f"{'mode':>10}   " + "  ".join(f"r={r:.1f}" for r in SEVERITY_PTS))
    for mode in args.modes:
        row = [f"{summary[mode][str(r)]['delta_mean_over_seeds']:+.4f}" for r in SEVERITY_PTS]
        print(f"{mode:>10}   " + "  ".join(row))


if __name__ == "__main__":
    main()
