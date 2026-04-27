"""
GWNet Hyperparameter Sweep — Direction C
=========================================
Addresses reviewer concern: "GWNet underperforms in-distribution (modest training protocol)."

Two-stage sweep:
  Stage 1: ~12 configs × 1 seed → identify best 2 by in-dist MAE
  Stage 2:  best 2 configs × 3 seeds → final mean ± std

Each config evaluated on:
  - In-dist MAE (METR-LA test, mph)
  - Sensor dropout Δ (30% zero-mask, normalized)
  - City transfer MAE (source-only adj, source-stat normalized) — P16 fair comparison

Usage:
  python experiments/gwnet_sweep.py --stage 1
  python experiments/gwnet_sweep.py --stage 2  # after stage 1 completes
  python experiments/gwnet_sweep.py --stage 2 --best_configs h256_e32_lr5e4_p30 h128_e20_lr5e4_p30
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow imports from experiments/ directory
sys.path.insert(0, os.path.dirname(__file__))
from stgnn_backbone_bank import (
    GWNet,
    load_metr_la,
    load_adjacency,
    make_dataloader,
    evaluate,
    set_seed,
)

OUTPUT_DIR = "results/gwnet_sweep"

# ── Stage 1 sweep grid ──────────────────────────────────────────────────────────
# Vary the capacity-determining dims and learning rate; keep wt_decay, batch_size
# identical to the frozen backbone-bank config for clean comparison.
STAGE1_CONFIGS = [
    # name                   hidden  embed  lr      patience  dropout
    ("h64_e10_lr1e3_p15",      64,    10,  1e-3,    15,      0.3),   # baseline = frozen gwnet
    ("h128_e20_lr1e3_p30",    128,    20,  1e-3,    30,      0.3),   # = gwnet_strong (P17)
    ("h256_e32_lr1e3_p30",    256,    32,  1e-3,    30,      0.3),
    ("h128_e20_lr5e4_p30",    128,    20,  5e-4,    30,      0.3),
    ("h256_e32_lr5e4_p30",    256,    32,  5e-4,    30,      0.3),
    ("h128_e20_lr2e4_p50",    128,    20,  2e-4,    50,      0.3),
    ("h256_e32_lr2e4_p50",    256,    32,  2e-4,    50,      0.3),
    ("h128_e20_lr5e4_p30_d1", 128,    20,  5e-4,    30,      0.1),   # lower dropout
    ("h256_e32_lr5e4_p30_d1", 256,    32,  5e-4,    30,      0.1),
    ("h128_e20_lr1e3_p30_d1", 128,    20,  1e-3,    30,      0.1),
    ("h256_e32_lr1e3_p30_d1", 256,    32,  1e-3,    30,      0.1),
    ("h192_e24_lr5e4_p30",    192,    24,  5e-4,    30,      0.3),   # intermediate size
]


# ── GWNet variant that accepts embed_dim at construction ────────────────────────

class GWNetSweep(nn.Module):
    """GWNet with configurable hidden_dim and embed_dim for the sweep.

    Identical architecture to backbone_bank.GWNet; the only difference is that
    embed_dim is a constructor argument rather than hardcoded to 10.
    """
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=64, dropout=0.3,
                 embed_dim=10, adaptive_adj=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.adaptive_adj = adaptive_adj

        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

        self.input_proj  = nn.Linear(seq_len, hidden_dim)
        self.graph_conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output  = nn.Linear(hidden_dim, pred_len)
        self.relu    = nn.ReLU()

    def get_adaptive_adj(self, device):
        # Match original GWNet: softmax(E1 @ E2^T), no tanh, dim=-1
        a = torch.mm(self.node_emb1, self.node_emb2.T)
        return torch.softmax(a, dim=-1)

    def forward(self, x, adj):
        # x: (B, T, N) or (B, T, N, 1)
        if x.dim() == 4:
            x = x.squeeze(-1)
        x = x.permute(0, 2, 1)     # (B, N, T)

        if self.adaptive_adj:
            adj = self.get_adaptive_adj(x.device)

        x = self.relu(self.input_proj(x))            # (B, N, hidden)
        x_agg1 = torch.einsum("mn,bnd->bmd", adj, x)
        x = self.relu(self.graph_conv1(x_agg1))
        x = self.dropout(x)
        x_agg2 = torch.einsum("mn,bnd->bmd", adj, x)
        x = self.relu(self.graph_conv2(x_agg2))
        x = self.dropout(x)
        return self.output(x)       # (B, N, pred_len)


# ── Training with config dict ───────────────────────────────────────────────────

def train_gwnet_config(cfg, num_nodes, adj, train_loader, val_loader, test_loader,
                       device, stats):
    """Train one GWNet config; return (model, test_metrics_mph)."""
    model = GWNetSweep(
        num_nodes=num_nodes,
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        hidden_dim=cfg["hidden_dim"],
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
        adaptive_adj=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.HuberLoss()

    best_val_mae = float("inf")
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b, adj)
            loss = criterion(pred, Y_b.permute(0, 2, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_m = evaluate(model, val_loader, adj, device, stats=None)
        if val_m["MAE"] < best_val_mae:
            best_val_mae = val_m["MAE"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0:
            print(f"    [ep{epoch:03d}] loss={total_loss/len(train_loader):.4f} "
                  f"val_MAE={val_m['MAE']:.4f}")

        if patience_ctr >= cfg["patience"]:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, adj, device, stats=stats)
    return model, test_m


# ── Sensor dropout evaluation (inline, mirrors shift_eval_v2) ──────────────────

@torch.no_grad()
def eval_sensor_dropout(model, X_te, Y_te, adj, device, ratio=0.30, seed=42,
                        batch_size=64):
    rng = np.random.RandomState(seed)
    N = X_te.shape[2]
    drop_idx = rng.choice(N, int(N * ratio), replace=False)
    keep_idx = [i for i in range(N) if i not in drop_idx]

    def preds_targets(loader_x, loader_y_perm):
        pl, tl = [], []
        for Xb, Yb in make_dataloader(loader_x, loader_y_perm, batch_size, shuffle=False):
            pl.append(model(Xb.to(device), adj).cpu())
            tl.append(Yb.permute(0, 2, 1))
        return torch.cat(pl, 0), torch.cat(tl, 0)

    pc, tc = preds_targets(X_te, Y_te)
    X_m = X_te.copy(); X_m[:, :, drop_idx] = 0.0
    pm, tm = preds_targets(X_m, Y_te)

    mae_c = (pc[:, keep_idx] - tc[:, keep_idx]).abs().mean().item()
    mae_m = (pm[:, keep_idx] - tm[:, keep_idx]).abs().mean().item()
    return {"delta_sensor": mae_m - mae_c, "clean_MAE": mae_c, "masked_MAE": mae_m}


# ── City transfer (source-only adj, source-stat normalized) ───────────────────

@torch.no_grad()
def eval_city_transfer_source_only(model, bay_raw, metr_stats, adj_metr, device,
                                   n_subsets=5, base_seed=42, seq_len=12, pred_len=12,
                                   batch_size=64, N=207):
    """Mirrors P16 protocol: all models use METR-LA adjacency on PEMS-BAY data."""
    T = bay_raw.shape[0]
    test_raw = bay_raw[int(T * 0.8):]
    N_bay = bay_raw.shape[1]

    src_mean = float(metr_stats["mean"].mean())
    src_std  = float(metr_stats["std"].mean())

    maes = []
    for i in range(n_subsets):
        rng = np.random.RandomState(base_seed + i)
        keep = sorted(rng.choice(N_bay, N, replace=False).tolist())

        raw_sub = test_raw[:, keep]
        raw_norm = (raw_sub - src_mean) / src_std

        X_list, Y_list = [], []
        for t in range(len(raw_norm) - seq_len - pred_len + 1):
            X_list.append(raw_norm[t:t + seq_len])
            Y_list.append(raw_norm[t + seq_len:t + seq_len + pred_len])
        X_arr = np.stack(X_list).astype(np.float32)
        Y_arr = np.stack(Y_list).astype(np.float32)

        loader = make_dataloader(X_arr, Y_arr, batch_size, shuffle=False)
        preds, tgts = [], []
        for Xb, Yb in loader:
            preds.append(model(Xb.to(device), adj_metr).cpu())
            tgts.append(Yb.permute(0, 2, 1))
        pred_all = torch.cat(preds, 0)
        tgt_all  = torch.cat(tgts, 0)
        maes.append((pred_all - tgt_all).abs().mean().item())

    return {"city_src_mae_mean": float(np.mean(maes)),
            "city_src_mae_std":  float(np.std(maes))}


# ── Load PEMS-BAY raw data ─────────────────────────────────────────────────────

def load_bay_raw(bay_dir="data/pems-bay"):
    csv_path = os.path.join(bay_dir, "PEMS-BAY.csv")
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    return raw[:, 1:].astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Stage 1: sweep all configs (1 seed). "
                             "Stage 2: retrain best-2 with 3 seeds.")
    parser.add_argument("--best_configs", nargs="+", default=None,
                        help="Stage 2 only: config names to retrain (auto-detected if omitted).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Seeds for Stage 2 (Stage 1 always uses seed 0).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metr_dir", default="data/metr-la")
    parser.add_argument("--bay_dir",  default="data/pems-bay")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(args.device)
    results_path = os.path.join(OUTPUT_DIR, "sweep_results.json")

    # Load METR-LA
    print("Loading METR-LA...")
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), stats = load_metr_la(
        args.metr_dir, seq_len=12, pred_len=12
    )
    num_nodes = X_tr.shape[2]
    print(f"  Train {X_tr.shape} | Val {X_va.shape} | Test {X_te.shape} | N={num_nodes}")

    adj = load_adjacency(os.path.join(args.metr_dir, "adj_mx.pkl"), num_nodes, device)

    tr_loader = make_dataloader(X_tr, Y_tr, 64, shuffle=True)
    va_loader = make_dataloader(X_va, Y_va, 64, shuffle=False)
    te_loader = make_dataloader(X_te, Y_te, 64, shuffle=False)

    # Load PEMS-BAY raw for city transfer eval
    print("Loading PEMS-BAY raw data...")
    bay_raw = load_bay_raw(args.bay_dir)
    print(f"  PEMS-BAY shape: {bay_raw.shape}")

    # Load existing results
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    base_cfg = {
        "epochs": 150, "weight_decay": 1e-4,
        "seq_len": 12, "pred_len": 12,
    }

    # Determine which configs to run
    if args.stage == 1:
        configs_to_run = STAGE1_CONFIGS
        seeds = [0]
        print(f"\n=== Stage 1: {len(configs_to_run)} configs × 1 seed ===")
    else:
        # Stage 2: auto-detect best-2 from Stage 1 if not specified
        if args.best_configs is None:
            # Pick top-2 by in-dist MAE (lower = better) among Stage 1 results
            stage1 = {k: v for k, v in all_results.items()
                      if isinstance(v, dict) and "in_dist_mae_seed0" in v}
            if len(stage1) < 2:
                print("Stage 1 results not found. Run Stage 1 first.")
                sys.exit(1)
            ranked = sorted(stage1.items(), key=lambda kv: kv[1]["in_dist_mae_seed0"])
            best2 = [ranked[0][0], ranked[1][0]]
            print(f"Auto-detected best-2 from Stage 1: {best2}")
        else:
            best2 = args.best_configs

        configs_to_run = [c for c in STAGE1_CONFIGS if c[0] in best2]
        seeds = args.seeds
        print(f"\n=== Stage 2: {len(configs_to_run)} configs × {len(seeds)} seeds ===")
        print(f"  Configs: {[c[0] for c in configs_to_run]}")

    for name, hidden_dim, embed_dim, lr, patience, dropout in configs_to_run:
        cfg = {**base_cfg,
               "hidden_dim": hidden_dim,
               "embed_dim":  embed_dim,
               "lr":         lr,
               "patience":   patience,
               "dropout":    dropout}

        if name not in all_results:
            all_results[name] = {"config": {
                "hidden_dim": hidden_dim, "embed_dim": embed_dim,
                "lr": lr, "patience": patience, "dropout": dropout,
            }}

        seed_maes_indist = []
        seed_deltas      = []
        seed_city_maes   = []

        for seed in seeds:
            seed_key = f"seed{seed}"
            if f"in_dist_mae_{seed_key}" in all_results[name]:
                print(f"[SKIP] {name} {seed_key} already done")
                seed_maes_indist.append(all_results[name][f"in_dist_mae_{seed_key}"])
                seed_deltas.append(all_results[name][f"delta_sensor_{seed_key}"])
                seed_city_maes.append(all_results[name][f"city_src_mae_{seed_key}"])
                continue

            print(f"\n{'='*60}")
            print(f"Config: {name} | seed={seed}")
            print(f"  hidden={hidden_dim} embed={embed_dim} lr={lr} patience={patience} dropout={dropout}")
            print(f"{'='*60}")
            set_seed(seed)
            t0 = time.time()

            model, test_m = train_gwnet_config(cfg, num_nodes, adj,
                                               tr_loader, va_loader, te_loader,
                                               device, stats)
            in_dist_mae = test_m["MAE"]
            print(f"  In-dist MAE (mph): {in_dist_mae:.4f}")

            # Sensor dropout evaluation
            model.eval()
            with torch.no_grad():
                sd = eval_sensor_dropout(model, X_te, Y_te, adj, device)
            delta_s = sd["delta_sensor"]
            print(f"  Δ_sensor (30%): {delta_s:+.4f}")

            # City transfer: source-only adj (P16 protocol)
            with torch.no_grad():
                ct = eval_city_transfer_source_only(
                    model, bay_raw, stats, adj, device, N=num_nodes
                )
            city_mae = ct["city_src_mae_mean"]
            print(f"  City MAE (src adj, norm): {city_mae:.4f} ± {ct['city_src_mae_std']:.4f}")
            print(f"  Elapsed: {time.time() - t0:.1f}s")

            all_results[name][f"in_dist_mae_{seed_key}"]  = in_dist_mae
            all_results[name][f"delta_sensor_{seed_key}"] = float(delta_s)
            all_results[name][f"city_src_mae_{seed_key}"] = float(city_mae)

            seed_maes_indist.append(in_dist_mae)
            seed_deltas.append(float(delta_s))
            seed_city_maes.append(float(city_mae))

            # Save after each seed
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

        # Aggregate across seeds
        if len(seed_maes_indist) > 0:
            all_results[name]["in_dist_mae_mean"] = float(np.mean(seed_maes_indist))
            all_results[name]["in_dist_mae_std"]  = float(np.std(seed_maes_indist))
            all_results[name]["delta_sensor_mean"] = float(np.mean(seed_deltas))
            all_results[name]["delta_sensor_std"]  = float(np.std(seed_deltas))
            all_results[name]["city_src_mae_mean"] = float(np.mean(seed_city_maes))
            all_results[name]["city_src_mae_std"]  = float(np.std(seed_city_maes))
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print(f"{'Config':<30} {'ID MAE':>8} {'Δ_sensor':>10} {'City(src)':>12}")
    print("-"*70)
    for name, *_ in STAGE1_CONFIGS:
        if name in all_results and "in_dist_mae_mean" in all_results[name]:
            r = all_results[name]
            print(f"{name:<30} {r['in_dist_mae_mean']:>8.4f} "
                  f"{r['delta_sensor_mean']:>+10.4f} {r['city_src_mae_mean']:>12.4f}")
    print("="*70)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
