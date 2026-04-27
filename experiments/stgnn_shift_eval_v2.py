"""
Shift Evaluation Script v2 — Direction C: Shift-Aware STGNN
Protocol fixes per Codex result-to-claim gate (2026-04-09):

Fix 1 — City transfer: both source-stat and target-stat normalization;
         denormalized MAE reported; multiple PEMS-BAY subsets (K=5).
Fix 2 — Sensor dropout: zero-masking on ORIGINAL 207-node model
         (no re-instantiation); MAE computed on kept nodes only.
Fix 3 — Temporal shift: weekday/weekend split as a genuine distributional
         separator within METR-LA (no data beyond the dataset is needed).

Reviewer fix v2 (2026-04-09):
Fix 4 — Subset-size sensitivity: 150/207 PEMS-BAY sensor coverage variants
         (zero-pad to 207 for model input; MAE on kept sensors only).
Fix 5 — Structured dropout: hub-based (top-degree nodes) and bursty/contiguous
         (BFS cluster from highest-degree seed) patterns.
Fix 6 — Mask-detection realism: sensitivity to false positive/negative rates
         in the mask-aware adjacency variant.

Reviewer fix v4 (2026-04-10):
Fix 7 — MTGNN/AGCRN inductive baseline: MTGNNLight (cosine-sim top-k graph)
         added to backbone bank; tested on same shift protocol.
Fix 8 — Cross-city fairness control: source-only adjacency variant forces
         DCRNN/STGCN to use METR-LA adj (not PEMS-BAY subgraph) at inference,
         matching GWNet's structural limitation for a fair 3-way comparison.
Fix 9 — Horizon-wise breakdown: per-step MAE for h=1..12 under city transfer
         and sensor dropout to show where degradation concentrates.
Fix 10 — Per-node degree stratification: MAE grouped by degree quartile under
          sensor dropout to link topology structure to fragility.
Fix 11 — Imputation baselines (LOCF, mean): compare zero-mask vs. LOCF vs.
          per-node mean imputation for dropped sensors.

NEVER re-trains. All backbone checkpoints remain frozen.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from stgnn_backbone_bank import (
    BACKBONE_REGISTRY, load_metr_la, load_adjacency,
    make_dataloader, evaluate, set_seed, _load_raw,
)
import sys
import os as _os
_sys_path_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys_path_dir not in sys.path:
    sys.path.insert(0, _sys_path_dir)


# ── PEMS-BAY Multi-Subset Loader ───────────────────────────────────────────────

def load_pems_bay_subsets(data_dir, adj_path_bay, metr_stats, num_nodes_metr,
                          n_subsets=5, seq_len=12, pred_len=12, base_seed=42):
    """Load PEMS-BAY and return K random 207-node subsets for robustness.

    For each subset:
      - source_norm: normalize with METR-LA global mean/std (cross-domain transfer)
      - target_norm: normalize with PEMS-BAY own train statistics (in-distribution scale)
    Returns list of (X, Y, adj, source_stats, target_stats) per subset.
    """
    data = _load_raw(data_dir, "PEMS-BAY")  # (T, 325)
    T, N_bay = data.shape

    # PEMS-BAY train/test split (same 70/10/20 ratio as METR-LA)
    train_end = int(T * 0.7)
    test_start = int(T * 0.8)
    test_raw = data[test_start:]  # (T_test, 325)
    train_raw = data[:train_end]

    # Target-domain statistics (PEMS-BAY own train mean/std per node)
    target_mean = train_raw.mean(axis=0, keepdims=True)   # (1, 325)
    target_std  = train_raw.std(axis=0, keepdims=True) + 1e-8

    # Source-domain statistics (METR-LA global — scalar)
    src_global_mean = float(metr_stats["mean"].mean())
    src_global_std  = float(metr_stats["std"].mean())

    # Adjacency
    with open(adj_path_bay, "rb") as f:
        _, _, adj_bay_full = pickle.load(f, encoding="latin1")

    def make_windows(arr, s_mean, s_std):
        norm = (arr - s_mean) / s_std
        X, Y = [], []
        for t in range(len(norm) - seq_len - pred_len + 1):
            X.append(norm[t:t + seq_len])
            Y.append(norm[t + seq_len:t + seq_len + pred_len])
        return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)

    subsets = []
    for i in range(n_subsets):
        rng = np.random.RandomState(base_seed + i)
        keep = sorted(rng.choice(N_bay, num_nodes_metr, replace=False).tolist())

        # Source-stat normalized (global scalar)
        X_src, Y_src = make_windows(test_raw[:, keep], src_global_mean, src_global_std)

        # Target-stat normalized (PEMS-BAY per-node, but on kept subset)
        t_mean = target_mean[:, keep]  # (1, num_nodes_metr)
        t_std  = target_std[:, keep]
        X_tgt, Y_tgt = make_windows(test_raw[:, keep], t_mean, t_std)

        # Subgraph adjacency (normalized)
        adj_sub = adj_bay_full[np.ix_(keep, keep)].astype(np.float32)
        adj_t = torch.FloatTensor(adj_sub)
        deg = adj_t.sum(dim=1).clamp(min=1e-8)
        D_inv = deg.pow(-0.5)
        adj_norm = D_inv.unsqueeze(1) * adj_t * D_inv.unsqueeze(0)

        subsets.append({
            "subset_id": i,
            "keep_idx": keep,
            "X_src": X_src, "Y_src": Y_src,
            "X_tgt": X_tgt, "Y_tgt": Y_tgt,
            "adj_norm": adj_norm,
            "src_stats": {"mean": np.full((1, num_nodes_metr), src_global_mean),
                          "std":  np.full((1, num_nodes_metr), src_global_std)},
            "tgt_stats": {"mean": t_mean, "std": t_std},
        })
    return subsets


# ── Sensor Dropout (Zero-Mask, Same Model) ─────────────────────────────────────

@torch.no_grad()
def evaluate_sensor_zeromask(model, X_te, Y_te, adj, device,
                              dropout_ratio=0.30, seed=42, batch_size=64):
    """Sensor dropout via zero-masking: SAME 207-node model, dropped sensors = 0.

    Clean reference: full X_te → MAE on kept nodes only.
    Masked: X_te with drop_idx zeroed → MAE on kept nodes only.
    Δ_sensor = MAE_masked_kept - MAE_clean_kept.
    """
    rng = np.random.RandomState(seed)
    N = X_te.shape[2]
    n_drop = int(N * dropout_ratio)
    drop_idx = rng.choice(N, n_drop, replace=False)
    keep_idx = [i for i in range(N) if i not in drop_idx]

    # Loaders
    clean_loader = make_dataloader(X_te, Y_te, batch_size, shuffle=False)

    X_masked = X_te.copy()
    X_masked[:, :, drop_idx] = 0.0
    masked_loader = make_dataloader(X_masked, Y_te, batch_size, shuffle=False)

    model.eval()

    def collect_preds_targets(loader):
        preds_list, tgts_list = [], []
        for X_b, Y_b in loader:
            X_b = X_b.to(device)
            pred = model(X_b, adj)           # (B, N, pred_len)
            preds_list.append(pred.cpu())
            tgts_list.append(Y_b.permute(0, 2, 1))  # (B, N, pred_len)
        return torch.cat(preds_list, 0), torch.cat(tgts_list, 0)

    pred_clean, tgt_clean   = collect_preds_targets(clean_loader)
    pred_masked, tgt_masked = collect_preds_targets(masked_loader)

    # MAE on kept nodes only
    mae_clean  = (pred_clean[:, keep_idx, :]  - tgt_clean[:, keep_idx, :]).abs().mean().item()
    mae_masked = (pred_masked[:, keep_idx, :] - tgt_masked[:, keep_idx, :]).abs().mean().item()

    return {
        "clean_MAE_kept":  mae_clean,
        "masked_MAE_kept": mae_masked,
        "delta_sensor":    mae_masked - mae_clean,
        "kept_nodes": len(keep_idx),
        "dropped_nodes": n_drop,
    }


# ── Structured Sensor Dropout ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sensor_structured_dropout(model, X_te, Y_te, adj_norm_np, device,
                                        dropout_ratio=0.30, mode="hub", seed=42,
                                        batch_size=64):
    """Structured sensor dropout using topology-aware selection.

    mode='hub':        Drop the top-k highest-degree sensors (most connected hubs).
    mode='contiguous': BFS from the highest-degree seed; drop the nearest-neighbor
                       cluster of size n_drop (simulates a geographic outage).

    Both use the same zero-mask protocol as evaluate_sensor_zeromask:
    same 207-node model, dropped sensors zeroed, MAE on kept sensors only.
    """
    N = X_te.shape[2]
    n_drop = int(N * dropout_ratio)

    # Normalize adjacency for model evaluation (D^{-1/2} A D^{-1/2})
    adj_t_raw = torch.FloatTensor(adj_norm_np)
    deg_raw = adj_t_raw.sum(dim=1).clamp(min=1e-8)
    D_inv = deg_raw.pow(-0.5)
    adj_norm_t = D_inv.unsqueeze(1) * adj_t_raw * D_inv.unsqueeze(0)
    adj_np_normalized = adj_norm_t.numpy()

    # Compute node degrees from adjacency (sum of row weights)
    degree = adj_norm_np.sum(axis=1)  # (N,) — use raw adj for degree ranking
    sorted_by_degree = np.argsort(degree)[::-1]  # descending

    if mode == "hub":
        drop_idx = sorted_by_degree[:n_drop].tolist()
    elif mode == "contiguous":
        # BFS from the highest-degree node; collect first n_drop unique nodes
        from collections import deque
        start = int(sorted_by_degree[0])
        visited = []
        queue = deque([start])
        seen = {start}
        # Build adjacency list from adj_norm_np (threshold >0 for edge)
        adj_thresh = adj_norm_np > 1e-4
        while queue and len(visited) < n_drop:
            node = queue.popleft()
            visited.append(node)
            neighbors = np.where(adj_thresh[node])[0]
            # Sort neighbors by degree descending so we pick hubs first
            neighbors = sorted(neighbors, key=lambda x: -degree[x])
            for nb in neighbors:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        drop_idx = visited[:n_drop]
    else:
        raise ValueError(f"Unknown structured dropout mode: {mode}")

    keep_idx = [i for i in range(N) if i not in drop_idx]

    clean_loader = make_dataloader(X_te, Y_te, batch_size, shuffle=False)
    X_masked = X_te.copy()
    X_masked[:, :, drop_idx] = 0.0
    masked_loader = make_dataloader(X_masked, Y_te, batch_size, shuffle=False)

    adj = torch.FloatTensor(adj_np_normalized).to(device)  # use normalized adj
    model.eval()

    def collect(loader):
        preds, tgts = [], []
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            pred = model(Xb, adj)
            preds.append(pred.cpu())
            tgts.append(Yb.permute(0, 2, 1))
        return torch.cat(preds, 0), torch.cat(tgts, 0)

    pred_clean, tgt_clean   = collect(clean_loader)
    pred_masked, tgt_masked = collect(masked_loader)

    mae_clean  = (pred_clean[:, keep_idx, :]  - tgt_clean[:, keep_idx, :]).abs().mean().item()
    mae_masked = (pred_masked[:, keep_idx, :] - tgt_masked[:, keep_idx, :]).abs().mean().item()

    return {
        "mode": mode,
        "clean_MAE_kept":  mae_clean,
        "masked_MAE_kept": mae_masked,
        "delta_sensor":    mae_masked - mae_clean,
        "kept_nodes": len(keep_idx),
        "dropped_nodes": n_drop,
        "dropped_mean_degree": float(degree[drop_idx].mean()),
        "kept_mean_degree":    float(degree[keep_idx].mean()),
    }


# ── Subset-Size Sensitivity (City Transfer) ────────────────────────────────────

@torch.no_grad()
def evaluate_city_transfer_variable_coverage(model, bay_data, adj_bay_full,
                                              metr_stats, target_mean, target_std,
                                              device, n_keep=150, n_model=207,
                                              n_subsets=5, base_seed=42,
                                              seq_len=12, pred_len=12, batch_size=64):
    """City transfer with variable sensor coverage.

    n_keep < n_model: select n_keep PEMS-BAY sensors; zero-pad remaining (n_model - n_keep)
    input slots; MAE computed on the n_keep kept sensors only.
    This tests whether GWNet's transfer disadvantage persists when fewer target-city
    sensors are available (simulating partial deployment coverage).
    """
    T, N_bay = bay_data.shape
    test_start = int(T * 0.8)
    test_raw = bay_data[test_start:]

    src_global_mean = float(metr_stats["mean"].mean())
    src_global_std  = float(metr_stats["std"].mean())

    mae_list = []
    for i in range(n_subsets):
        rng = np.random.RandomState(base_seed + 100 + i)  # distinct from main subsets
        # First pick n_keep from N_bay PEMS-BAY sensors
        keep_bay = sorted(rng.choice(N_bay, n_keep, replace=False).tolist())
        # Build a (n_model, n_model) adjacency for the padded input:
        # kept slots = subgraph among keep_bay (subset of n_keep); padded slots = disconnected
        adj_sub = adj_bay_full[np.ix_(keep_bay, keep_bay)].astype(np.float32)
        # Embed in n_model × n_model (remaining rows/cols = 0)
        adj_padded = np.zeros((n_model, n_model), dtype=np.float32)
        idx = np.arange(n_keep)
        adj_padded[np.ix_(idx, idx)] = adj_sub
        adj_padded_t = torch.FloatTensor(adj_padded).to(device)
        deg = adj_padded_t.sum(dim=1).clamp(min=1e-8)
        D_inv = deg.pow(-0.5)
        adj_norm = D_inv.unsqueeze(1) * adj_padded_t * D_inv.unsqueeze(0)

        # Normalize test data with source stats
        raw_subset = test_raw[:, keep_bay]  # (T_test, n_keep)
        raw_norm = (raw_subset - src_global_mean) / src_global_std

        # Build windows, embed in n_model-wide array (pad with zeros)
        X_list, Y_list = [], []
        for t in range(len(raw_norm) - seq_len - pred_len + 1):
            x_win = np.zeros((seq_len, n_model), dtype=np.float32)
            y_win = np.zeros((pred_len, n_model), dtype=np.float32)
            x_win[:, :n_keep] = raw_norm[t:t + seq_len]
            y_win[:, :n_keep] = raw_norm[t + seq_len:t + seq_len + pred_len]
            X_list.append(x_win)
            Y_list.append(y_win)
        X_arr = np.stack(X_list)
        Y_arr = np.stack(Y_list)

        loader = make_dataloader(X_arr, Y_arr, batch_size, shuffle=False)
        model.eval()

        preds, tgts = [], []
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            pred = model(Xb, adj_norm)
            preds.append(pred.cpu())
            tgts.append(Yb.permute(0, 2, 1))
        pred_all = torch.cat(preds, 0)  # (S, n_model, pred_len)
        tgt_all  = torch.cat(tgts, 0)

        # Denormalize kept slots using target stats (PEMS-BAY)
        t_mean = target_mean[:, keep_bay]  # (1, n_keep)
        t_std  = target_std[:, keep_bay]
        pred_kept = pred_all[:, :n_keep, :].numpy() * src_global_std + src_global_mean
        # t_mean/t_std shape (1, n_keep) → broadcast to (S, n_keep, pred_len)
        tgt_kept  = tgt_all[:, :n_keep, :].numpy() * t_std[:, :, np.newaxis] + t_mean[:, :, np.newaxis]
        mae = float(np.abs(pred_kept - tgt_kept).mean())
        mae_list.append(mae)

    return {
        "n_keep": n_keep,
        "n_model": n_model,
        "coverage_ratio": n_keep / n_model,
        "city_mae_mean": float(np.mean(mae_list)),
        "city_mae_std":  float(np.std(mae_list)),
        "n_subsets": n_subsets,
    }


# ── Mask-Aware False Positive/Negative Sensitivity ────────────────────────────

@torch.no_grad()
def evaluate_mask_aware_fp_fn(model, X_te, Y_te, adj_metr_np, device,
                               true_dropout_ratio=0.30, fp_rate=0.0, fn_rate=0.0,
                               seed=42, batch_size=64):
    """Test mask-aware GWNet robustness to imperfect mask detection.

    fp_rate: fraction of KEPT sensors mistakenly flagged as dropped (false positives)
    fn_rate: fraction of DROPPED sensors missed by detector (false negatives)

    A realistic detector has fp=0.05 (5% false alarm), fn=0.10 (10% miss rate).
    Reports: MAE on kept sensors under the noisy mask vs. clean mask.
    """
    from stgnn_backbone_bank import GWNet
    if not hasattr(model, 'node_emb1'):
        return None  # Not GWNet adaptive; skip

    rng = np.random.RandomState(seed)
    N = X_te.shape[2]
    n_drop = int(N * true_dropout_ratio)

    # True dropped sensors
    true_drop = set(rng.choice(N, n_drop, replace=False).tolist())
    true_keep = [i for i in range(N) if i not in true_drop]

    # Noisy mask: FP = keep sensors flagged as dropped, FN = dropped sensors missed
    n_fp = int(len(true_keep) * fp_rate)
    n_fn = int(len(true_drop) * fn_rate)
    fp_sensors = set(rng.choice(list(true_keep), n_fp, replace=False).tolist()) if n_fp > 0 else set()
    fn_sensors = set(rng.choice(list(true_drop), n_fn, replace=False).tolist()) if n_fn > 0 else set()

    # Detected drop set = (true_drop - fn) + fp
    detected_drop = (true_drop - fn_sensors) | fp_sensors
    eval_keep = [i for i in range(N) if i not in true_drop]  # MAE on truly kept sensors

    # Zero-mask input with TRUE drops
    X_masked = X_te.copy()
    X_masked[:, :, list(true_drop)] = 0.0

    # Normalize the adjacency for model evaluation
    adj_t_raw = torch.FloatTensor(adj_metr_np)
    deg_raw = adj_t_raw.sum(dim=1).clamp(min=1e-8)
    D_inv_sqrt = deg_raw.pow(-0.5)
    adj_metr_norm = D_inv_sqrt.unsqueeze(1) * adj_t_raw * D_inv_sqrt.unsqueeze(0)
    adj_t = adj_metr_norm.to(device)

    def get_mask_aware_adj(model_m, drop_set):
        with torch.no_grad():
            a = torch.softmax(model_m.node_emb1 @ model_m.node_emb2.T, dim=-1).clone()
            for idx in drop_set:
                a[idx, :] = 0.0
                a[:, idx] = 0.0
            row_sum = a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            a = a / row_sum
        return a

    loader = make_dataloader(X_masked, Y_te, batch_size, shuffle=False)
    model.eval()

    # Perfect mask (true_drop known)
    adj_perfect = get_mask_aware_adj(model, true_drop)
    # Noisy mask (detected_drop)
    adj_noisy   = get_mask_aware_adj(model, detected_drop)

    def collect_with_adj(adj_use):
        preds, tgts = [], []
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            # Temporarily override adaptive_adj behavior by directly computing
            with torch.no_grad():
                if Xb.dim() == 4:
                    Xb = Xb.squeeze(-1)
                B, T, n = Xb.shape
                x = Xb.permute(0, 2, 1)
                x = model.relu(model.input_proj(x))
                x = model.relu(model.graph_conv_layer(x, adj_use, model.graph_conv1))
                x = model.dropout(x)
                x = model.relu(model.graph_conv_layer(x, adj_use, model.graph_conv2))
                x = model.dropout(x)
                pred = model.output(x)
            preds.append(pred.cpu())
            tgts.append(Yb.permute(0, 2, 1))
        return torch.cat(preds, 0), torch.cat(tgts, 0)

    pred_perf, tgt_perf = collect_with_adj(adj_perfect)
    pred_noisy, tgt_noisy = collect_with_adj(adj_noisy)

    mae_perfect = (pred_perf[:, eval_keep, :] - tgt_perf[:, eval_keep, :]).abs().mean().item()
    mae_noisy   = (pred_noisy[:, eval_keep, :] - tgt_noisy[:, eval_keep, :]).abs().mean().item()

    return {
        "fp_rate": fp_rate, "fn_rate": fn_rate,
        "n_fp": n_fp, "n_fn": n_fn,
        "detected_drop_size": len(detected_drop),
        "mae_perfect_mask": mae_perfect,
        "mae_noisy_mask":   mae_noisy,
        "delta_from_perfect": mae_noisy - mae_perfect,
    }


# ── Temporal Shift: Weekday vs Weekend ─────────────────────────────────────────

def get_weekday_weekend_split(data_dir, seq_len=12, pred_len=12):
    """Use METR-LA weekday vs weekend as a genuine temporal distributional split.
    METR-LA starts 2012-03-01 (Thursday). 5-min intervals → 288 steps/day.
    Training is on the full dataset (weekday-dominated).
    We evaluate separately on weekday and weekend windows from the test set.
    Returns (X_weekday, Y_weekday), (X_weekend, Y_weekend), stats.
    """
    data = _load_raw(data_dir, "METR-LA")  # (T, 207)
    T = data.shape[0]

    train_end = int(T * 0.7)
    test_start = int(T * 0.8)

    # Train statistics (same as backbone bank)
    mean = data[:train_end].mean(axis=0, keepdims=True)
    std  = data[:train_end].std(axis=0, keepdims=True) + 1e-8

    # Full dataset normalized
    data_norm = (data - mean) / std

    # METR-LA starts 2012-03-01 00:00 (Thursday = day index 3; Mon=0)
    # 5-min intervals → 288 per day
    steps_per_day = 288
    start_day_of_week = 3  # Thursday

    # Day-of-week for each timestep
    day_idx = ((np.arange(T) // steps_per_day) + start_day_of_week) % 7
    is_weekend = (day_idx == 5) | (day_idx == 6)  # Sat=5, Sun=6

    # Only use test period
    test_norm = data_norm[test_start:]
    is_weekend_test = is_weekend[test_start:]

    def make_windows_masked(arr, mask):
        """Windows where the START timestep is in mask."""
        X, Y = [], []
        n = len(arr) - seq_len - pred_len + 1
        for t in range(n):
            if mask[t]:
                X.append(arr[t:t + seq_len])
                Y.append(arr[t + seq_len:t + seq_len + pred_len])
        if not X:
            return None, None
        return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)

    X_wd, Y_wd = make_windows_masked(test_norm, ~is_weekend_test)
    X_we, Y_we = make_windows_masked(test_norm,  is_weekend_test)
    stats = {"mean": mean, "std": std}
    return (X_wd, Y_wd), (X_we, Y_we), stats


# ── Horizon-Wise MAE Breakdown ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_horizon_wise(model, loader, adj, device):
    """Compute MAE for each prediction horizon step h=1..pred_len.

    Returns list of length pred_len, where entry h is the mean absolute error
    at that specific forecast horizon.
    """
    model.eval()
    preds_list, tgts_list = [], []
    for X_b, Y_b in loader:
        X_b = X_b.to(device)
        pred = model(X_b, adj)                      # (B, N, pred_len)
        preds_list.append(pred.cpu())
        tgts_list.append(Y_b.permute(0, 2, 1))     # (B, N, pred_len)
    preds  = torch.cat(preds_list, 0)               # (total, N, pred_len)
    tgts   = torch.cat(tgts_list,  0)
    # MAE per horizon: average over batch and nodes
    horizon_mae = (preds - tgts).abs().mean(dim=(0, 1))   # (pred_len,)
    return horizon_mae.tolist()


# ── Per-Node Degree Stratification ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_node_degree(model, X_te, Y_te, adj_norm_t, adj_np, device,
                              dropout_ratio=0.30, seed=42, batch_size=64,
                              n_quartiles=4):
    """Evaluate sensor dropout MAE stratified by node degree quartile.

    Partitions nodes into n_quartiles groups by row-sum degree in adj_np.
    For each group, computes clean MAE and masked MAE on those group nodes only.

    Returns dict keyed by quartile label ('Q1'...'Q4') with:
        clean_MAE, masked_MAE, delta, mean_degree, n_nodes
    """
    rng = np.random.RandomState(seed)
    N = X_te.shape[2]
    n_drop = int(N * dropout_ratio)
    drop_idx = set(rng.choice(N, n_drop, replace=False).tolist())
    keep_idx = [i for i in range(N) if i not in drop_idx]

    degree = adj_np.sum(axis=1)  # (N,) row-sum degrees

    # Quartile boundaries
    quantiles = np.percentile(degree, [25 * q for q in range(1, n_quartiles)])
    # Assign each node to a quartile (0-indexed)
    quartile_id = np.digitize(degree, quantiles)  # values 0..n_quartiles-1

    X_masked = X_te.copy()
    X_masked[:, :, list(drop_idx)] = 0.0

    clean_loader  = make_dataloader(X_te,     Y_te, batch_size, shuffle=False)
    masked_loader = make_dataloader(X_masked, Y_te, batch_size, shuffle=False)

    model.eval()

    def collect(loader):
        p, t = [], []
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            pred = model(Xb, adj_norm_t)
            p.append(pred.cpu())
            t.append(Yb.permute(0, 2, 1))
        return torch.cat(p, 0), torch.cat(t, 0)

    pred_clean, tgt_clean   = collect(clean_loader)
    pred_masked, tgt_masked = collect(masked_loader)

    results = {}
    for q in range(n_quartiles):
        nodes_q = [i for i in keep_idx if quartile_id[i] == q]
        if not nodes_q:
            continue
        label = f"Q{q+1}"
        mae_c = (pred_clean[:, nodes_q, :]  - tgt_clean[:, nodes_q, :]).abs().mean().item()
        mae_m = (pred_masked[:, nodes_q, :] - tgt_masked[:, nodes_q, :]).abs().mean().item()
        results[label] = {
            "clean_MAE": mae_c,
            "masked_MAE": mae_m,
            "delta": mae_m - mae_c,
            "mean_degree": float(degree[nodes_q].mean()),
            "n_nodes": len(nodes_q),
        }
    return results


# ── Imputation Baselines for Sensor Dropout ────────────────────────────────────

@torch.no_grad()
def evaluate_imputation_baseline(model, X_te, Y_te, adj_norm_t, device,
                                  dropout_ratio=0.30, seed=42, batch_size=64,
                                  method="locf"):
    """Compare imputation strategies for dropped sensors.

    method='locf': fill dropped sensor channels with the last observed value
                   (the value at the last timestep before the window, approximated
                   by the last timestep of the input window itself).
    method='mean': fill with per-node training-split mean (approximated by
                   per-node mean over the test windows).

    Returns: {clean_MAE, imputed_MAE, zero_MAE, delta_imputed, delta_zero}
    evaluated on kept sensors only.
    """
    rng = np.random.RandomState(seed)
    N = X_te.shape[2]
    n_drop = int(N * dropout_ratio)
    drop_idx = rng.choice(N, n_drop, replace=False).tolist()
    keep_idx = [i for i in range(N) if i not in drop_idx]

    # Build imputed input
    X_zero   = X_te.copy()
    X_impute = X_te.copy()

    if method == "locf":
        # Last observed value per window: X_te[:, -1, drop_idx] → replicate across T
        last_val = X_te[:, -1:, drop_idx]          # (windows, 1, n_drop)
        X_impute[:, :, drop_idx] = np.broadcast_to(last_val, X_te[:, :, drop_idx].shape).copy()
    elif method == "mean":
        # Per-node mean across all test windows (approximation of training mean)
        node_mean = X_te[:, :, drop_idx].mean(axis=(0, 1), keepdims=True)  # (1, 1, n_drop)
        X_impute[:, :, drop_idx] = np.broadcast_to(
            node_mean, X_te[:, :, drop_idx].shape
        ).copy()

    X_zero[:, :, drop_idx] = 0.0

    clean_loader   = make_dataloader(X_te,     Y_te, batch_size, shuffle=False)
    zero_loader    = make_dataloader(X_zero,   Y_te, batch_size, shuffle=False)
    impute_loader  = make_dataloader(X_impute, Y_te, batch_size, shuffle=False)

    model.eval()

    def collect(loader):
        p, t = [], []
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            pred = model(Xb, adj_norm_t)
            p.append(pred.cpu())
            t.append(Yb.permute(0, 2, 1))
        return torch.cat(p, 0), torch.cat(t, 0)

    pred_c, tgt_c = collect(clean_loader)
    pred_z, tgt_z = collect(zero_loader)
    pred_i, tgt_i = collect(impute_loader)

    mae_c = (pred_c[:, keep_idx, :] - tgt_c[:, keep_idx, :]).abs().mean().item()
    mae_z = (pred_z[:, keep_idx, :] - tgt_z[:, keep_idx, :]).abs().mean().item()
    mae_i = (pred_i[:, keep_idx, :] - tgt_i[:, keep_idx, :]).abs().mean().item()

    return {
        "method": method,
        "clean_MAE": mae_c,
        "zero_MAE": mae_z,
        "imputed_MAE": mae_i,
        "delta_zero": mae_z - mae_c,
        "delta_imputed": mae_i - mae_c,
        "imputation_improvement": (mae_z - mae_i) / max(mae_z - mae_c, 1e-8),
    }


# ── Source-Only Adjacency Control (Cross-City Fairness) ───────────────────────

@torch.no_grad()
def evaluate_city_transfer_source_adj(model, bay_subsets, adj_metr_src, device,
                                       metr_stats, N):
    """City transfer using source-city (METR-LA) adjacency for ALL models.

    This is the fairness control: DCRNN/STGCN normally receive the PEMS-BAY
    subgraph adjacency (target-city topology).  When forced to use adj_metr,
    all three backbones (DCRNN, STGCN, GWNet) operate under the same structural
    constraint — their source-city graph is applied to target-city sensor data.

    Returns: mean city_src_MAE ± std over subsets.
    """
    city_src_maes = []
    clean_norm_mae = None

    for sub in bay_subsets:
        src_loader = make_dataloader(sub["X_src"], sub["Y_src"], 64, shuffle=False)
        # Force source-city adj for all models (same adj_metr used during training)
        src_metrics = evaluate(model, src_loader, adj_metr_src, device)
        city_src_maes.append(src_metrics["MAE"])

    return {
        "city_src_MAE_mean": float(np.mean(city_src_maes)),
        "city_src_MAE_std":  float(np.std(city_src_maes)),
        "n_subsets": len(bay_subsets),
        "note": "source-only adjacency control (METR-LA adj applied to PEMS-BAY data)",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry",        default="baselines/stgnn_protocol_registry.json")
    parser.add_argument("--backbones",       nargs="+", default=["dcrnn", "gwnet", "gru"])
    parser.add_argument("--source_dataset",  default="metr-la",
                        help="Source dataset key used as the checkpoint suffix filter. "
                             "v5: use 'metr-la' (default) for the main pipeline; for "
                             "PEMS04 source models use stgnn_shift_eval_v5_pems04.py instead.")
    parser.add_argument("--output_dir",      default="results/tier1/")
    parser.add_argument("--metr_la_dir",     default="data/metr-la")
    parser.add_argument("--pems_bay_dir",    default="data/pems-bay")
    parser.add_argument("--adj_path_metr",   default="data/metr-la/adj_mx.pkl")
    parser.add_argument("--adj_path_bay",    default="data/pems-bay/adj_mx_bay.pkl")
    parser.add_argument("--n_bay_subsets",   type=int, default=5,
                        help="Number of random PEMS-BAY 207-node subsets for robustness")
    parser.add_argument("--sensor_dropout",  type=float, default=0.30)
    # Reviewer fix v2 flags
    parser.add_argument("--subset_size_sensitivity", action="store_true",
                        help="Run city transfer with variable coverage (150/207 nodes)")
    parser.add_argument("--structured_dropout", action="store_true",
                        help="Run hub-based and contiguous structured sensor dropout")
    parser.add_argument("--mask_fp_fn_sensitivity", action="store_true",
                        help="Test mask-aware GWNet robustness to FP/FN detection errors")
    parser.add_argument("--multi_mask_seeds", type=int, default=5,
                        help="Number of random mask seeds for multi-mask averaging")
    # Reviewer fix v4 flags
    parser.add_argument("--horizon_wise",    action="store_true",
                        help="[v4] Compute per-horizon MAE breakdown (h=1..12)")
    parser.add_argument("--per_node_centrality", action="store_true",
                        help="[v4] Stratify sensor-dropout MAE by node degree quartile")
    parser.add_argument("--imputation_baselines", action="store_true",
                        help="[v4] Compare LOCF and mean imputation vs zero-masking")
    parser.add_argument("--source_only_adj_control", action="store_true",
                        help="[v4] City transfer fairness: force METR-LA adj for all models")
    parser.add_argument("--capacity_matched_gwnet", action="store_true",
                        help="[v4] Include gwnet_cm in evaluation if checkpoints exist")
    # Accept True/False strings (for convenience when called from shell scripts)
    for _flag in ["horizon_wise", "per_node_centrality", "imputation_baselines",
                  "source_only_adj_control", "capacity_matched_gwnet",
                  "subset_size_sensitivity", "structured_dropout", "mask_fp_fn_sensitivity"]:
        # Allow --flag True / --flag False syntax in addition to store_true
        pass  # already defined above; shell callers should use bare --flag not --flag True
    parser.add_argument("--sensor_zero_mask",    default="True",
                        help="(legacy; zero-masking is always on; kept for CLI compatibility)")
    parser.add_argument("--gwnet_fixed_prior_target", default="False",
                        help="[v4] Also run GWNet[fixed] on city transfer for comparison")
    parser.add_argument("--shifts",          nargs="+", default=["city", "sensor", "temporal"],
                        help="(legacy; all shifts run by default; kept for CLI compatibility)")
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Expand capacity_matched_gwnet flag → add gwnet_cm to backbones if checkpoints will exist
    if args.capacity_matched_gwnet and "gwnet_cm" not in args.backbones:
        args.backbones = list(args.backbones) + ["gwnet_cm"]

    gwnet_fixed_prior_target = str(args.gwnet_fixed_prior_target).lower() == "true"

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load registry
    with open(args.registry) as f:
        registry = json.load(f)

    # ── Load METR-LA ────────────────────────────────────────────────────────
    print("Loading METR-LA...")
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), metr_stats = load_metr_la(args.metr_la_dir)
    N = X_te.shape[2]
    adj_metr = load_adjacency(args.adj_path_metr, N, device)
    clean_loader = make_dataloader(X_te, Y_te, 64, shuffle=False)

    # ── Weekday/Weekend split ────────────────────────────────────────────────
    print("Building weekday/weekend split...")
    (X_wd, Y_wd), (X_we, Y_we), _ = get_weekday_weekend_split(args.metr_la_dir)
    print(f"  Weekday windows: {len(X_wd) if X_wd is not None else 0}")
    print(f"  Weekend windows: {len(X_we) if X_we is not None else 0}")
    wd_loader = make_dataloader(X_wd, Y_wd, 64, shuffle=False) if X_wd is not None else None
    we_loader = make_dataloader(X_we, Y_we, 64, shuffle=False) if X_we is not None else None

    # ── PEMS-BAY subsets ────────────────────────────────────────────────────
    print(f"Loading PEMS-BAY ({args.n_bay_subsets} subsets)...")
    bay_subsets = load_pems_bay_subsets(
        args.pems_bay_dir, args.adj_path_bay, metr_stats, N,
        n_subsets=args.n_bay_subsets,
    )
    print(f"  Loaded {len(bay_subsets)} subsets × 207 nodes")

    # Load raw PEMS-BAY data + full adjacency for variable-coverage experiment
    bay_data_raw = None
    adj_bay_full_np = None
    bay_target_mean = bay_target_std = None
    if args.subset_size_sensitivity:
        from stgnn_backbone_bank import _load_raw
        print("  Loading PEMS-BAY raw data for subset-size sensitivity...")
        bay_data_raw = _load_raw(args.pems_bay_dir, "PEMS-BAY")  # (T, 325)
        Tb, Nb = bay_data_raw.shape
        train_end_b = int(Tb * 0.7)
        bay_target_mean = bay_data_raw[:train_end_b].mean(axis=0, keepdims=True)
        bay_target_std  = bay_data_raw[:train_end_b].std(axis=0, keepdims=True) + 1e-8
        with open(args.adj_path_bay, "rb") as f_bay:
            _, _, adj_bay_full_np = pickle.load(f_bay, encoding="latin1")
        adj_bay_full_np = adj_bay_full_np.astype(np.float32)
        print(f"  PEMS-BAY raw: {bay_data_raw.shape}, adj: {adj_bay_full_np.shape}")

    # Load adj_metr as numpy for structured dropout
    import pickle as _pkl
    with open(args.adj_path_metr, "rb") as f_m:
        _, _, adj_metr_np = _pkl.load(f_m, encoding="latin1")
    adj_metr_np = adj_metr_np[:N, :N].astype(np.float32)

    all_results = {}

    for backbone_name in args.backbones:
        bkeys = sorted([k for k in registry
                        if k.startswith(f"{backbone_name}__{args.source_dataset}__")])
        if not bkeys:
            print(f"[WARN] No checkpoints for {backbone_name}. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {backbone_name.upper()}")
        print(f"{'='*60}")

        bb_results = {}

        for key in bkeys:
            entry  = registry[key]
            seed   = entry["seed"]
            ckpt   = torch.load(entry["checkpoint"], map_location=device,
                                weights_only=False)
            cfg    = ckpt["config"]

            model_cls = BACKBONE_REGISTRY[backbone_name]
            model = model_cls(
                num_nodes=N,
                seq_len=cfg["seq_len"],
                pred_len=cfg["pred_len"],
                hidden_dim=cfg["hidden_dim"],
                dropout=cfg["dropout"],
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            seed_res = {"backbone": backbone_name, "seed": seed}

            # ── [A] City Transfer (multi-subset, source+target norm) ─────
            print(f"\n  [A] City transfer — {args.n_bay_subsets} PEMS-BAY subsets | seed={seed}")
            city_src_maes, city_tgt_maes = [], []
            clean_norm_mae = evaluate(model, clean_loader, adj_metr, device)["MAE"]
            clean_denorm_mae = evaluate(model, clean_loader, adj_metr, device,
                                        stats=metr_stats)["MAE"]

            for sub in bay_subsets:
                adj_bay = sub["adj_norm"].to(device)

                # Source-stat: both preds and targets in METR-LA normalized space
                src_loader = make_dataloader(sub["X_src"], sub["Y_src"], 64, shuffle=False)
                src_metrics = evaluate(model, src_loader, adj_bay, device)
                city_src_maes.append(src_metrics["MAE"])

                # Target-stat: targets in PEMS-BAY space; denormalize with tgt stats
                tgt_loader = make_dataloader(sub["X_tgt"], sub["Y_tgt"], 64, shuffle=False)
                tgt_metrics = evaluate(model, tgt_loader, adj_bay, device,
                                       stats=sub["tgt_stats"])
                city_tgt_maes.append(tgt_metrics["MAE"])

            delta_city_src = np.mean(city_src_maes) - clean_norm_mae
            seed_res["city"] = {
                "clean_norm_MAE":         clean_norm_mae,
                "clean_denorm_MAE":       clean_denorm_mae,
                "city_src_MAE_mean":      float(np.mean(city_src_maes)),
                "city_src_MAE_std":       float(np.std(city_src_maes)),
                "city_tgt_MAE_mean":      float(np.mean(city_tgt_maes)),
                "city_tgt_MAE_std":       float(np.std(city_tgt_maes)),
                "delta_city_src":         float(delta_city_src),
                "n_subsets":              args.n_bay_subsets,
            }
            print(f"    clean(norm)={clean_norm_mae:.4f}  "
                  f"bay(src-norm)={np.mean(city_src_maes):.4f}±{np.std(city_src_maes):.4f}  "
                  f"Δ_city(src)={delta_city_src:+.4f}")
            print(f"    bay(tgt-denorm)={np.mean(city_tgt_maes):.4f}±{np.std(city_tgt_maes):.4f}")

            # ── [B] Sensor Dropout (zero-mask, same model) ───────────────
            print(f"  [B] Sensor dropout {int(args.sensor_dropout*100)}% zero-mask | seed={seed}")
            sd_res = evaluate_sensor_zeromask(
                model, X_te, Y_te, adj_metr, device,
                dropout_ratio=args.sensor_dropout,
            )
            seed_res["sensor_zeromask"] = sd_res
            print(f"    clean_kept={sd_res['clean_MAE_kept']:.4f}  "
                  f"masked_kept={sd_res['masked_MAE_kept']:.4f}  "
                  f"Δ_sensor={sd_res['delta_sensor']:+.4f}")

            # ── [C] Temporal: Weekday vs Weekend ─────────────────────────
            print(f"  [C] Weekday vs Weekend shift | seed={seed}")
            if wd_loader and we_loader:
                wd_metrics = evaluate(model, wd_loader, adj_metr, device)
                we_metrics = evaluate(model, we_loader, adj_metr, device)
                delta_temporal = we_metrics["MAE"] - wd_metrics["MAE"]
                seed_res["temporal_wdwe"] = {
                    "weekday_MAE": wd_metrics["MAE"],
                    "weekend_MAE": we_metrics["MAE"],
                    "delta_temporal": delta_temporal,
                }
                print(f"    weekday={wd_metrics['MAE']:.4f}  "
                      f"weekend={we_metrics['MAE']:.4f}  "
                      f"Δ_temporal={delta_temporal:+.4f}")
            else:
                print("    [SKIP] insufficient weekday/weekend windows")

            # ── [D] Structured Dropout (hub + contiguous) ─────────────────
            if args.structured_dropout:
                print(f"  [D] Structured dropout (hub + contiguous) | seed={seed}")
                for sdmode in ["hub", "contiguous"]:
                    sdr = evaluate_sensor_structured_dropout(
                        model, X_te, Y_te, adj_metr_np, device,
                        dropout_ratio=args.sensor_dropout, mode=sdmode, seed=seed,
                    )
                    seed_res[f"structured_dropout_{sdmode}"] = sdr
                    print(f"    [{sdmode}] Δ_sensor={sdr['delta_sensor']:+.4f}  "
                          f"drop_deg={sdr['dropped_mean_degree']:.3f}  "
                          f"keep_deg={sdr['kept_mean_degree']:.3f}")

            # ── [E] Subset-size sensitivity (city transfer) ───────────────
            if args.subset_size_sensitivity and bay_data_raw is not None:
                print(f"  [E] Subset-size sensitivity (150/207 coverage) | seed={seed}")
                seed_res["subset_sensitivity"] = {}
                for n_keep in [150, 207]:
                    cov_res = evaluate_city_transfer_variable_coverage(
                        model, bay_data_raw, adj_bay_full_np, metr_stats,
                        bay_target_mean, bay_target_std, device,
                        n_keep=n_keep, n_model=N, n_subsets=3, base_seed=seed * 100,
                    )
                    seed_res["subset_sensitivity"][str(n_keep)] = cov_res
                    print(f"    n_keep={n_keep} ({cov_res['coverage_ratio']:.0%} coverage)  "
                          f"city_MAE={cov_res['city_mae_mean']:.4f}±{cov_res['city_mae_std']:.4f}")

            # ── [F] Mask-aware FP/FN sensitivity (GWNet only) ────────────
            if args.mask_fp_fn_sensitivity and backbone_name in ("gwnet", "gwnet_cm"):
                print(f"  [F] Mask-aware FP/FN sensitivity | seed={seed}")
                seed_res["mask_fp_fn"] = {}
                for fp, fn in [(0.0, 0.0), (0.05, 0.10), (0.10, 0.20)]:
                    fpfn_res = evaluate_mask_aware_fp_fn(
                        model, X_te, Y_te, adj_metr_np, device,
                        true_dropout_ratio=args.sensor_dropout,
                        fp_rate=fp, fn_rate=fn, seed=seed,
                    )
                    if fpfn_res:
                        key_s = f"fp{int(fp*100)}_fn{int(fn*100)}"
                        seed_res["mask_fp_fn"][key_s] = fpfn_res
                        print(f"    FP={fp:.0%}/FN={fn:.0%}  "
                              f"MAE_perfect={fpfn_res['mae_perfect_mask']:.4f}  "
                              f"MAE_noisy={fpfn_res['mae_noisy_mask']:.4f}  "
                              f"Δ={fpfn_res['delta_from_perfect']:+.4f}")

            # ── [G] Horizon-wise MAE breakdown (v4) ─────────────────────
            if args.horizon_wise:
                print(f"  [G] Horizon-wise breakdown | seed={seed}")
                hw_clean = evaluate_horizon_wise(model, clean_loader, adj_metr, device)
                # Also compute horizon-wise under sensor dropout (30%)
                X_masked30 = X_te.copy()
                rng30 = np.random.RandomState(42)
                drop30 = rng30.choice(N, int(N * 0.30), replace=False).tolist()
                X_masked30[:, :, drop30] = 0.0
                masked_loader30 = make_dataloader(X_masked30, Y_te, 64, shuffle=False)
                hw_masked = evaluate_horizon_wise(model, masked_loader30, adj_metr, device)
                # City transfer horizon-wise (first subset, source-stat norm)
                src_loader0 = make_dataloader(
                    bay_subsets[0]["X_src"], bay_subsets[0]["Y_src"], 64, shuffle=False)
                adj_bay0 = bay_subsets[0]["adj_norm"].to(device)
                hw_city = evaluate_horizon_wise(model, src_loader0, adj_bay0, device)
                seed_res["horizon_wise"] = {
                    "clean": hw_clean,
                    "sensor_masked_30pct": hw_masked,
                    "city_transfer_src": hw_city,
                }
                print(f"    clean  h1={hw_clean[0]:.4f} h6={hw_clean[5]:.4f} h12={hw_clean[-1]:.4f}")
                print(f"    masked h1={hw_masked[0]:.4f} h6={hw_masked[5]:.4f} h12={hw_masked[-1]:.4f}")
                print(f"    city   h1={hw_city[0]:.4f}  h6={hw_city[5]:.4f}  h12={hw_city[-1]:.4f}")

            # ── [H] Per-node degree stratification (v4) ──────────────────
            if args.per_node_centrality:
                print(f"  [H] Per-node degree stratification | seed={seed}")
                # Build normalized adj tensor (CPU) for evaluate functions
                adj_metr_t_cpu = adj_metr.cpu()
                degree_res = evaluate_per_node_degree(
                    model, X_te, Y_te, adj_metr, adj_metr_np, device,
                    dropout_ratio=args.sensor_dropout, seed=42,
                )
                seed_res["per_node_degree"] = degree_res
                for q_label, q_res in degree_res.items():
                    print(f"    [{q_label}] deg={q_res['mean_degree']:.2f}  "
                          f"clean={q_res['clean_MAE']:.4f}  "
                          f"masked={q_res['masked_MAE']:.4f}  "
                          f"Δ={q_res['delta']:+.4f}  n={q_res['n_nodes']}")

            # ── [I] Imputation baselines: LOCF + mean (v4) ───────────────
            if args.imputation_baselines:
                print(f"  [I] Imputation baselines (LOCF + mean) | seed={seed}")
                seed_res["imputation"] = {}
                for imp_method in ["locf", "mean"]:
                    imp_res = evaluate_imputation_baseline(
                        model, X_te, Y_te, adj_metr, device,
                        dropout_ratio=args.sensor_dropout, seed=42, method=imp_method,
                    )
                    seed_res["imputation"][imp_method] = imp_res
                    print(f"    [{imp_method}] delta_zero={imp_res['delta_zero']:+.4f}  "
                          f"delta_imputed={imp_res['delta_imputed']:+.4f}  "
                          f"improvement={imp_res['imputation_improvement']:.1%}")

            # ── [J] Source-only adjacency control (v4) ───────────────────
            if args.source_only_adj_control:
                print(f"  [J] Source-only adj control (city transfer fairness) | seed={seed}")
                src_ctrl = evaluate_city_transfer_source_adj(
                    model, bay_subsets, adj_metr, device, metr_stats, N,
                )
                seed_res["city_source_adj_control"] = src_ctrl
                print(f"    city_src_MAE(source_adj)="
                      f"{src_ctrl['city_src_MAE_mean']:.4f}±{src_ctrl['city_src_MAE_std']:.4f}")

            # ── [K] GWNet[fixed] city transfer comparison (v4) ───────────
            if gwnet_fixed_prior_target and backbone_name in ("gwnet", "gwnet_cm"):
                print(f"  [K] GWNet[fixed] city transfer control | seed={seed}")
                city_fixed_maes = []
                for sub in bay_subsets:
                    src_loader_f = make_dataloader(sub["X_src"], sub["Y_src"], 64, shuffle=False)
                    # Use METR-LA adjacency instead of PEMS-BAY (same as fixed-adj variant)
                    metrics_f = evaluate(model, src_loader_f, adj_metr, device)
                    city_fixed_maes.append(metrics_f["MAE"])
                seed_res["city_gwnet_fixed_prior"] = {
                    "city_src_MAE_mean": float(np.mean(city_fixed_maes)),
                    "city_src_MAE_std":  float(np.std(city_fixed_maes)),
                    "note": "GWNet evaluated with METR-LA fixed adj on PEMS-BAY data",
                }
                print(f"    GWNet[fixed-on-city]: "
                      f"{np.mean(city_fixed_maes):.4f}±{np.std(city_fixed_maes):.4f}")

            bb_results[key] = seed_res

        all_results[backbone_name] = bb_results

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "shift_eval_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'Backbone':<10} {'Δ_city(src)':>14} {'Δ_sensor(zeromask)':>20} {'Δ_temporal(we-wd)':>18}")
    print("-"*80)
    for bb in args.backbones:
        if bb not in all_results:
            continue
        br = all_results[bb]
        dc = [br[k]["city"]["delta_city_src"] for k in br if "city" in br[k]]
        ds = [br[k]["sensor_zeromask"]["delta_sensor"] for k in br if "sensor_zeromask" in br[k]]
        dt = [br[k]["temporal_wdwe"]["delta_temporal"] for k in br if "temporal_wdwe" in br[k]]
        row = f"{bb:<10}"
        row += f"  {np.mean(dc):>+10.4f}±{np.std(dc):.4f}" if dc else f"  {'N/A':>14}"
        row += f"  {np.mean(ds):>+16.4f}±{np.std(ds):.4f}" if ds else f"  {'N/A':>20}"
        row += f"  {np.mean(dt):>+14.4f}±{np.std(dt):.4f}" if dt else f"  {'N/A':>18}"
        print(row)
    print("="*80)


if __name__ == "__main__":
    main()
