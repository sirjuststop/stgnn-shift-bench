"""
joint_shift_main.py — Unified training framework for joint-shift GNN benchmark.
Benchmark-First Graph Generalization | Direction D | KDD/NeurIPS 2026 D&B

Unified training config (FROZEN per CLAUDE.md constitution):
  epochs=300, lr=1e-3, wd=1e-4, dropout=0.5, layers=2, hidden=256, CosineAnnealingLR

Shift configs:
  clean   — no shift (baseline)
  struct  — structural shift only (edge rewiring at rate r_e)
  feat    — feature shift only (masking at rate r_f)
  joint   — both shifts simultaneously

Interventions (each < 200 LoC):
  none            — no intervention (baseline)
  coupled_dropout — Iso-Drop: scale dropout by estimated shift severity
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import degree, homophily

# ─── Constants ───────────────────────────────────────────────────────────────

UNIFIED_CONFIG = dict(
    epochs=300,
    lr=1e-3,
    weight_decay=1e-4,
    dropout=0.5,
    num_layers=2,
    hidden_dim=256,
    scheduler="CosineAnnealingLR",
)

H2GB_DATASETS = {
    "roman-empire":    ("Roman-empire",    "acc"),
    "amazon-ratings":  ("Amazon-ratings",  "acc"),
    "minesweeper":     ("Minesweeper",     "roc_auc"),
    "tolokers":        ("Tolokers",        "roc_auc"),
    "questions":       ("Questions",       "roc_auc"),
}

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
BASELINES_ROOT = os.path.join(os.path.dirname(__file__), "..", "baselines")
REGISTRY_PATH = os.path.join(BASELINES_ROOT, "protocol_registry.json")

# ─── Dataset Loading ──────────────────────────────────────────────────────────

def load_h2gb(name: str, split_idx: int = 0, device: torch.device = torch.device("cpu")):
    pyg_name, metric = H2GB_DATASETS[name]
    ds = HeterophilousGraphDataset(root=DATA_ROOT, name=pyg_name)
    data = ds[0].to(device)
    train_mask = data.train_mask[:, split_idx]
    val_mask   = data.val_mask[:, split_idx]
    test_mask  = data.test_mask[:, split_idx]
    num_classes = int(data.y.max().item()) + 1
    return data, train_mask, val_mask, test_mask, metric, num_classes

# ─── Shift Injection ──────────────────────────────────────────────────────────

def inject_structural_shift(edge_index: torch.Tensor, num_nodes: int,
                             rate: float, seed: int = 0) -> torch.Tensor:
    """Rewire `rate` fraction of edges to random targets (increases heterophily)."""
    if rate == 0.0:
        return edge_index
    rng = np.random.RandomState(seed)
    edges = edge_index.cpu().t().numpy().copy()  # [E, 2]
    n_rewire = int(len(edges) * rate)
    if n_rewire == 0:
        return edge_index
    idx = rng.choice(len(edges), n_rewire, replace=False)
    edges[idx, 1] = rng.randint(0, num_nodes, n_rewire)
    # Remove self-loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]
    return torch.tensor(edges.T, dtype=torch.long, device=edge_index.device)


def inject_feature_shift(x: torch.Tensor, rate: float, seed: int = 0) -> torch.Tensor:
    """Randomly zero out `rate` fraction of features (multimodal corruption proxy)."""
    if rate == 0.0:
        return x
    rng = np.random.RandomState(seed)
    mask = torch.tensor(rng.random(x.shape) < rate, dtype=torch.bool, device=x.device)
    x_shifted = x.clone()
    x_shifted[mask] = 0.0
    return x_shifted


def apply_shift(data, shift_config: str, r_e: float, r_f: float,
                seed: int = 0):
    """Return (edge_index, x) with requested shift applied."""
    edge_index = data.edge_index
    x = data.x
    if shift_config in ("struct", "joint"):
        edge_index = inject_structural_shift(edge_index, data.num_nodes, r_e, seed)
    if shift_config in ("feat", "joint"):
        x = inject_feature_shift(x, r_f, seed)
    return edge_index, x

# ─── Iso-Drop Severity Calibration (< 200 LoC intervention) ──────────────────

def shift_severity(r_e: float, r_f: float) -> float:
    """Normalised joint-shift severity in [0, 1]."""
    return float(np.sqrt(r_e**2 + r_f**2) / np.sqrt(2))


def coupled_dropout_rate(base_dropout: float, r_e: float, r_f: float,
                          alpha: float = 1.0) -> float:
    """Iso-Drop: scale dropout by shift severity."""
    s = shift_severity(r_e, r_f)
    return min(base_dropout * (1.0 + alpha * s), 0.9)

# ─── GNN Backbones ───────────────────────────────────────────────────────────

class GCNBackbone(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GATBackbone(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        heads = 4
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class SAGEBackbone(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


BACKBONES = {"gcn": GCNBackbone, "gat": GATBackbone, "sage": SAGEBackbone}

# ─── Local Failure Atlas ──────────────────────────────────────────────────────

def compute_atlas(data, logits: torch.Tensor, mask: torch.Tensor,
                  deg_q25: float, deg_q75: float) -> Dict:
    """
    8-cell atlas: Degree × LocalHomophily × FeatureQuality → error rate.
    Pre-registered axes (per CLAUDE.md):
      Degree:          Low (≤Q1) vs High (≥Q4)
      LocalHomophily:  Het (<0.3) vs Hom (≥0.3)
      FeatureQuality:  Corrupted (>30% zeros) vs Clean
    """
    device = data.edge_index.device
    N = data.num_nodes

    # Degree bucket
    deg = degree(data.edge_index[0], num_nodes=N).cpu().numpy()
    deg_bucket = np.where(deg <= deg_q25, "Low", np.where(deg >= deg_q75, "High", "Mid"))

    # Local homophily per node
    y = data.y.cpu().numpy()
    src, dst = data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy()
    local_hom = np.zeros(N)
    counts = np.zeros(N)
    same = (y[src] == y[dst]).astype(float)
    np.add.at(local_hom, src, same)
    np.add.at(counts, src, 1)
    counts = np.maximum(counts, 1)
    local_hom /= counts
    hom_bucket = np.where(local_hom < 0.3, "Het", "Hom")

    # Feature quality
    x_np = data.x.cpu().numpy()
    zero_rate = (x_np == 0).mean(axis=1)
    feat_bucket = np.where(zero_rate > 0.3, "Corrupted", "Clean")

    # Per-node correctness
    preds = logits.argmax(dim=1).cpu().numpy()
    correct = (preds == y)
    mask_np = mask.cpu().numpy()

    atlas = {}
    for i in range(N):
        if not mask_np[i]:
            continue
        if deg_bucket[i] == "Mid":
            continue  # only track Q1/Q4 extremes
        cell = f"{deg_bucket[i]}_{hom_bucket[i]}_{feat_bucket[i]}"
        if cell not in atlas:
            atlas[cell] = {"correct": 0, "total": 0}
        atlas[cell]["total"] += 1
        atlas[cell]["correct"] += int(correct[i])

    return {k: {"error_rate": 1 - v["correct"] / max(v["total"], 1),
                "n": v["total"]} for k, v in atlas.items()}

# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_run(data, train_mask, val_mask, test_mask,
                  backbone_cls, num_classes: int, metric: str,
                  shift_config: str, r_e: float, r_f: float,
                  seed: int, intervention: str,
                  cfg: dict, device: torch.device,
                  deg_q25: float, deg_q75: float) -> Dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Apply shift (deterministic per seed)
    edge_index, x = apply_shift(data, shift_config, r_e, r_f, seed=seed + 1000)

    # Compute effective dropout
    dp = cfg["dropout"]
    if intervention == "coupled_dropout":
        dp = coupled_dropout_rate(cfg["dropout"], r_e, r_f)

    model = backbone_cls(
        in_dim=x.shape[1],
        hidden=cfg["hidden_dim"],
        out_dim=num_classes,
        dropout=dp,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    best_val, best_test, best_logits = -1.0, -1.0, None

    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == cfg["epochs"] - 1:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
            val_score  = compute_metric(logits, data.y, val_mask,  metric)
            test_score = compute_metric(logits, data.y, test_mask, metric)
            if val_score > best_val:
                best_val = val_score
                best_test = test_score
                best_logits = logits.detach()

    # Atlas on test set with best logits
    atlas = {}
    if best_logits is not None:
        atlas = compute_atlas(data, best_logits, test_mask, deg_q25, deg_q75)

    return {"test": best_test, "val": best_val, "atlas": atlas,
            "effective_dropout": dp}


def compute_metric(logits: torch.Tensor, y: torch.Tensor,
                   mask: torch.Tensor, metric: str) -> float:
    if metric == "acc":
        pred = logits[mask].argmax(dim=1)
        return float((pred == y[mask]).float().mean())
    else:  # roc_auc
        probs = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
        labels = y[mask].cpu().numpy()
        try:
            return float(roc_auc_score(labels, probs))
        except Exception:
            return 0.5

# ─── Protocol Bank ────────────────────────────────────────────────────────────

def load_registry() -> Dict:
    os.makedirs(BASELINES_ROOT, exist_ok=True)
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def registry_key(dataset, backbone, shift_config, intervention="none"):
    return f"{dataset}__{backbone}__{shift_config}__{intervention}"


def save_to_protocol_bank(dataset, backbone, shift_config, seeds,
                           per_seed_results, r_e, r_f, intervention, cfg):
    registry = load_registry()
    key = registry_key(dataset, backbone, shift_config, intervention)

    scores = [r["test"] for r in per_seed_results]
    result = {
        "key": key,
        "dataset": dataset,
        "backbone": backbone,
        "shift_config": shift_config,
        "intervention": intervention,
        "seeds": seeds,
        "metrics": {
            "test_mean": float(np.mean(scores)),
            "test_std":  float(np.std(scores)),
            "per_seed":  scores,
        },
        "shift_params": {"r_e": r_e, "r_f": r_f},
        "atlas_per_seed": [r["atlas"] for r in per_seed_results],
        "config": cfg,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(BASELINES_ROOT, exist_ok=True)
    fpath = os.path.join(BASELINES_ROOT, f"{key}.json")
    with open(fpath, "w") as f:
        json.dump(result, f, indent=2)

    registry[key] = {
        "key": key, "file": fpath,
        "test_mean": result["metrics"]["test_mean"],
        "test_std":  result["metrics"]["test_std"],
        "status": "frozen",
    }
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    return result

# ─── Calibration ──────────────────────────────────────────────────────────────

def calibrate_rate(data, train_mask, val_mask, test_mask,
                   backbone_cls, num_classes, metric,
                   shift_type: str, target_drop: float,
                   base_acc: float, cfg: dict, device: torch.device,
                   deg_q25: float, deg_q75: float,
                   seed: int = 0, tol: float = 0.005) -> float:
    """Binary search for shift rate that produces `target_drop` accuracy drop."""
    lo, hi = 0.0, 0.8
    for _ in range(15):
        mid = (lo + hi) / 2
        r_e = mid if shift_type in ("struct", "joint") else 0.0
        r_f = mid if shift_type in ("feat",  "joint") else 0.0
        res = train_one_run(data, train_mask, val_mask, test_mask,
                            backbone_cls, num_classes, metric,
                            shift_type, r_e, r_f, seed, "none",
                            cfg, device, deg_q25, deg_q75)
        drop = base_acc - res["test"]
        if abs(drop - target_drop) < tol:
            return mid
        if drop < target_drop:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_degree_quartiles(data) -> Tuple[float, float]:
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
    return float(np.percentile(deg, 25)), float(np.percentile(deg, 75))

# ─── Main Orchestration ───────────────────────────────────────────────────────

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_name = args.dataset
    if dataset_name not in H2GB_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(H2GB_DATASETS)}")

    data, train_mask, val_mask, test_mask, metric, num_classes = load_h2gb(
        dataset_name, split_idx=args.split_idx, device=device
    )
    print(f"Dataset: {dataset_name} | N={data.num_nodes} | E={data.edge_index.shape[1]} | "
          f"Classes={num_classes} | Metric={metric}")

    deg_q25, deg_q75 = compute_degree_quartiles(data)

    # ── Calibration step ──────────────────────────────────────────────────────
    r_e, r_f = args.r_e, args.r_f

    if args.r_e == "AUTO" or args.r_f == "AUTO":
        backbone_cls = BACKBONES[args.backbones[0]]
        print("\n[Calibration] Running clean baseline for calibration...")
        base_res = train_one_run(data, train_mask, val_mask, test_mask,
                                 backbone_cls, num_classes, metric,
                                 "clean", 0.0, 0.0, 0, "none",
                                 UNIFIED_CONFIG, device, deg_q25, deg_q75)
        base_acc = base_res["test"]
        print(f"[Calibration] Clean {metric}: {base_acc:.4f}")

        if args.r_e == "AUTO":
            print(f"[Calibration] Searching r_e for -{args.calibrate_target:.0%} drop...")
            r_e = calibrate_rate(data, train_mask, val_mask, test_mask,
                                 backbone_cls, num_classes, metric,
                                 "struct", args.calibrate_target, base_acc,
                                 UNIFIED_CONFIG, device, deg_q25, deg_q75)
            print(f"[Calibration] r_e = {r_e:.4f}")

        if args.r_f == "AUTO":
            print(f"[Calibration] Searching r_f for -{args.calibrate_target:.0%} drop...")
            r_f = calibrate_rate(data, train_mask, val_mask, test_mask,
                                 backbone_cls, num_classes, metric,
                                 "feat", args.calibrate_target, base_acc,
                                 UNIFIED_CONFIG, device, deg_q25, deg_q75)
            print(f"[Calibration] r_f = {r_f:.4f}")

        # Save calibration to diagnostics
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "diagnostics"), exist_ok=True)
        calib_path = os.path.join(os.path.dirname(__file__), "..", "diagnostics",
                                  f"{dataset_name}_calibration.json")
        with open(calib_path, "w") as f:
            json.dump({"dataset": dataset_name, "base_acc": base_acc,
                       "r_e": r_e, "r_f": r_f, "target_drop": args.calibrate_target,
                       "timestamp": datetime.now().isoformat()}, f, indent=2)
        print(f"[Calibration] Saved → {calib_path}")
    else:
        r_e = float(args.r_e) if args.r_e else 0.0
        r_f = float(args.r_f) if args.r_f else 0.0

    # ── Main experiment loop ──────────────────────────────────────────────────
    shift_configs  = args.shift_configs
    backbones      = args.backbones
    seeds          = args.seeds
    interventions  = args.interventions

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []

    for backbone_name in backbones:
        backbone_cls = BACKBONES[backbone_name]
        for shift_config in shift_configs:
            for intervention in interventions:
                key = registry_key(dataset_name, backbone_name, shift_config, intervention)
                registry = load_registry()
                if key in registry and not args.force:
                    print(f"[SKIP] {key} already in Protocol Bank (use --force to override)")
                    summary.append({"key": key, **registry[key]})
                    continue

                print(f"\n{'='*60}")
                print(f"Running: {key}")
                print(f"  shift={shift_config} | r_e={r_e:.4f} | r_f={r_f:.4f} | "
                      f"seeds={seeds} | intervention={intervention}")
                print(f"{'='*60}")

                per_seed = []
                for seed in seeds:
                    t0 = time.time()
                    res = train_one_run(
                        data, train_mask, val_mask, test_mask,
                        backbone_cls, num_classes, metric,
                        shift_config, r_e, r_f, seed, intervention,
                        UNIFIED_CONFIG, device, deg_q25, deg_q75,
                    )
                    elapsed = time.time() - t0
                    per_seed.append(res)
                    print(f"  seed={seed} | {metric}={res['test']:.4f} | "
                          f"dp_eff={res['effective_dropout']:.3f} | {elapsed:.1f}s")

                result = save_to_protocol_bank(
                    dataset_name, backbone_name, shift_config, seeds,
                    per_seed, r_e, r_f, intervention, UNIFIED_CONFIG
                )
                print(f"  → {metric}_mean={result['metrics']['test_mean']:.4f} "
                      f"± {result['metrics']['test_std']:.4f}")
                summary.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset_name}")
    print(f"{'='*60}")
    print(f"{'Key':<45} {'Mean':>8} {'Std':>8}")
    print("-" * 65)
    for r in summary:
        m = r.get("metrics", r)
        print(f"{r['key']:<45} {m.get('test_mean', m.get('test_mean', '?')):>8.4f} "
              f"{m.get('test_std', '?'):>8.4f}")

    # Save summary
    summary_path = os.path.join(args.out_dir, f"{dataset_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved → {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Joint-shift GNN benchmark framework")
    parser.add_argument("--dataset",   default="roman-empire",
                        choices=list(H2GB_DATASETS))
    parser.add_argument("--backbones", nargs="+", default=["gcn"],
                        choices=["gcn", "gat", "sage"])
    parser.add_argument("--seeds",     nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--shift_configs", nargs="+",
                        default=["clean", "struct", "feat", "joint"],
                        choices=["clean", "struct", "feat", "joint"])
    parser.add_argument("--interventions", nargs="+", default=["none"],
                        choices=["none", "coupled_dropout"])
    parser.add_argument("--r_e",  default="AUTO",
                        help="Edge rewiring rate or AUTO for calibration")
    parser.add_argument("--r_f",  default="AUTO",
                        help="Feature masking rate or AUTO for calibration")
    parser.add_argument("--calibrate_target", type=float, default=0.05,
                        help="Target accuracy drop for calibration (default: 0.05 = 5%%)")
    parser.add_argument("--split_idx", type=int, default=0)
    parser.add_argument("--out_dir",  default=os.path.join(RESULTS_ROOT, "tier0"))
    parser.add_argument("--force",    action="store_true",
                        help="Re-run even if key exists in Protocol Bank")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
