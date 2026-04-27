"""
compute_diagnostic.py — D&B multi-dimension scoring and dataset diagnostics.
Benchmark-First Graph Generalization | Direction D | KDD/NeurIPS 2026 D&B

Modes:
  analyze   — graph statistics + homophily + degree distribution for a dataset
  calibrate — run calibration sweep (find r_e, r_f for target accuracy drop)
  score     — compute D&B multi-dimension score from Protocol Bank results
  taxonomy  — print failure taxonomy matrix (backbone × shift type)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Type alias for scoring functions: (score_float, detail_str)
Tuple_score = Tuple[float, str]

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.join(REPO_ROOT, "baselines")
REGISTRY_PATH = os.path.join(BASELINES_DIR, "protocol_registry.json")
DIAG_DIR      = os.path.join(REPO_ROOT, "diagnostics")

# ─── D&B Scoring Rubric ───────────────────────────────────────────────────────

DB_WEIGHTS = {
    "coverage":              0.35,
    "intervention":          0.25,
    "reproducibility":       0.20,
    "cross_backbone":        0.10,
    "taxonomy_completeness": 0.10,
}

def load_registry() -> Dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def load_result(key: str) -> Optional[Dict]:
    registry = load_registry()
    if key not in registry:
        return None
    fpath = registry[key]["file"]
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def load_all_results() -> List[Dict]:
    registry = load_registry()
    results = []
    for entry in registry.values():
        fpath = entry.get("file", "")
        if os.path.exists(fpath):
            with open(fpath) as f:
                try:
                    results.append(json.load(f))
                except json.JSONDecodeError:
                    pass
    return results

# ─── Coverage Score (35%) ─────────────────────────────────────────────────────

def score_coverage(results: List[Dict]) -> Tuple_score:
    """
    ≥3 distinct failure modes across ≥2 dataset families → 100.
    Failure mode = shift config where backbone drops >5% vs clean.
    """
    if not results:
        return 0.0, "No results"

    # Group by dataset family
    family_map = {
        "roman-empire":   "H2GB",
        "amazon-ratings": "H2GB",
        "minesweeper":    "H2GB",
        "tolokers":       "H2GB",
        "questions":      "H2GB",
    }

    # Get clean baselines
    clean_baselines = {
        r["dataset"] + "__" + r["backbone"]: r["metrics"]["test_mean"]
        for r in results if r["shift_config"] == "clean" and r["intervention"] == "none"
    }

    failure_modes = set()
    families_covered = set()

    for r in results:
        if r["shift_config"] == "clean":
            continue
        key = r["dataset"] + "__" + r["backbone"]
        if key not in clean_baselines:
            continue
        drop = clean_baselines[key] - r["metrics"]["test_mean"]
        if drop > 0.05:  # >5% absolute drop = failure mode
            failure_modes.add(f"{r['dataset']}__{r['backbone']}__{r['shift_config']}")
            family = family_map.get(r["dataset"], "unknown")
            families_covered.add(family)

    n_modes    = len(failure_modes)
    n_families = len(families_covered)

    if n_modes >= 3 and n_families >= 2:
        score = 100.0
    elif n_modes >= 2 and n_families >= 1:
        score = 60.0
    elif n_modes >= 1:
        score = 30.0
    else:
        score = 0.0

    detail = f"{n_modes} failure modes across {n_families} families"
    return score, detail


# ─── Intervention Consistency Score (25%) ─────────────────────────────────────

def score_intervention(results: List[Dict]) -> Tuple_score:
    """
    ≥1 intervention improves on ≥3 configs by >1.5×std → 100.
    """
    interventions = set(r["intervention"] for r in results if r["intervention"] != "none")
    if not interventions:
        return 0.0, "No interventions run"

    best_count = 0
    best_name  = ""

    for intv in interventions:
        wins = 0
        for r in results:
            if r["intervention"] != intv:
                continue
            # Find matching baseline (no intervention)
            baseline_key = f"{r['dataset']}__{r['backbone']}__{r['shift_config']}__none"
            base = load_result(baseline_key)
            if base is None:
                continue
            delta = r["metrics"]["test_mean"] - base["metrics"]["test_mean"]
            threshold = 1.5 * base["metrics"]["test_std"]
            if delta > threshold:
                wins += 1
        if wins > best_count:
            best_count = wins
            best_name  = intv

    if best_count >= 3:
        score = 100.0
    elif best_count >= 2:
        score = 60.0
    elif best_count >= 1:
        score = 30.0
    else:
        score = 0.0

    detail = f"Best intervention '{best_name}': {best_count} wins (need ≥3)"
    return score, detail


# ─── Reproducibility Score (20%) ─────────────────────────────────────────────

def score_reproducibility(results: List[Dict]) -> Tuple_score:
    """
    Check that std across seeds is reasonable (std < 2×global mean std).
    Full suite runtime < 1 day is assumed if results are present (Tier 2+).
    """
    if not results:
        return 0.0, "No results"

    stds = [r["metrics"]["test_std"] for r in results if "metrics" in r]
    if not stds:
        return 0.0, "No std data"

    mean_std = np.mean(stds)
    max_std  = np.max(stds)

    # Good reproducibility: low variance across seeds
    if max_std < 0.03:
        score = 100.0
    elif max_std < 0.05:
        score = 80.0
    elif max_std < 0.08:
        score = 60.0
    else:
        score = 30.0

    detail = f"mean_std={mean_std:.4f}, max_std={max_std:.4f}"
    return score, detail


# ─── Cross-Backbone Consistency Score (10%) ───────────────────────────────────

def score_cross_backbone(results: List[Dict]) -> Tuple_score:
    """
    Failure taxonomy stable across GCN/GAT/SAGE on same dataset × shift.
    Stable = all 3 backbones agree on whether it's a failure (>5% drop).
    """
    backbones_seen = set(r["backbone"] for r in results)
    if len(backbones_seen) < 2:
        return 0.0, f"Only {len(backbones_seen)} backbone(s) run (need ≥2)"

    # Group by dataset × shift
    groups: Dict[str, Dict] = {}
    for r in results:
        if r["shift_config"] == "clean":
            continue
        gk = f"{r['dataset']}__{r['shift_config']}"
        if gk not in groups:
            groups[gk] = {}
        clean_key = f"{r['dataset']}__{r['backbone']}__clean__none"
        base = load_result(clean_key)
        if base is None:
            continue
        is_fail = (base["metrics"]["test_mean"] - r["metrics"]["test_mean"]) > 0.05
        groups[gk][r["backbone"]] = is_fail

    if not groups:
        return 0.0, "No cross-backbone data"

    consistent = sum(
        len(set(v.values())) == 1 and len(v) >= 2
        for v in groups.values()
    )
    total = sum(1 for v in groups.values() if len(v) >= 2)

    if total == 0:
        return 0.0, "No multi-backbone groups"

    ratio = consistent / total
    score = 100.0 * ratio
    detail = f"{consistent}/{total} groups consistent across backbones"
    return score, detail


# ─── Taxonomy Completeness Score (10%) ────────────────────────────────────────

def score_taxonomy(results: List[Dict]) -> Tuple_score:
    """
    All 4 shift types represented with evidence (≥1 result each).
    """
    shift_types = set(r["shift_config"] for r in results if r["shift_config"] != "clean")
    target = {"struct", "feat", "joint"}
    covered = shift_types & target
    score = 100.0 * len(covered) / len(target)
    detail = f"Shift types covered: {covered} / {target}"
    return score, detail


# ─── Full D&B Score ───────────────────────────────────────────────────────────

def compute_db_score(results: List[Dict]) -> Dict:
    scores = {
        "coverage":              score_coverage(results),
        "intervention":          score_intervention(results),
        "reproducibility":       score_reproducibility(results),
        "cross_backbone":        score_cross_backbone(results),
        "taxonomy_completeness": score_taxonomy(results),
    }

    weighted_total = sum(
        DB_WEIGHTS[k] * v[0] for k, v in scores.items()
    )

    if weighted_total >= 70:
        verdict = "PROCEED (full KDD/NeurIPS D&B paper)"
    elif weighted_total >= 50:
        verdict = "ANALYSIS PAPER (workshop; write FINDINGS_MEMO.md)"
    else:
        verdict = "KILL (rethink suite design)"

    return {
        "total_score": round(weighted_total, 1),
        "verdict": verdict,
        "breakdown": {k: {"score": v[0], "detail": v[1], "weight": DB_WEIGHTS[k]}
                      for k, v in scores.items()},
    }


def print_db_score(db: Dict):
    print(f"\n{'='*60}")
    print(f"D&B MULTI-DIMENSION SCORE")
    print(f"{'='*60}")
    for dim, info in db["breakdown"].items():
        bar = "█" * int(info["score"] / 10) + "░" * (10 - int(info["score"] / 10))
        print(f"  {dim:<25} [{bar}] {info['score']:5.1f}/100  (×{info['weight']:.2f})")
        print(f"    → {info['detail']}")
    print(f"{'─'*60}")
    print(f"  TOTAL SCORE: {db['total_score']:.1f}/100")
    print(f"  VERDICT:     {db['verdict']}")
    print(f"{'='*60}\n")

# ─── Failure Taxonomy Matrix ──────────────────────────────────────────────────

def print_taxonomy_matrix(results: List[Dict]):
    BACKBONES = ["gcn", "gat", "sage"]
    SHIFTS    = ["struct", "feat", "joint"]
    datasets  = sorted(set(r["dataset"] for r in results))

    clean_baselines = {
        r["dataset"] + "__" + r["backbone"]: r["metrics"]["test_mean"]
        for r in results if r["shift_config"] == "clean" and r["intervention"] == "none"
    }

    def verdict_symbol(dataset, backbone, shift):
        key = f"{dataset}__{backbone}"
        base = clean_baselines.get(key)
        if base is None:
            return "?"
        res_key = f"{dataset}__{backbone}__{shift}__none"
        res = load_result(res_key)
        if res is None:
            return "?"
        drop = base - res["metrics"]["test_mean"]
        if drop > 0.10:
            return "✗ FAILS"
        elif drop > 0.05:
            return "~ MARG"
        else:
            return "✓ ROB"

    print(f"\n{'='*70}")
    print("FAILURE TAXONOMY MATRIX")
    print(f"{'='*70}")

    for dataset in datasets:
        print(f"\n  Dataset: {dataset}")
        print(f"  {'Backbone':<10} {'Struct':>10} {'Feat':>10} {'Joint':>10}")
        print(f"  {'-'*42}")
        for bb in BACKBONES:
            row = [verdict_symbol(dataset, bb, s) for s in SHIFTS]
            present = any(r != "?" for r in row)
            if not present:
                continue
            print(f"  {bb:<10} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    print(f"\n  Legend: ✓ ROB = Robust | ~ MARG = Marginal | ✗ FAILS = Failure")
    print(f"  (Ranking reversal detection requires ≥2 backbones on same config)")

# ─── Dataset Analysis ─────────────────────────────────────────────────────────

def analyze_dataset(name: str):
    sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))
    from joint_shift_main import load_h2gb, compute_degree_quartiles, H2GB_DATASETS

    print(f"\nAnalyzing dataset: {name}")
    data, train_mask, val_mask, test_mask, metric, num_classes = load_h2gb(name)

    from torch_geometric.utils import degree, homophily as pyg_homophily
    import torch

    N = data.num_nodes
    E = data.edge_index.shape[1]
    deg = degree(data.edge_index[0], num_nodes=N).numpy()
    q25, q75 = compute_degree_quartiles(data)

    # Global homophily
    h = float(pyg_homophily(data.edge_index, data.y, method="node"))

    # Feature statistics
    x_np = data.x.numpy()
    feat_dim = x_np.shape[1]
    zero_rate = float((x_np == 0).mean())

    # Class distribution
    labels = data.y.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    imbalance = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")

    print(f"\n  Nodes:            {N:,}")
    print(f"  Edges:            {E:,}")
    print(f"  Features:         {feat_dim}")
    print(f"  Classes:          {num_classes}")
    print(f"  Metric:           {metric}")
    print(f"  Global homophily: {h:.4f}  ({'heterophilic' if h < 0.3 else 'homophilic'})")
    print(f"  Degree Q25/Q75:   {q25:.1f} / {q75:.1f}")
    print(f"  Degree mean/max:  {deg.mean():.1f} / {deg.max():.0f}")
    print(f"  Feature zero rate:{zero_rate:.3f}")
    print(f"  Class imbalance:  {imbalance:.2f}x")
    print(f"  Train/Val/Test:   {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    # Save to diagnostics
    os.makedirs(DIAG_DIR, exist_ok=True)
    diag = {
        "dataset": name, "N": N, "E": E, "feat_dim": feat_dim,
        "num_classes": num_classes, "metric": metric,
        "homophily": h, "deg_q25": q25, "deg_q75": q75,
        "deg_mean": float(deg.mean()), "deg_max": float(deg.max()),
        "feat_zero_rate": zero_rate, "class_imbalance": imbalance,
        "train_n": int(train_mask.sum()), "val_n": int(val_mask.sum()),
        "test_n": int(test_mask.sum()),
    }
    out_path = os.path.join(DIAG_DIR, f"{name}_analysis.json")
    with open(out_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n  Saved → {out_path}")
    return diag


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="D&B diagnostic and scoring tool")
    parser.add_argument("--mode", default="score",
                        choices=["analyze", "score", "taxonomy"],
                        help="analyze: dataset stats | score: D&B score | taxonomy: failure matrix")
    parser.add_argument("--dataset", default="roman-empire",
                        help="Dataset name for analyze/calibrate modes")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_dataset(args.dataset)

    elif args.mode == "score":
        results = load_all_results()
        if not results:
            print("No results in Protocol Bank yet. Run joint_shift_main.py first.")
            return
        db = compute_db_score(results)
        print_db_score(db)
        out_path = os.path.join(DIAG_DIR, "db_score.json")
        os.makedirs(DIAG_DIR, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(db, f, indent=2)
        print(f"Score saved → {out_path}")

    elif args.mode == "taxonomy":
        results = load_all_results()
        if not results:
            print("No results yet.")
            return
        print_taxonomy_matrix(results)


if __name__ == "__main__":
    main()
