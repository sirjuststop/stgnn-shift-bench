"""Decision-tree diagnostic artifact (Polish v10 item #9).

Given a deployment profile (sensor-churn expected? city-transfer? weekend-heavy?),
recommend the architecture from our 8-model bank that is predicted to degrade
least under the dominant shift axis.

Inputs:
  results/official_adaptive_relation/merged_comparison_table.csv
    — per-model Δ_sensor, Δ_city (normalized), Δ_temporal (normalized).

Outputs:
  results/decision_tree/decision_tree.json
    — the 3-rule tree, per-profile best model, and confidence margins.
  results/decision_tree/decision_tree_summary.md
    — human-readable rules for the paper / artifact.

0 GPU-h. Stdlib only (csv, json, statistics, math).

Logic: for each deployment profile (triple of booleans over 3 axes), we compute
a weighted-risk score per model. Weights reflect which shifts dominate in that
profile. We recommend the argmin-risk model and report the margin over the
runner-up as a confidence proxy.

We derive the 3-rule tree by (i) enumerating all 8 profiles, (ii) asking which
single axis maximally separates recommendations, and (iii) recursing.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
MERGED = REPO / "results" / "official_adaptive_relation" / "merged_comparison_table.csv"
OUT_DIR = REPO / "results" / "decision_tree"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows() -> List[Dict[str, float]]:
    with open(MERGED) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    parsed = []
    for row in rows:
        rec = {"model": row["model"], "family": row.get("family", "")}
        for k, v in row.items():
            if k in ("model", "family"):
                continue
            try:
                rec[k] = float(v) if v else float("nan")
            except ValueError:
                rec[k] = float("nan")
        parsed.append(rec)
    return parsed


AXIS_COLUMNS = {
    "sensor": "dropout_delta_mean",
    "city": "city_src_mae_mean",
    "temporal": "temporal_delta_mean",
}

# Flagged (model, axis) pairs whose apparent robustness is attributable to a
# known artifact rather than a genuine deployment advantage. Derived from
# Polish v10 Finding 3 (MegaCRN memory-bank ablation: memzero deepens the
# U-shape; cross-node flow is strictly adjacency-mediated; σ ≈ mean across
# seeds → the negative Δ_sensor is a seed-fragile smoothing artifact, not
# regularization).
FLAGGED_ROBUSTNESS = {("megacrn", "sensor")}


def normalize_axes(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Return per-model vectors in [0,1] on (sensor, city, temporal) axes.

    Higher = more fragile on that axis. Negative deltas are clipped to 0 at
    the reporting layer; the flagged-robustness set above overrides those
    values where the zero/negative reading is a known artifact.
    """
    raw = {axis: [max(r[col], 0.0) for r in rows] for axis, col in AXIS_COLUMNS.items()}
    hi = {axis: max(v) for axis, v in raw.items()}
    lo = {axis: min(v) for axis, v in raw.items()}
    out = {}
    for r in rows:
        out[r["model"]] = {
            axis: (max(r[col], 0.0) - lo[axis]) / (hi[axis] - lo[axis] + 1e-12)
            for axis, col in AXIS_COLUMNS.items()
        }
    return out


def apply_flagged_robustness(norm_axes):
    """Return a second axis vector where flagged (model, axis) pairs are set
    to the median across other models on that axis.

    This is the "honest" ranking used for safety-aware recommendation: a
    model flagged for artifact-driven low fragility on an axis is treated as
    average on that axis rather than best.
    """
    medians = {}
    for axis in AXIS_COLUMNS:
        vals = sorted(ax[axis] for ax in norm_axes.values())
        medians[axis] = vals[len(vals) // 2]
    honest = {m: dict(ax) for m, ax in norm_axes.items()}
    for model, axis in FLAGGED_ROBUSTNESS:
        if model in honest:
            honest[model][axis] = medians[axis]
    return honest


def profile_weights(sensor: bool, city: bool, temporal: bool) -> Dict[str, float]:
    """Weight each axis by whether the deployment expects that shift.

    If a profile has no "yes", weight all axes equally (in-distribution profile).
    """
    flags = {"sensor": sensor, "city": city, "temporal": temporal}
    active = [a for a, v in flags.items() if v]
    if not active:
        return {a: 1.0 / 3 for a in flags}
    return {a: (1.0 / len(active) if flags[a] else 0.0) for a in flags}


def rank_models(norm_axes: Dict[str, Dict[str, float]], weights: Dict[str, float]):
    scores = []
    for model, ax in norm_axes.items():
        risk = sum(weights[a] * ax[a] for a in ax)
        scores.append((model, risk))
    scores.sort(key=lambda x: x[1])
    return scores


def enumerate_recommendations(norm_axes) -> Dict[Tuple[bool, bool, bool], Dict]:
    out = {}
    for s in (False, True):
        for c in (False, True):
            for t in (False, True):
                w = profile_weights(s, c, t)
                ranked = rank_models(norm_axes, w)
                winner, winner_risk = ranked[0]
                runner, runner_risk = ranked[1]
                out[(s, c, t)] = {
                    "weights": w,
                    "ranked": ranked,
                    "winner": winner,
                    "runner_up": runner,
                    "margin": runner_risk - winner_risk,
                    "winner_risk": winner_risk,
                    "confidence": min(1.0, (runner_risk - winner_risk) / (runner_risk + 1e-12)),
                }
    return out


def tree_split(recs: Dict[Tuple[bool, bool, bool], Dict]) -> Dict:
    """Greedy tree-like rule summary.

    Axis order chosen greedily: split first on the axis that maximally reduces
    recommendation disagreement.
    """
    axes = ["sensor", "city", "temporal"]

    def winners(profile_subset):
        return {p: recs[p]["winner"] for p in profile_subset}

    all_profiles = list(recs.keys())

    def disagreement(subset):
        ws = set(recs[p]["winner"] for p in subset)
        return len(ws)

    def best_axis(subset):
        best, best_score = axes[0], -math.inf
        for idx, axis in enumerate(axes):
            left = [p for p in subset if not p[idx]]
            right = [p for p in subset if p[idx]]
            score = disagreement(subset) - (
                (disagreement(left) if left else 0) + (disagreement(right) if right else 0)
            )
            if score > best_score:
                best_score, best = score, axis
        return best, best_score

    root_axis, root_gain = best_axis(all_profiles)
    root_idx = axes.index(root_axis)

    def summarize(subset: List[Tuple[bool, bool, bool]]):
        return {
            "winners": sorted(set(recs[p]["winner"] for p in subset)),
            "profiles": [
                {
                    "sensor_churn": p[0],
                    "city_transfer": p[1],
                    "weekend_heavy": p[2],
                    "recommend": recs[p]["winner"],
                    "runner_up": recs[p]["runner_up"],
                    "margin": round(recs[p]["margin"], 4),
                    "confidence": round(recs[p]["confidence"], 3),
                }
                for p in subset
            ],
        }

    left = [p for p in all_profiles if not p[root_idx]]
    right = [p for p in all_profiles if p[root_idx]]
    return {
        "root_axis": root_axis,
        "split_gain": root_gain,
        f"{root_axis}=no": summarize(left),
        f"{root_axis}=yes": summarize(right),
    }


def main():
    rows = load_rows()
    norm = normalize_axes(rows)
    honest = apply_flagged_robustness(norm)
    recs = enumerate_recommendations(norm)
    recs_honest = enumerate_recommendations(honest)
    tree = tree_split(recs_honest)

    artifact = {
        "source_data": str(MERGED.relative_to(REPO)),
        "num_models": len(rows),
        "flagged_robustness": sorted([list(x) for x in FLAGGED_ROBUSTNESS]),
        "normalized_axes_raw": norm,
        "normalized_axes_honest": honest,
        "profile_recommendations_raw": {
            f"s{int(p[0])}c{int(p[1])}t{int(p[2])}": {
                "sensor_churn": p[0],
                "city_transfer": p[1],
                "weekend_heavy": p[2],
                "winner": r["winner"],
                "runner_up": r["runner_up"],
                "margin": round(r["margin"], 6),
                "confidence": round(r["confidence"], 3),
                "ranked": [[m, round(s, 4)] for m, s in r["ranked"]],
            }
            for p, r in recs.items()
        },
        "profile_recommendations_honest": {
            f"s{int(p[0])}c{int(p[1])}t{int(p[2])}": {
                "sensor_churn": p[0],
                "city_transfer": p[1],
                "weekend_heavy": p[2],
                "winner": r["winner"],
                "runner_up": r["runner_up"],
                "margin": round(r["margin"], 6),
                "confidence": round(r["confidence"], 3),
                "ranked": [[m, round(s, 4)] for m, s in r["ranked"]],
            }
            for p, r in recs_honest.items()
        },
        "tree": tree,
    }
    out_json = OUT_DIR / "decision_tree.json"
    with open(out_json, "w") as f:
        json.dump(artifact, f, indent=2)

    lines = []
    lines.append("# Shift-Aware Architecture Selection Tree (Polish v10)\n")
    lines.append(
        "Source: `results/official_adaptive_relation/merged_comparison_table.csv`\n"
    )
    lines.append(f"Models evaluated: {len(rows)}\n\n")
    lines.append(
        "Inputs: boolean deployment profile over three axes — "
        "`sensor_churn` (unplanned sensor dropouts expected), "
        "`city_transfer` (zero-shot deployment in a new city), "
        "`weekend_heavy` (weekend/holiday traffic is load-bearing). "
        "Output: architecture recommended from the 8-model bank, "
        "and a confidence margin over the runner-up. Scores are normalized "
        "per-axis so a score of 0 means least fragile and 1 means most fragile.\n\n"
    )
    lines.append("## Flagged robustness caveat\n")
    for model, axis in sorted(FLAGGED_ROBUSTNESS):
        lines.append(
            f"- **{model}** on `{axis}` axis: apparent near-zero / negative "
            "fragility is a known artifact (adjacency-mediated smoothing, "
            "seed-fragile). The *honest* table below replaces this reading "
            "with the median across models.\n"
        )
    lines.append("\n## Honest 2x2x2 profile table (used for the tree)\n")
    lines.append("| sensor | city | weekend | recommend | runner-up | margin | confidence |\n")
    lines.append("|--------|------|---------|-----------|-----------|--------|------------|\n")
    for p, r in sorted(recs_honest.items()):
        lines.append(
            f"| {str(p[0]):5} | {str(p[1]):5} | {str(p[2]):7} | "
            f"**{r['winner']}** | {r['runner_up']} | "
            f"{r['margin']:.4f} | {r['confidence']:.3f} |\n"
        )
    lines.append("\n## Raw 2x2x2 profile table (unflagged, for reference)\n")
    lines.append("| sensor | city | weekend | recommend | runner-up | margin | confidence |\n")
    lines.append("|--------|------|---------|-----------|-----------|--------|------------|\n")
    for p, r in sorted(recs.items()):
        lines.append(
            f"| {str(p[0]):5} | {str(p[1]):5} | {str(p[2]):7} | "
            f"**{r['winner']}** | {r['runner_up']} | "
            f"{r['margin']:.4f} | {r['confidence']:.3f} |\n"
        )
    lines.append(f"\n## Greedy split\n")
    lines.append(f"Root axis: `{tree['root_axis']}` (disagreement reduction {tree['split_gain']}).\n\n")
    for branch in (f"{tree['root_axis']}=no", f"{tree['root_axis']}=yes"):
        lines.append(f"### {branch}\n")
        lines.append(
            f"Recommendations in this branch: {', '.join(tree[branch]['winners'])}\n\n"
        )
        for entry in tree[branch]["profiles"]:
            lines.append(
                f"- sensor={entry['sensor_churn']}, city={entry['city_transfer']}, "
                f"weekend={entry['weekend_heavy']} → **{entry['recommend']}** "
                f"(margin {entry['margin']}, confidence {entry['confidence']})\n"
            )
        lines.append("\n")
    lines.append("## Per-model normalized fragility vectors\n")
    lines.append("| model | sensor | city | temporal |\n")
    lines.append("|-------|--------|------|----------|\n")
    for model, ax in norm.items():
        lines.append(
            f"| {model} | {ax['sensor']:.3f} | {ax['city']:.3f} | {ax['temporal']:.3f} |\n"
        )
    out_md = OUT_DIR / "decision_tree_summary.md"
    with open(out_md, "w") as f:
        f.writelines(lines)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
