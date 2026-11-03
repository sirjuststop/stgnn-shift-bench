"""Architecture recommender backed by the frozen decision-tree artifact.

Usage:
    from stgnn_stress import recommend_architecture
    rec = recommend_architecture(sensor_churn=True, city_transfer=True, weekend_heavy=False)
    assert rec.architecture in {"dcrnn", "agcrn", "megacrn", "staeformer", "mtgnn", "gru", "stgcn", "gwnet"}

The recommender loads the artifact JSON shipped alongside this package
(see `data/decision_tree.json`). Recommendations default to the *honest*
ranking which demotes models whose axis fragility is attributable to a
known artifact (e.g. MegaCRN's sensor-dropout U-shape; see Polish v10
Finding 3 for the mechanism).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

DATA_PATH = Path(__file__).resolve().parent / "data" / "decision_tree.json"


@dataclass(frozen=True)
class DeploymentProfile:
    sensor_churn: bool
    city_transfer: bool
    weekend_heavy: bool

    def key(self) -> str:
        return f"s{int(self.sensor_churn)}c{int(self.city_transfer)}t{int(self.weekend_heavy)}"


@dataclass(frozen=True)
class Recommendation:
    architecture: str
    runner_up: str
    margin: float
    confidence: float
    ranked: List[Tuple[str, float]]
    honest: bool
    """True if the ranking used here demotes models with artifact-driven
    near-zero fragility on a flagged axis."""


def _load_artifact():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"decision_tree.json missing at {DATA_PATH}; "
            f"re-install the stgnn-stress package or copy the artifact "
            f"from results/decision_tree/ in the paper repository."
        )
    with open(DATA_PATH) as f:
        return json.load(f)


def available_architectures() -> List[str]:
    art = _load_artifact()
    return sorted(art["normalized_axes_honest"].keys())


def recommend_architecture(
    sensor_churn: bool,
    city_transfer: bool,
    weekend_heavy: bool,
    honest: bool = True,
) -> Recommendation:
    """Recommend an architecture for a deployment profile.

    Args:
        sensor_churn: expect unplanned sensor dropouts.
        city_transfer: expect zero-shot deployment in a new city.
        weekend_heavy: weekend/holiday regimes are load-bearing.
        honest: if True (default), demote models with artifact-driven low
            fragility on an axis. Set False to see the raw ranking.

    Returns:
        Recommendation with architecture name, runner-up, margin, confidence
        (1.0 = clear winner, 0 = tied), and the full ranked list.
    """
    art = _load_artifact()
    profile = DeploymentProfile(sensor_churn, city_transfer, weekend_heavy)
    bucket = "profile_recommendations_honest" if honest else "profile_recommendations_raw"
    entry = art[bucket].get(profile.key())
    if entry is None:
        raise KeyError(f"No recommendation for profile {profile}")
    return Recommendation(
        architecture=entry["winner"],
        runner_up=entry["runner_up"],
        margin=float(entry["margin"]),
        confidence=float(entry["confidence"]),
        ranked=[(m, float(s)) for m, s in entry["ranked"]],
        honest=honest,
    )
