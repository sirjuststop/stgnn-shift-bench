"""stgnn-stress — deployment-shift stress testing for frozen STGNN checkpoints.

Artifact accompanying the NeurIPS 2026 E&D camera-ready paper
"Traffic STGNNs across Sensor, City, and Time Shifts."

Core API:
    from stgnn_stress import recommend_architecture, available_architectures
    rec = recommend_architecture(sensor_churn=True, city_transfer=False, weekend_heavy=True)
    print(rec.architecture, rec.confidence, rec.runner_up)

    from stgnn_stress.shifts import evaluate_checkpoint
    report = evaluate_checkpoint(ckpt_path, shifts=["sensor", "city", "temporal"])

Design goals:
    * Work from the frozen-checkpoint registry shipped with the paper's artifact.
    * 0 external dependencies beyond torch and numpy.
    * Every recommendation carries a margin and an honesty flag
      (models with artifact-driven robustness on an axis are demoted).
"""
from __future__ import annotations

from .recommend import (
    Recommendation,
    available_architectures,
    recommend_architecture,
    DeploymentProfile,
)

__version__ = "0.1.0a1"
__all__ = [
    "Recommendation",
    "DeploymentProfile",
    "available_architectures",
    "recommend_architecture",
]
