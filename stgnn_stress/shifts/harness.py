"""Shift evaluation harness (scaffold).

Loads a frozen checkpoint and runs the three canonical deployment shifts:
  - sensor_dropout: zero-mask a fraction of nodes at inference time
  - city_transfer: swap the test city's adjacency + sensors
  - temporal_split: stratify by weekday vs weekend regime

This scaffold delegates to the canonical reference implementations under
`experiments/` in the paper repository. When bundled as a pip-installable
artifact, the reference implementations are copied under
`stgnn_stress/_backends/` and imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ShiftReport:
    checkpoint: str
    architecture: str
    source_city: str
    shifts: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"checkpoint={self.checkpoint}", f"architecture={self.architecture}",
                 f"source_city={self.source_city}"]
        for shift, metrics in self.shifts.items():
            m = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            lines.append(f"  [{shift}] {m}")
        return "\n".join(lines)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    shifts: Optional[List[str]] = None,
    target_cities: Optional[List[str]] = None,
) -> ShiftReport:
    """Run the canonical shift suite on a frozen checkpoint.

    Args:
        checkpoint_path: path to a `.pt` file produced by the paper's training
            scripts. Must contain keys `state_dict`, `backbone` (or
            `model_name`), `dataset`.
        shifts: subset of {"sensor", "city", "temporal"}; defaults to all.
        target_cities: target-city keys for the `city` shift. Defaults to
            ["pems-bay"] if the source city is metr-la.

    Returns:
        ShiftReport with normalized-MAE deltas per shift.

    Note (scaffold):
        The current release wires up the recommender but defers the shift
        evaluators to a follow-up. For the exact numbers reported in the
        paper, run `experiments/stgnn_shift_eval_v2.py` in the repo.
    """
    import torch  # local import so `import stgnn_stress` does not require torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    arch = ckpt.get("backbone") or ckpt.get("model_name") or "unknown"
    source = ckpt.get("dataset", "unknown")
    report = ShiftReport(
        checkpoint=str(checkpoint_path),
        architecture=str(arch),
        source_city=str(source),
    )
    if shifts is None:
        shifts = ["sensor", "city", "temporal"]
    for shift in shifts:
        report.shifts[shift] = {
            "status": "not_evaluated",
            "reason": "scaffold — run experiments/stgnn_shift_eval_v2.py for exact numbers",
        }
    return report
