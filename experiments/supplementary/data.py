"""Train-only scaling, lazy chronological windows, and explicit failure masks."""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("metr-la", "pems-bay", "pems04")
IMPLEMENTATION = "supplementary-reconstruction-2026-11-07"


def runtime_metadata(device):
    sources = list((ROOT / "experiments/supplementary").glob("*.py"))
    sources += list((ROOT / "third_party/supplementary_gts").glob("*.py"))
    sources += list((ROOT / "third_party/supplementary_dgcrn").glob("*.py"))
    sources += [ROOT / "experiments" / n for n in (
        "stgnn_backbone_bank.py", "official_adaptive_models.py", "official_adaptive_common.py",
        "routing_sparsity_spectrum.py")]
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "device": str(device),
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(sources)}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_data(root, name):
    folder = Path(root) / name
    stem = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY", "pems04": "pems04"}[name]
    candidates = [folder / (stem + ".csv"), folder / (stem + ".npz"),
                  folder / (name.replace("-", "_") + ".h5"), folder / (stem + ".h5")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No traffic data: expected one of {candidates}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            values = z["data"]
        if values.ndim == 3:
            values = values[..., 0]  # PEMS04 flow, not speed.
        index = pd.date_range("2018-01-01", periods=len(values), freq="5min")
    else:
        frame = (pd.read_csv(path, index_col=0, parse_dates=True) if path.suffix == ".csv"
                 else pd.read_hdf(path))
        values, index = frame.to_numpy(), pd.DatetimeIndex(frame.index)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite (time, nodes) traffic matrix: {path}")
    tod = (index.hour * 60 + index.minute).to_numpy(dtype=np.float32) / (24 * 60)
    return values, tod, {"path": str(path), "sha256": sha256(path),
                         "dataset": name, "unit": "flow" if name == "pems04" else "mph"}


def split_bounds(length):
    return {"train": (0, int(length * .7)), "val": (int(length * .7), int(length * .8)),
            "test": (int(length * .8), length)}


def fit_scaler(values, mode="per_node"):
    train = values[:split_bounds(len(values))["train"][1]]
    axis = None if mode == "global" else 0
    return np.asarray(train.mean(axis=axis), np.float32).reshape(-1), np.asarray(
        train.std(axis=axis) + 1e-8, np.float32).reshape(-1)


def adjacency(root, name, nodes, device="cpu", subset=None):
    folder = Path(root) / name
    candidates = [folder / "adj_mx.pkl", folder / "adj_mx_bay.pkl", folder / "adj_mx.npy"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Adjacency not found: {candidates}")
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    else:
        # Only load locally trusted dataset releases (pickle is executable).
        with open(path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        if isinstance(raw, (tuple, list)) and len(raw) == 3:
            raw = raw[2]
    raw = np.asarray(raw, dtype=np.float32)
    if subset is not None:
        raw = raw[np.ix_(subset, subset)]
    if raw.shape != (nodes, nodes) or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError(f"Invalid adjacency {raw.shape}, expected {(nodes, nodes)}")
    a = torch.as_tensor(raw, device=device)
    d = a.sum(-1).clamp_min(1e-8).rsqrt()
    return a, d[:, None] * a * d[None, :], {"path": str(path), "sha256": sha256(path)}


class Windows(Dataset):
    """Windows never cross chronological split boundaries. Labels remain physical."""
    def __init__(self, values, tod, mean, std, split="test", seq_len=12, horizon=12,
                 limit=0):
        self.values, self.tod = values, tod
        self.mean, self.std = np.asarray(mean), np.asarray(std)
        self.start, stop = split_bounds(len(values))[split]
        self.seq_len, self.horizon = seq_len, horizon
        self.count = stop - self.start - seq_len - horizon + 1
        if self.count < 1:
            raise ValueError(f"{split} split too short for {seq_len}+{horizon} window")
        if limit:
            self.count = min(limit, self.count)

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        t = self.start + int(i)
        end = t + self.seq_len
        x = (self.values[t:end] - self.mean) / self.std
        x_time = np.broadcast_to(self.tod[t:end, None], x.shape)
        y = self.values[end:end + self.horizon]
        y_time = np.broadcast_to(self.tod[end:end + self.horizon, None], y.shape)
        return (torch.from_numpy(np.stack([x, x_time], axis=-1).astype(np.float32)),
                torch.from_numpy(y.copy()), torch.from_numpy(y_time.copy()), int(i))


def failure_mask(nodes, ratio, seed):
    if not 0 < ratio < 1:
        raise ValueError("Affected fraction must be strictly between zero and one")
    count = int(nodes * ratio)
    if not 0 < count < nodes:
        raise ValueError("Affected fraction leaves an empty failed/healthy split")
    result = np.zeros(nodes, dtype=bool)
    result[np.random.RandomState(seed).choice(nodes, count, replace=False)] = True
    return result


def corrupt(x, failed, kind, window_ids, seed, *, std, mean, fill_space="normalized",
            noise_std=.1, bias_offset=1., intermittent_probability=.5):
    """Alter only traffic values at explicitly selected sensors; preserve covariates.

    Noise and bias are in z-score units. Stale hold repeats the first input value.
    Random fields are keyed by window, so batching does not change corruptions.
    """
    if noise_std < 0 or not 0 <= intermittent_probability <= 1:
        raise ValueError("Invalid corruption magnitude or intermittent probability")
    out = x.clone()
    mask = torch.as_tensor(failed, dtype=torch.bool, device=x.device)
    zero = torch.zeros(x.shape[2], device=x.device, dtype=x.dtype)
    if fill_space == "physical":
        zero = -torch.as_tensor(mean, device=x.device) / torch.as_tensor(std, device=x.device)
        zero = zero.expand(x.shape[2])
    elif fill_space != "normalized":
        raise ValueError(fill_space)
    if kind == "zero":
        out[:, :, mask, 0] = zero[mask]
    elif kind == "stale":
        out[:, :, mask, 0] = x[:, :1, mask, 0]
    elif kind == "bias":
        out[:, :, mask, 0] += bias_offset
    elif kind in ("noise", "intermittent"):
        for b, i in enumerate(window_ids):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(i)]))
            shape = (x.shape[1], int(mask.sum()))
            if kind == "noise":
                values = rng.normal(0., noise_std, size=shape).astype(np.float32)
                out[b, :, mask, 0] += torch.as_tensor(values, device=x.device)
            else:
                events = torch.as_tensor(rng.random(shape) < intermittent_probability, device=x.device)
                out[b, :, mask, 0] = torch.where(events, zero[mask], x[b, :, mask, 0])
    else:
        raise ValueError(f"Unknown corruption: {kind}")
    return out


class Errors:
    """Accumulate sums/counts, avoiding batch-mean weighting and implicit masks."""
    def __init__(self, failed):
        self.failed = np.asarray(failed, bool)
        self.totals = {s: {m: [0., 0] for m in ("normalized_mae", "masked_physical_mae")}
                       for s in ("healthy", "failed", "whole")}

    def add(self, prediction_norm, target_raw, mean, std):
        mean = torch.as_tensor(mean, device=prediction_norm.device)
        std = torch.as_tensor(std, device=prediction_norm.device)
        pred_raw = prediction_norm * std + mean
        errors = {"normalized_mae": (prediction_norm - (target_raw - mean) / std).abs(),
                  "masked_physical_mae": (pred_raw - target_raw).abs()}
        for split in self.totals:
            nodes = (np.ones_like(self.failed) if split == "whole" else
                     self.failed if split == "failed" else ~self.failed)
            nodes = torch.as_tensor(nodes, device=target_raw.device)
            for metric, error in errors.items():
                selected = error[..., nodes]
                if metric == "masked_physical_mae":
                    selected = selected[target_raw[..., nodes] > 0]
                self.totals[split][metric][0] += selected.double().sum().item()
                self.totals[split][metric][1] += selected.numel()

    def result(self):
        return {s: {m: {"mae": total / count if count else None, "count": count}
                    for m, (total, count) in metrics.items()} for s, metrics in self.totals.items()}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
