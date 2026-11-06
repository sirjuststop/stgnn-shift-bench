"""Measured predictions -> split-wise deltas -> architecture-level statistics."""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from experiments.official_adaptive_common import set_seed
from experiments.supplementary.data import (
    IMPLEMENTATION, Errors, Windows, adjacency, corrupt, failure_mask, read_data, save_json, runtime_metadata,
)
from experiments.supplementary.models import load_model, sequence_lengths


def paired_evaluation(model, windows, failed, kind, mask_seed, protocol, device):
    clean, shifted = Errors(failed), Errors(failed)
    model.eval()
    model.routing_sum, model.routing_count = 0., 0
    with torch.no_grad():
        for batch, (x, target, future_time, ids) in enumerate(DataLoader(
                windows, batch_size=protocol["batch_size"], shuffle=False)):
            x, target, future_time = x.to(device), target.to(device), future_time.to(device)
            changed = corrupt(x, failed, kind, ids, mask_seed, std=windows.std, mean=windows.mean,
                              **protocol["corruption"])
            # GTS graph draws and DGCRN initial-state randomness are paired. No
            # labels are supplied at inference, including decoder inputs.
            rng_seed = protocol["evaluation_seed"] + batch
            set_seed(rng_seed)
            model.capture_routing = model.name == "dgcrn"
            p = model(x, future_time)
            model.capture_routing = False
            set_seed(rng_seed)
            q = model(changed, future_time)
            clean.add(p, target, windows.mean, windows.std)
            shifted.add(q, target, windows.mean, windows.std)
    clean, shifted = clean.result(), shifted.result()
    delta = {s: {m: (None if clean[s][m]["mae"] is None else
                    shifted[s][m]["mae"] - clean[s][m]["mae"]) for m in clean[s]} for s in clean}
    return {"clean": clean, "corrupted": shifted, "delta": delta,
            "failed_nodes": np.flatnonzero(failed).tolist(), "windows": len(windows)}


def correlations(x, y, names):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation requires >=3 complete finite architecture measurements")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "pearson": None, "reason": "constant architecture vector"}
    observed, p_t = stats.pearsonr(x, y)
    permutation = [float(np.corrcoef(x, v)[0, 1]) for v in itertools.permutations(y)]
    loo = {}
    for i, name in enumerate(names):
        a, b = np.delete(x, i), np.delete(y, i)
        loo[name] = float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) and np.ptp(b) else None
    return {"n": len(x), "pearson": float(observed), "p_t_two_sided": float(p_t),
            "p_exact_two_sided": sum(abs(v) >= abs(observed) - 1e-12 for v in permutation) / len(permutation),
            "permutations": len(permutation), "leave_one_out": loo,
            "spearman": float(stats.spearmanr(x, y).statistic)}


def summarize(records):
    grouped = {}
    for r in records:
        for split, metrics in r["delta"].items():
            for metric, value in metrics.items():
                key = (r["dataset"], r["model"], r["corruption"], r["ratio"], split, metric)
                grouped.setdefault(key, {}).setdefault(r["seed"], []).append(value)
    summary = []
    for key, seeds in grouped.items():
        means = [np.mean(v) for v in seeds.values() if all(x is not None for x in v)]
        summary.append(dict(zip(("dataset", "model", "corruption", "ratio", "split", "metric"), key),
                            mean=float(np.mean(means)) if means else None,
                            seed_std=float(np.std(means)) if means else None,
                            seeds=len(means), masks_per_seed={str(k): len(v) for k, v in seeds.items()}))
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_table(table, config, config_path, data_root, output, device, limit=0):
    specification = config[table]
    protocol = config["protocol"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, masses = [], []
    datasets = specification["datasets"]
    for dataset in datasets:
        values, times, data_meta = read_data(data_root, dataset)
        raw_adj, adj, adj_meta = adjacency(data_root, dataset, values.shape[1], device)
        for name in specification["models"]:
            for seed in protocol["seeds"]:
                path = config["checkpoints"][dataset][name].format(seed=seed)
                model, mean, std, ckpt_meta = load_model(path, name, values, raw_adj, adj, device, dataset)
                seq, horizon = sequence_lengths(model)
                windows = Windows(values, times, mean, std, seq_len=seq, horizon=horizon, limit=limit)
                routing_samples = []
                for ratio in specification["ratios"]:
                    for kind in specification["corruptions"]:
                        for mask_seed in protocol["mask_seeds"]:
                            failed = failure_mask(values.shape[1], ratio, mask_seed)
                            measured = paired_evaluation(model, windows, failed, kind, mask_seed, protocol, device)
                            record = dict(dataset=dataset, model=name, seed=seed, mask_seed=mask_seed,
                                          ratio=ratio, corruption=kind, data=data_meta, adjacency=adj_meta,
                                          **ckpt_meta, **measured)
                            records.append(record)
                            if table == "table31":
                                routing_samples.append(model.routing_mass())
                if routing_samples:
                    masses.append({"dataset": dataset, "model": name, "seed": seed,
                                   "max_mass": float(np.mean(routing_samples))})
                # Save a checkpoint of measured progress, explicitly incomplete until all models finish.
                save_json(output / "partial.json", {"implementation": IMPLEMENTATION,
                          "complete": False, "table": table, "records": records, "routing": masses})
                print(f"{table}: {dataset}/{name}/seed{seed} complete", flush=True)
                del model
    result = {"implementation": IMPLEMENTATION, "complete": True, "table": table,
              "config_path": str(config_path), "config": config,
              "window_limit": limit, "runtime": runtime_metadata(device),
              "records": records, "summary": summarize(records)}
    if table == "table31":
        architecture = []
        for dataset in datasets:
            for name in specification["models"]:
                m = [r["max_mass"] for r in masses if r["dataset"] == dataset and r["model"] == name]
                d = [r["delta"]["healthy"]["normalized_mae"] for r in records
                     if r["dataset"] == dataset and r["model"] == name]
                if len(m) != len(protocol["seeds"]) or not d or any(v is None for v in d):
                    raise ValueError(f"Incomplete architecture: {dataset}/{name}")
                architecture.append({"dataset": dataset, "model": name,
                                     "max_mass": float(np.mean(m)), "delta_sensor": float(np.mean(d))})
        result["routing_per_seed"] = masses
        result["architecture_measurements"] = architecture
        result["correlations"] = {}
        for dataset in datasets:
            rows = [r for r in architecture if r["dataset"] == dataset]
            result["correlations"][dataset] = correlations(
                [r["max_mass"] for r in rows], [r["delta_sensor"] for r in rows],
                [r["model"] for r in rows])
    save_json(output / "results.json", result)
    write_csv(output / "summary.csv", result["summary"])
    if table == "table31":
        write_csv(output / "architecture_measurements.csv", result["architecture_measurements"])
    return result
