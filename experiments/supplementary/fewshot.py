"""Table 21: reset each source checkpoint for each target-data budget."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.official_adaptive_common import set_seed
from experiments.supplementary.data import (
    IMPLEMENTATION, Errors, Windows, adjacency, fit_scaler, read_data, save_json, runtime_metadata,
)
from experiments.supplementary.models import load_model, sequence_lengths
from experiments.supplementary.evaluate import write_csv


def adapt(model, training, selected, *, steps, batch_size, lr, seed, device):
    """Exactly `steps` updates; sample only the supplied target-training windows."""
    if not selected:
        return []
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(Subset(training, selected), batch_size=min(batch_size, len(selected)),
                        shuffle=True, generator=generator)
    iterator = iter(loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        try:
            x, y, yt, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, yt, _ = next(iterator)
        x, y, yt = x.to(device), y.to(device), yt.to(device)
        target = (y - torch.as_tensor(training.mean, device=device)) / torch.as_tensor(training.std, device=device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, yt, target, step)
        loss = torch.nn.functional.huber_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        losses.append(loss.item())
    return losses


def run(config, config_path, data_root, output, device, limit=0):
    spec, protocol = config["table21"], config["protocol"]
    source, _, source_meta = read_data(data_root, spec["source"])
    target, times, target_meta = read_data(data_root, spec["target"])
    if target.shape[1] < source.shape[1]:
        raise ValueError("Target must have at least as many nodes as the source checkpoint")
    subset = np.sort(np.random.RandomState(spec["subset_seed"]).choice(
        target.shape[1], source.shape[1], replace=False))
    target = target[:, subset]
    mean, std = fit_scaler(target)  # Both inputs and output conversion use target-training stats.
    if spec["adjacency"] == "target":
        raw_adj, adj, adj_meta = adjacency(data_root, spec["target"], len(subset), device, subset)
    elif spec["adjacency"] == "source":
        raw_adj, adj, adj_meta = adjacency(data_root, spec["source"], len(subset), device)
    else:
        raise ValueError("table21.adjacency must be target or source")
    records = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in spec["models"]:
        for seed in protocol["seeds"]:
            path = config["checkpoints"][spec["source"]][name].format(seed=seed)
            model, _, _, metadata = load_model(path, name, source, raw_adj, adj, device, spec["source"])
            seq, horizon = sequence_lengths(model)
            training = Windows(target, times, mean, std, "train", seq, horizon)
            test = Windows(target, times, mean, std, "test", seq, horizon, limit)
            if max(spec["budgets"]) > len(training):
                raise ValueError(f"Requested {max(spec['budgets'])} windows; only {len(training)} available")
            ordering = np.random.RandomState(spec["window_seed"]).permutation(len(training)).tolist()
            initial = copy.deepcopy(model.model.state_dict())
            for k in spec["budgets"]:
                model.model.load_state_dict(initial, strict=True)
                selected = ordering[:k]
                losses = adapt(model, training, selected, steps=spec["steps"],
                               batch_size=protocol["batch_size"], lr=spec["learning_rate"],
                               seed=spec["adaptation_seed"] + seed, device=device)
                error = Errors(np.zeros(target.shape[1], bool))
                model.eval()
                with torch.no_grad():
                    for batch, (x, y, yt, _) in enumerate(DataLoader(
                            test, batch_size=protocol["batch_size"], shuffle=False)):
                        set_seed(protocol["evaluation_seed"] + batch)
                        pred = model(x.to(device), yt.to(device))
                        error.add(pred, y.to(device), mean, std)
                records.append({"model": name, "seed": seed, "k": k, "steps": len(losses),
                                "training_window_indices": selected, "training_losses": losses,
                                "metrics": error.result()["whole"], "test_windows": len(test), **metadata})
            print(f"table21: {name}/seed{seed} complete", flush=True)
    summary = []
    for name in spec["models"]:
        for k in spec["budgets"]:
            vals = [r["metrics"]["masked_physical_mae"]["mae"] for r in records if r["model"] == name and r["k"] == k]
            summary.append({"model": name, "k": k, "masked_mph_mae": float(np.mean(vals)),
                            "seed_std": float(np.std(vals)), "seeds": len(vals)})
    result = {"implementation": IMPLEMENTATION, "table": "table21", "complete": True,
              "config_path": str(config_path), "config": config, "window_limit": limit,
              "runtime": runtime_metadata(device),
              "source_data": source_meta, "target_data": target_meta, "adjacency": adj_meta,
              "target_subset": subset.tolist(), "target_mean": mean.tolist(), "target_std": std.tolist(),
              "records": records, "summary": summary}
    save_json(output / "results.json", result)
    write_csv(output / "summary.csv", summary)
    return result
