"""Train missing native-source checkpoints without overwriting the frozen bank."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.official_adaptive_common import set_seed
from experiments.supplementary.data import (
    IMPLEMENTATION, Windows, adjacency, fit_scaler, read_data, split_bounds, runtime_metadata,
)
from experiments.supplementary.models import Adapter, defaults


def train(config, data_root, name, dataset, seed, output, device, epochs=None,
          max_batches=0, hidden=64, seq_len=12, horizon=12):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    settings = config["training"]
    set_seed(seed)
    values, times, data_meta = read_data(data_root, dataset)
    raw_adj, adj, adj_meta = adjacency(data_root, dataset, values.shape[1], device)
    mean, std = fit_scaler(values, "global" if name == "megacrn" else "per_node")
    feature_steps = min(settings["gts_feature_steps"], split_bounds(len(values))["train"][1])
    cfg = defaults(name, values.shape[1], seq_len, horizon, hidden, feature_steps)
    features = ((values[:feature_steps] - mean) / std).astype(np.float32) if name == "gts" else None
    model = Adapter(name, cfg, raw_adj, adj, device, features)
    training = Windows(values, times, mean, std, "train", seq_len, horizon)
    validation = Windows(values, times, mean, std, "val", seq_len, horizon)
    batch_size = config["protocol"]["batch_size"]
    loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    graph_target = None
    if name == "gts":
        feature_tensor = torch.as_tensor(features.T, device=device)
        similarity = F.normalize(feature_tensor, dim=-1) @ F.normalize(feature_tensor, dim=-1).T
        similarity.fill_diagonal_(-torch.inf)
        neighbours = similarity.topk(min(settings["gts_knn"], values.shape[1] - 1), -1).indices
        graph_target = torch.zeros_like(similarity).scatter_(-1, neighbours, 1.)
    best, best_state, wait, seen, history = float("inf"), None, 0, 0, []
    for epoch in range(epochs or settings["epochs"]):
        model.train()
        losses = []
        for batch, (x, y, yt, _) in enumerate(loader):
            if max_batches and batch >= max_batches:
                break
            x, y, yt = x.to(device), y.to(device), yt.to(device)
            labels = (y - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, yt, labels, seen)
            loss = F.huber_loss(pred, labels)
            if graph_target is not None:
                loss = loss + settings["gts_graph_loss_weight"] * F.binary_cross_entropy(
                    model.last_graph.clamp(1e-6, 1 - 1e-6), graph_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            optimizer.step()
            seen += 1
            losses.append(loss.item())
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for batch, (x, y, yt, _) in enumerate(val_loader):
                if max_batches and batch >= max_batches:
                    break
                set_seed(config["protocol"]["evaluation_seed"] + batch)
                x, y, yt = x.to(device), y.to(device), yt.to(device)
                target = (y - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device)
                errors = (model(x, yt) - target).abs()
                total += errors.double().sum().item()
                count += errors.numel()
        score = total / count
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_mae": score})
        if score < best:
            best, wait = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
        else:
            wait += 1
        print(f"{dataset}/{name}/seed{seed}: epoch={epoch+1} val={score:.6f}", flush=True)
        if wait >= settings["patience"]:
            break
    if best_state is None:
        raise RuntimeError("Training produced no finite validation score")
    ckpt = {"implementation": IMPLEMENTATION, "model_name": name, "backbone": name,
            "dataset": dataset, "seed": seed, "model_config": cfg,
            "config": cfg, "num_nodes": values.shape[1], "state_dict": best_state,
            "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
            "data": data_meta, "adjacency": adj_meta, "training": copy.deepcopy(settings),
            "epochs_requested": epochs or settings["epochs"], "max_batches": max_batches,
            "runtime": runtime_metadata(device),
            "history": history, "best_validation_mae": best}
    if features is not None:
        ckpt["node_features"] = torch.from_numpy(features.copy())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(ckpt, temporary)
    temporary.replace(output)
    return str(output)
