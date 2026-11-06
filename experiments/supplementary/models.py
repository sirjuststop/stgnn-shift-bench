"""Adapters for the existing bank and pinned upstream GTS/DGCRN implementations."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.stgnn_backbone_bank import BACKBONE_REGISTRY
from experiments.official_adaptive_models import build_model_from_config
from experiments.supplementary.data import fit_scaler, sha256
from experiments.routing_sparsity_spectrum import (
    adj_from_agcrn, adj_from_gwnet, adj_from_megacrn, adj_from_mtgnn,
)
from third_party.supplementary_gts.model import GTSModel
from third_party.supplementary_dgcrn.net import DGCRN

ADAPTIVE = ("agcrn", "megacrn", "mtgnn", "gwnet", "gts", "dgcrn")
ALL_MODELS = tuple(BACKBONE_REGISTRY) + ("agcrn", "megacrn", "gts", "dgcrn")


def row_normalize(a):
    a = a.clamp_min(0)
    total = a.sum(-1, keepdim=True)
    normalized = a / total.clamp_min(1e-12)
    # Isolated rows route to self rather than becoming sub-stochastic.
    return torch.where(total > 0, normalized, torch.eye(a.shape[-1], device=a.device))


def defaults(name, nodes, seq_len=12, horizon=12, hidden=64, feature_steps=2016):
    if name in BACKBONE_REGISTRY:
        return dict(num_nodes=nodes, seq_len=seq_len, pred_len=horizon,
                    hidden_dim=128 if name == "gwnet_strong" and hidden == 64 else hidden,
                    dropout=.3)
    if name == "agcrn":
        return dict(num_nodes=nodes, input_dim=1, output_dim=1, horizon=horizon,
                    num_layers=2, rnn_units=hidden, embed_dim=10, cheb_k=2)
    if name == "megacrn":
        return dict(num_nodes=nodes, input_dim=1, output_dim=1, horizon=horizon,
                    num_layers=1, rnn_units=hidden, cheb_k=3, mem_num=20, mem_dim=hidden,
                    cl_decay_steps=2000, use_curriculum_learning=True)
    if name == "gts":
        if feature_steps < 19:
            raise ValueError("GTS requires at least 19 training timestamps for graph features")
        return dict(num_nodes=nodes, seq_len=seq_len, horizon=horizon, input_dim=2,
                    output_dim=1, rnn_units=hidden, num_rnn_layers=1,
                    max_diffusion_step=3, filter_type="dual_random_walk", cl_decay_steps=2000,
                    use_curriculum_learning=True, dim_fc=(feature_steps - 18) * 16,
                    feature_steps=feature_steps, temperature=.5)
    if name == "dgcrn":
        return dict(num_nodes=nodes, seq_length=seq_len, out_dim=horizon, gcn_depth=2,
                    in_dim=2, rnn_size=hidden, hyperGNN_dim=16, node_dim=40,
                    middle_dim=2, cl_decay_steps=2000)
    raise ValueError(name)


class Adapter(nn.Module):
    def __init__(self, name, config, raw_adj, symmetric_adj, device, node_features=None):
        super().__init__()
        self.name, self.config = name, dict(config)
        self.device = torch.device(device)
        self.register_buffer("raw_adj", raw_adj.to(device), persistent=False)
        self.register_buffer("adj", symmetric_adj.to(device), persistent=False)
        self.last_graph = None
        self.routing_sum, self.routing_count = 0., 0
        self.capture_routing = False
        cfg = dict(config)
        if name in BACKBONE_REGISTRY:
            self.model = BACKBONE_REGISTRY[name](**cfg).to(device)
        elif name in ("agcrn", "megacrn"):
            self.model = build_model_from_config(name, cfg, device)
        elif name == "gts":
            cfg.pop("feature_steps")
            temperature = cfg.pop("temperature")
            self.model = GTSModel(temperature, logging.getLogger("gts"), **cfg).to(device)
            if node_features is None or len(node_features) != config["feature_steps"]:
                raise ValueError("GTS node features must match saved training-feature length")
            self.register_buffer("node_features", torch.as_tensor(node_features, device=device),
                                 persistent=False)
            # Upstream cells create their parameters on first use: materialize BEFORE
            # loading state_dict or constructing an optimizer, in eval mode (no BN update).
            self.eval()
            with torch.no_grad():
                x = torch.zeros(2, cfg["seq_len"], cfg["num_nodes"], 2, device=device)
                yt = torch.zeros(2, cfg["horizon"], cfg["num_nodes"], device=device)
                self(x, yt)
        elif name == "dgcrn":
            predefined = [row_normalize(raw_adj).to(device), row_normalize(raw_adj.T).to(device)]
            self.model = DGCRN(device=torch.device(device), predefined_A=predefined, **cfg).to(device)
            original = self.model.preprocessing

            def observe(graph, predefined_a):
                result = original(graph, predefined_a)
                if self.capture_routing:
                    mass = result[0].detach().amax(dim=-1)
                    self.routing_sum += mass.double().sum().item()
                    self.routing_count += mass.numel()
                return result

            self.model.preprocessing = observe
        else:
            raise ValueError(name)

    def forward(self, x, future_time, labels_norm=None, batches_seen=0):
        if self.name in BACKBONE_REGISTRY:
            return self.model(x[..., 0], self.adj).transpose(1, 2)
        if self.name == "agcrn":
            return self.model(x[..., :1], None)[..., 0]
        if self.name == "megacrn":
            pred, *_ = self.model(x[..., :1], future_time[..., None],
                                 None if labels_norm is None else labels_norm[..., None], batches_seen)
            return pred[..., 0]
        if self.name == "gts":
            inputs = x.permute(1, 0, 2, 3).reshape(x.shape[1], x.shape[0], -1)
            labels = None if labels_norm is None else labels_norm.transpose(0, 1)
            pred, graph = self.model(None, inputs, self.node_features, self.config["temperature"],
                                     True, labels=labels, batches_seen=batches_seen)
            self.last_graph = graph
            return pred.transpose(0, 1)
        y = torch.stack([torch.zeros_like(future_time) if labels_norm is None else labels_norm,
                         future_time], dim=1).transpose(2, 3)
        pred = self.model(x.permute(0, 3, 2, 1), ycl=y, batches_seen=batches_seen,
                          task_level=future_time.shape[1])
        return pred[..., 0]

    def routing_mass(self):
        functions = {"agcrn": adj_from_agcrn, "megacrn": adj_from_megacrn,
                     "mtgnn": adj_from_mtgnn, "gwnet": adj_from_gwnet}
        if self.name in functions:
            # Preserve the published four-model proxy, including GWNet ReLU and
            # MegaCRN We1/We2 proxy; do not substitute a different estimand.
            graph = functions[self.name](self.model.state_dict())
            return row_normalize(graph).amax(-1).mean().item()
        if self.name == "gts":
            graph = self.last_graph.detach().clone()
            graph.fill_diagonal_(0.)
            return row_normalize(graph).amax(-1).mean().item()
        if self.name == "dgcrn":
            if self.routing_count == 0:
                raise ValueError("Run clean inputs with capture_routing=True first")
            return self.routing_sum / self.routing_count
        raise ValueError(f"No adaptive routing proxy for {self.name}")


def load_model(path, name, values, raw_adj, symmetric_adj, device, dataset):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint {path}; train it with the supplementary train command")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved_name = ckpt.get("model_name", ckpt.get("backbone"))
    if saved_name != name:
        raise ValueError(f"Checkpoint is {saved_name}, requested {name}: {path}")
    if ckpt.get("dataset", dataset) != dataset:
        raise ValueError(f"Checkpoint source dataset mismatch: {path}")
    if name in BACKBONE_REGISTRY:
        c = ckpt["config"]
        cfg = dict(num_nodes=ckpt["num_nodes"], seq_len=c["seq_len"], pred_len=c["pred_len"],
                   hidden_dim=c["hidden_dim"], dropout=c["dropout"])
    else:
        cfg = ckpt["model_config"]
    if cfg["num_nodes"] != values.shape[1]:
        raise ValueError(f"Checkpoint has {cfg['num_nodes']} nodes, data has {values.shape[1]}: {path}")
    if "scaler_mean" in ckpt:
        mean, std = np.array(ckpt["scaler_mean"], np.float32).reshape(-1), np.array(
            ckpt["scaler_std"], np.float32).reshape(-1)
    else:
        mean, std = fit_scaler(values)
    if mean.size not in (1, values.shape[1]) or std.size != mean.size or (std <= 0).any():
        raise ValueError("Checkpoint scaler shape or standard deviation is invalid")
    features = ckpt.get("node_features")
    if name == "gts" and features is None:
        raise ValueError("GTS checkpoint must include its training-only node_features")
    adapter = Adapter(name, cfg, raw_adj, symmetric_adj, device, features)
    adapter.model.load_state_dict(ckpt["state_dict"], strict=True)
    adapter.eval()
    metadata = {"checkpoint": str(path), "checkpoint_sha256": sha256(path),
                "checkpoint_origin": ckpt.get("implementation", "existing-bank"),
                "model_config": cfg}
    return adapter, mean, std, metadata


def sequence_lengths(adapter):
    cfg = adapter.config
    return cfg.get("seq_len", cfg.get("seq_length", 12)), cfg.get("pred_len", cfg.get("horizon", cfg.get("out_dim", 12)))
