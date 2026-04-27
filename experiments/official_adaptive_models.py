import sys
from argparse import Namespace
from pathlib import Path

import torch.nn as nn
import yaml

from experiments.official_adaptive_common import DATASET_META, REPO_ROOT


AGCRN_ROOT = REPO_ROOT / "external_repos" / "AGCRN"
if str(AGCRN_ROOT) not in sys.path:
    sys.path.insert(0, str(AGCRN_ROOT))

from model.AGCRN import AGCRN as OfficialAGCRN  # noqa: E402
from third_party.official_megacrn.MegaCRN import MegaCRN as OfficialMegaCRN  # noqa: E402
from third_party.official_staeformer.model.STAEformer import (  # noqa: E402
    STAEformer as OfficialSTAEformer,
)


MODEL_SOURCE_INFO = {
    "agcrn": {
        "paper_title": "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting",
        "venue_year": "NeurIPS 2020",
        "repo_url": "https://github.com/LeiBAI/AGCRN",
        "commit": "7fbbf2aeb099242098a3cf482b55cd45d7295c28",
        "default_datasets": ["PEMSD4", "PEMSD8"],
        "default_train_command": "python model/Run.py",
    },
    "megacrn": {
        "paper_title": "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting",
        "venue_year": "AAAI 2023",
        "repo_url": "https://github.com/deepkashiwa20/MegaCRN",
        "commit": "5158190948de7e8628f84780a429eb9189f5908f",
        "default_datasets": ["METRLA", "PEMSBAY", "EXPYTKY"],
        "default_train_command": "python traintest_MegaCRN.py --dataset=METRLA --gpu=0",
    },
    "staeformer": {
        "paper_title": "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting",
        "venue_year": "CIKM 2023",
        "repo_url": "https://github.com/XDZhelheim/STAEformer",
        "commit": "fc49d39b2f1a8e3cf37b6289d7240680e1690f3f",
        "default_datasets": ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"],
        "default_train_command": "cd model && python train.py -d METRLA -g 0",
    },
}


DEFAULT_TRAIN_CONFIG = {
    "agcrn": {
        "epochs": 100,
        "patience": 15,
        "batch_size": 64,
        "lr": 0.003,
        "weight_decay": 0.0,
        "grad_clip": 5.0,
    },
    "megacrn": {
        "epochs": 200,
        "patience": 20,
        "batch_size": 64,
        "lr": 0.01,
        "weight_decay": 0.0,
        "epsilon": 1e-3,
        "steps": [50, 100],
        "lr_decay_ratio": 0.1,
        "grad_clip": 5.0,
        "lamb": 0.01,
        "lamb1": 0.01,
    },
    "staeformer": None,
}


def resolve_train_config(model_name: str, dataset_key: str) -> dict:
    if model_name != "staeformer":
        return dict(DEFAULT_TRAIN_CONFIG[model_name])
    cfg_path = REPO_ROOT / "third_party" / "official_staeformer" / "model" / "STAEformer.yaml"
    with open(cfg_path) as f:
        yaml_cfg = yaml.safe_load(f)
    dataset_cfg = yaml_cfg[DATASET_META[dataset_key]["official_name"]]
    return {
        "epochs": dataset_cfg["max_epochs"],
        "patience": dataset_cfg["early_stop"],
        "batch_size": dataset_cfg["batch_size"],
        "lr": dataset_cfg["lr"],
        "weight_decay": dataset_cfg.get("weight_decay", 0.0),
        "milestones": dataset_cfg["milestones"],
        "lr_decay_ratio": dataset_cfg.get("lr_decay_rate", 0.1),
        "grad_clip": dataset_cfg.get("clip_grad", 0.0) or 0.0,
        "model_args": dataset_cfg["model_args"],
    }


def build_model_from_config(model_name: str, model_cfg: dict, device):
    if model_name == "agcrn":
        args = Namespace(**model_cfg, default_graph=True)
        model = OfficialAGCRN(args).to(device)
        apply_official_initialization(model_name, model)
        return model
    if model_name == "megacrn":
        return OfficialMegaCRN(
            num_nodes=model_cfg["num_nodes"],
            input_dim=model_cfg["input_dim"],
            output_dim=model_cfg["output_dim"],
            horizon=model_cfg["horizon"],
            rnn_units=model_cfg["rnn_units"],
            num_layers=model_cfg["num_layers"],
            cheb_k=model_cfg["cheb_k"],
            mem_num=model_cfg["mem_num"],
            mem_dim=model_cfg["mem_dim"],
            cl_decay_steps=model_cfg["cl_decay_steps"],
            use_curriculum_learning=model_cfg["use_curriculum_learning"],
        ).to(device)
    if model_name == "staeformer":
        return OfficialSTAEformer(**model_cfg).to(device)
    raise KeyError(f"Unsupported model: {model_name}")


def apply_official_initialization(model_name: str, model) -> None:
    if model_name != "agcrn":
        return
    # Official AGCRN re-initializes all parameters in Run.py after model construction.
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.uniform_(param)


def build_model(model_name: str, dataset_key: str, device):
    num_nodes = DATASET_META[dataset_key]["num_nodes"]
    if model_name == "agcrn":
        args = Namespace(
            num_nodes=num_nodes,
            input_dim=1,
            output_dim=1,
            horizon=12,
            num_layers=2,
            rnn_units=64,
            embed_dim=10,
            cheb_k=2,
            default_graph=True,
        )
        model = OfficialAGCRN(args).to(device)
        apply_official_initialization(model_name, model)
        model_cfg = {
            "num_nodes": num_nodes,
            "input_dim": 1,
            "output_dim": 1,
            "horizon": 12,
            "num_layers": 2,
            "rnn_units": 64,
            "embed_dim": 10,
            "cheb_k": 2,
        }
    elif model_name == "megacrn":
        model = OfficialMegaCRN(
            num_nodes=num_nodes,
            input_dim=1,
            output_dim=1,
            horizon=12,
            rnn_units=64,
            num_layers=1,
            cheb_k=3,
            mem_num=20,
            mem_dim=64,
            cl_decay_steps=2000,
            use_curriculum_learning=True,
        ).to(device)
        model_cfg = {
            "num_nodes": num_nodes,
            "input_dim": 1,
            "output_dim": 1,
            "horizon": 12,
            "rnn_units": 64,
            "num_layers": 1,
            "cheb_k": 3,
            "mem_num": 20,
            "mem_dim": 64,
            "cl_decay_steps": 2000,
            "use_curriculum_learning": True,
        }
    elif model_name == "staeformer":
        train_cfg = resolve_train_config("staeformer", dataset_key)
        model = OfficialSTAEformer(**train_cfg["model_args"]).to(device)
        model_cfg = dict(train_cfg["model_args"])
    else:
        raise KeyError(f"Unsupported model: {model_name}")
    return model, model_cfg
