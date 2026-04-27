import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_META = {
    "metr-la": {
        "csv_path": REPO_ROOT / "data" / "metr-la" / "METR-LA.csv",
        "num_nodes": 207,
        "steps_per_day": 288,
        "official_name": "METRLA",
    },
    "pems-bay": {
        "csv_path": REPO_ROOT / "data" / "pems-bay" / "PEMS-BAY.csv",
        "num_nodes": 325,
        "steps_per_day": 288,
        "official_name": "PEMSBAY",
    },
    "pems04": {
        "npz_path": REPO_ROOT / "data" / "pems04" / "pems04.npz",
        "num_nodes": 307,
        "steps_per_day": 288,
        "official_name": "PEMS04",
        "npz_start": "2018-01-01 00:00:00",
        "npz_freq": "5min",
    },
}


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def transform_numpy(self, data):
        return (data - self.mean) / self.std

    def inverse_transform_numpy(self, data):
        return data * self.std + self.mean

    def _torch_stats(self, data: torch.Tensor):
        mean = torch.as_tensor(self.mean.squeeze(0), dtype=data.dtype, device=data.device)
        std = torch.as_tensor(self.std.squeeze(0), dtype=data.dtype, device=data.device)
        if data.shape[-1] == mean.numel():
            shape = [1] * data.ndim
            shape[-1] = mean.numel()
        elif data.ndim >= 2 and data.shape[-2] == mean.numel():
            shape = [1] * data.ndim
            shape[-2] = mean.numel()
        else:
            raise ValueError(f"Cannot broadcast scaler with node dim {mean.numel()} to tensor shape {tuple(data.shape)}")
        return mean.view(*shape), std.view(*shape)

    def transform_torch(self, data: torch.Tensor) -> torch.Tensor:
        mean, std = self._torch_stats(data)
        return (data - mean) / std

    def inverse_transform_torch(self, data: torch.Tensor) -> torch.Tensor:
        mean, std = self._torch_stats(data)
        return data * std + mean

    @property
    def global_mean(self) -> float:
        return float(self.mean.mean())

    @property
    def global_std(self) -> float:
        return float(self.std.mean())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_parent(path: os.PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_traffic_frame(dataset_key: str) -> pd.DataFrame:
    if dataset_key not in DATASET_META:
        raise KeyError(f"Unsupported dataset: {dataset_key}")
    meta = DATASET_META[dataset_key]
    if "csv_path" in meta:
        df = pd.read_csv(meta["csv_path"], index_col=0, parse_dates=True)
        return df.astype(np.float32)
    if "npz_path" in meta:
        raw = np.load(meta["npz_path"])["data"]
        if raw.ndim != 3:
            raise ValueError(f"{dataset_key}: expected npz shape (T,N,F); got {raw.shape}")
        flow = raw[:, :, 0].astype(np.float32)
        index = pd.date_range(start=meta["npz_start"], periods=flow.shape[0], freq=meta["npz_freq"])
        columns = [str(i) for i in range(flow.shape[1])]
        return pd.DataFrame(flow, index=index, columns=columns)
    raise ValueError(f"{dataset_key}: no csv_path or npz_path in DATASET_META")


def split_traffic_frame(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.1):
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def fit_train_scaler(train_df: pd.DataFrame, mode: str = "per_node") -> StandardScaler:
    values = train_df.values.astype(np.float32)
    if mode == "per_node":
        mean = values.mean(axis=0, keepdims=True).astype(np.float32)
        std = values.std(axis=0, keepdims=True).astype(np.float32) + 1e-8
        return StandardScaler(mean, std)
    if mode == "global":
        mean_scalar = float(values.mean())
        std_scalar = float(values.std()) + 1e-8
        mean = np.full((1, values.shape[1]), mean_scalar, dtype=np.float32)
        std = np.full((1, values.shape[1]), std_scalar, dtype=np.float32)
        return StandardScaler(mean, std)
    raise ValueError(f"Unsupported scaler mode: {mode}")


def scaler_mode_for_model(model_name: str) -> str:
    if model_name in {"megacrn", "staeformer"}:
        return "global"
    return "per_node"


def scalar_source_scaler(source_scaler: StandardScaler, num_nodes: int) -> StandardScaler:
    mean = np.full((1, num_nodes), source_scaler.global_mean, dtype=np.float32)
    std = np.full((1, num_nodes), source_scaler.global_std, dtype=np.float32)
    return StandardScaler(mean, std)


def build_time_in_day(index: pd.Index, num_nodes: int) -> np.ndarray:
    timestamps = index.values
    time_ind = (timestamps - timestamps.astype("datetime64[D]")) / np.timedelta64(1, "D")
    feat = np.tile(time_ind[:, None], (1, num_nodes))
    return feat[..., None].astype(np.float32)


def build_day_of_week(index: pd.Index, num_nodes: int) -> np.ndarray:
    dow = index.dayofweek.to_numpy(dtype=np.float32)
    feat = np.tile(dow[:, None], (1, num_nodes))
    return feat[..., None].astype(np.float32)


def make_windows(arr: np.ndarray, seq_len: int = 12, pred_len: int = 12):
    x_list, y_list = [], []
    limit = len(arr) - seq_len - pred_len + 1
    for t in range(limit):
        x_list.append(arr[t : t + seq_len])
        y_list.append(arr[t + seq_len : t + seq_len + pred_len])
    return np.stack(x_list).astype(np.float32), np.stack(y_list).astype(np.float32)


def build_agcrn_arrays(frame: pd.DataFrame, scaler: StandardScaler, seq_len=12, pred_len=12):
    values = frame.values.astype(np.float32)
    values_norm = scaler.transform_numpy(values)
    return make_windows(values_norm[..., None], seq_len=seq_len, pred_len=pred_len)


def build_megacrn_arrays(frame: pd.DataFrame, scaler: StandardScaler, seq_len=12, pred_len=12):
    values = frame.values.astype(np.float32)
    values_norm = scaler.transform_numpy(values)[..., None]
    tod = build_time_in_day(frame.index, values.shape[1])
    data = np.concatenate([values_norm, tod], axis=-1)
    return make_windows(data, seq_len=seq_len, pred_len=pred_len)


def build_staeformer_arrays(frame: pd.DataFrame, scaler: StandardScaler, seq_len=12, pred_len=12):
    values = frame.values.astype(np.float32)
    values_norm = scaler.transform_numpy(values)[..., None]
    tod = build_time_in_day(frame.index, values.shape[1])
    dow = build_day_of_week(frame.index, values.shape[1])
    x_full = np.concatenate([values_norm, tod, dow], axis=-1)
    y_full = values[..., None]
    x, _ = make_windows(x_full, seq_len=seq_len, pred_len=pred_len)
    _, y = make_windows(y_full, seq_len=seq_len, pred_len=pred_len)
    return x, y


def make_torch_loader(x, y, batch_size, shuffle=False, drop_last=False):
    ds = torch.utils.data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred - target) ** 2).mean().sqrt().item()


def mape(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (((pred - target).abs() / (target.abs() + 1e-8)).mean().item() * 100.0)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    return {"MAE": mae(pred, target), "RMSE": rmse(pred, target), "MAPE": mape(pred, target)}


def _masked_values(pred: torch.Tensor, target: torch.Tensor, mask_value: float = 0.0):
    mask = torch.gt(target, mask_value)
    pred = torch.masked_select(pred, mask)
    target = torch.masked_select(target, mask)
    return pred, target


def compute_masked_metrics(pred: torch.Tensor, target: torch.Tensor, mask_value: float = 0.0):
    pred_masked, target_masked = _masked_values(pred, target, mask_value=mask_value)
    if pred_masked.numel() == 0:
        return {"MAE": float('nan'), "RMSE": float('nan'), "MAPE": float('nan')}
    return {
        "MAE": mae(pred_masked, target_masked),
        "RMSE": rmse(pred_masked, target_masked),
        "MAPE": mape(pred_masked, target_masked),
    }


def update_registry(registry_path: os.PathLike, key: str, entry: dict):
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path) as f:
            registry = json.load(f)
    else:
        registry = {}
    registry[key] = entry
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def load_registry(registry_path: os.PathLike) -> dict:
    with open(registry_path) as f:
        return json.load(f)


def build_source_dataset(dataset_key: str, seq_len=12, pred_len=12, scaler_mode: str = "per_node"):
    df = load_traffic_frame(dataset_key)
    train_df, val_df, test_df = split_traffic_frame(df)
    scaler = fit_train_scaler(train_df, mode=scaler_mode)
    return {
        "full": df,
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "scaler": scaler,
        "seq_len": seq_len,
        "pred_len": pred_len,
    }


def sample_target_subsets(
    target_dataset_key: str,
    subset_size: int,
    n_subsets: int = 5,
    base_seed: int = 42,
):
    df = load_traffic_frame(target_dataset_key)
    train_df, _, test_df = split_traffic_frame(df)
    subsets = []
    for i in range(n_subsets):
        rng = np.random.RandomState(base_seed + i)
        keep = sorted(rng.choice(df.shape[1], subset_size, replace=False).tolist())
        subsets.append(
            {
                "subset_id": i,
                "keep_idx": keep,
                "train": train_df.iloc[:, keep],
                "test": test_df.iloc[:, keep],
            }
        )
    return subsets
