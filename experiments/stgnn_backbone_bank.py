"""
Backbone Bank Builder — Direction C: Shift-Aware STGNN
Trains each backbone ONCE on METR-LA with frozen config, saves checkpoints + registry.

Usage:
  python experiments/stgnn_backbone_bank.py \
    --backbones dcrnn gwnet gru \
    --seeds 0 1 2 \
    --output_dir baselines/stgnn/ \
    --registry baselines/stgnn_protocol_registry.json

Rule: after this script finishes, ALL checkpoints are FROZEN. Never re-train.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Data Loading ───────────────────────────────────────────────────────────────

def _load_raw(data_dir: str, prefix: str) -> np.ndarray:
    """Load METR-LA or PEMS-BAY from CSV (zenodo) or h5 (legacy). Returns (T, N)."""
    # Try CSV first (zenodo record 5724362 format)
    csv_path = os.path.join(data_dir, f"{prefix}.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1024:
        df = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        # CSV columns: timestamp(str), sensor0, sensor1, ...
        # genfromtxt converts timestamp to nan; drop col 0
        return df[:, 1:].astype(np.float32)
    # Try h5 (legacy DCRNN format)
    h5_path = os.path.join(data_dir, f"{prefix.lower().replace('-', '_')}.h5")
    if not os.path.exists(h5_path):
        h5_path = os.path.join(data_dir, f"{prefix}.h5")
    if os.path.exists(h5_path) and os.path.getsize(h5_path) > 1024:
        import h5py
        with h5py.File(h5_path, "r") as f:
            return f["df"]["block0_values"][:].astype(np.float32)
    raise FileNotFoundError(
        f"Data not found in {data_dir}. Expected {prefix}.csv or {prefix}.h5\n"
        f"Download: wget -O {csv_path} "
        f"https://zenodo.org/record/5724362/files/{prefix}.csv"
    )


def _windows(arr, seq_len, pred_len):
    """Materialize sliding windows (T, N) -> (num_windows, seq_len, N), (num_windows, pred_len, N)."""
    X, Y = [], []
    for t in range(len(arr) - seq_len - pred_len + 1):
        X.append(arr[t:t + seq_len])
        Y.append(arr[t + seq_len:t + seq_len + pred_len])
    return np.stack(X), np.stack(Y)


def _chronological_split(data, seq_len, pred_len, train_frac=0.7, val_frac=0.1):
    """70/10/20 chronological split + z-score normalization fit on train only."""
    T = data.shape[0]
    train_end = int(T * train_frac)
    val_end   = int(T * (train_frac + val_frac))
    train_raw = data[:train_end]
    val_raw   = data[train_end:val_end]
    test_raw  = data[val_end:]

    mean = train_raw.mean(axis=0, keepdims=True)
    std  = train_raw.std(axis=0, keepdims=True) + 1e-8
    train_norm = (train_raw - mean) / std
    val_norm   = (val_raw   - mean) / std
    test_norm  = (test_raw  - mean) / std

    X_tr, Y_tr = _windows(train_norm, seq_len, pred_len)
    X_va, Y_va = _windows(val_norm,   seq_len, pred_len)
    X_te, Y_te = _windows(test_norm,  seq_len, pred_len)
    return (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), {"mean": mean, "std": std}


def load_metr_la(data_dir: str, seq_len: int = 12, pred_len: int = 12):
    """Load METR-LA (CSV or h5) → train/val/test splits (speed, mph)."""
    data = _load_raw(data_dir, "METR-LA")
    return _chronological_split(data, seq_len, pred_len)


def load_pems04(data_dir: str, seq_len: int = 12, pred_len: int = 12):
    """Load PEMS04 from ASTGCN release (.npz with 'data' key, shape (T, N, 3)).

    Channels: 0=flow (veh/5min), 1=occupancy, 2=speed (mph).
    We use the FLOW channel as the target variable — this is the standard ASTGCN
    setting and is what all prior work reports MAE in. Units differ from METR-LA,
    so absolute mph values are not comparable across datasets; the paper's claim
    is about ranking preservation, not absolute error.
    """
    npz_path = os.path.join(data_dir, "pems04.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"PEMS04 data not found at {npz_path}. "
            f"Run scripts/download_pems04.sh to fetch it."
        )
    raw = np.load(npz_path)["data"]  # (T, N, F)
    if raw.ndim != 3:
        raise ValueError(f"PEMS04 expected shape (T,N,F); got {raw.shape}")
    flow = raw[:, :, 0].astype(np.float32)  # (T, N)
    return _chronological_split(flow, seq_len, pred_len)


# Dataset registry: maps dataset key → (loader_fn, default_data_dir, default_adj_path)
DATASET_REGISTRY = {
    "metr-la": (load_metr_la, "data/metr-la",       "data/metr-la/adj_mx.pkl"),
    "pems04":  (load_pems04,  "data/pems04",        "data/pems04/adj_mx.pkl"),
}


def load_adjacency(adj_path: str, num_nodes: int, device: torch.device):
    """Load adjacency matrix from pkl and return symmetric-normalized torch tensor.

    Accepts either the DCRNN-style 3-tuple (sensor_ids, sensor_id_to_ind, adj_mx)
    or a plain numpy/torch adjacency array (PEMS04 format).
    """
    import pickle
    with open(adj_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        adj = obj[2]
    else:
        adj = obj
    adj = torch.FloatTensor(np.asarray(adj)).to(device)
    # Symmetric normalization D^{-1/2} A D^{-1/2}
    D = adj.sum(dim=1).clamp(min=1e-8)
    D_inv_sqrt = D.pow(-0.5)
    adj_norm = D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(0)
    return adj_norm


# ── Lightweight Backbone Implementations ───────────────────────────────────────

class STGCNLayer(nn.Module):
    """Single STGCN block: temporal conv → graph conv → temporal conv."""
    def __init__(self, in_dim, out_dim, num_nodes, Ks=3, Kt=3):
        super().__init__()
        self.temporal1 = nn.Conv2d(in_dim, out_dim, (1, Ks), padding=(0, Ks // 2))
        self.spatial = nn.Conv2d(out_dim, out_dim, (1, 1))
        self.temporal2 = nn.Conv2d(out_dim, out_dim, (1, Ks), padding=(0, Ks // 2))
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # x: (B, C, N, T)
        x = self.relu(self.temporal1(x))
        # Graph convolution: X' = adj @ X (simplified spectral)
        x = torch.einsum("mn,bcnt->bcmt", adj, x)
        x = self.relu(self.spatial(x))
        x = self.relu(self.temporal2(x))
        return x


class STGCN(nn.Module):
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.layer1 = STGCNLayer(1, hidden_dim, num_nodes)
        self.layer2 = STGCNLayer(hidden_dim, hidden_dim, num_nodes)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * seq_len, pred_len)
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x, adj):
        # x: (B, T, N, 1) → (B, 1, N, T)
        x = x.squeeze(-1).permute(0, 3, 2, 1) if x.dim() == 4 else x.permute(0, 2, 1).unsqueeze(1)
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        x = self.dropout(x)
        # x: (B, C, N, T) → (B, N, C*T)
        B, C, N, T = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, N, C * T)
        out = self.output(x)  # (B, N, pred_len)
        return out


class GWNet(nn.Module):
    """
    Simplified Graph WaveNet with adaptive adjacency.
    Key feature: learns E1, E2 (node embeddings) → adaptive adj = softmax(E1 @ E2^T).
    This learned topology is the hypothesis's culprit for city-transfer brittleness.
    """
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=64, dropout=0.3,
                 adaptive_adj=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.adaptive_adj = adaptive_adj
        self.hidden_dim = hidden_dim

        if adaptive_adj:
            self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 10))
            self.node_emb2 = nn.Parameter(torch.randn(num_nodes, 10))

        self.input_proj = nn.Linear(seq_len, hidden_dim)
        self.graph_conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, pred_len)
        self.relu = nn.ReLU()

    def get_adaptive_adj(self, topology_dropout_p=0.0):
        """Build adaptive adjacency from learned node embeddings.
        topology_dropout_p > 0 implements Topology-Dropout intervention.
        """
        adj = torch.softmax(self.node_emb1 @ self.node_emb2.T, dim=-1)
        if topology_dropout_p > 0.0 and self.training is False:
            # Topology-Dropout: randomly zero out edges at test time
            mask = torch.bernoulli(
                torch.ones_like(adj) * (1 - topology_dropout_p)
            )
            adj = adj * mask
            # Re-normalize rows after dropout
            row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            adj = adj / row_sum
        return adj

    def graph_conv_layer(self, x, adj, layer, degree_norm=False):
        """Graph convolution with optional degree normalization (DN intervention)."""
        if degree_norm:
            # Degree normalization: D^{-1/2} A D^{-1/2}
            deg = adj.sum(dim=-1).clamp(min=1e-8)
            D_inv_sqrt = deg.pow(-0.5)
            adj = D_inv_sqrt.unsqueeze(-1) * adj * D_inv_sqrt.unsqueeze(-2)
        # x: (B, N, H); adj: (N, N)
        x_agg = torch.einsum("mn,bnh->bmh", adj, x)
        return layer(x_agg)

    def forward(self, x, adj_fixed, topology_dropout_p=0.0, degree_norm=False):
        # x: (B, T, N) or (B, T, N, 1)
        if x.dim() == 4:
            x = x.squeeze(-1)
        B, T, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, T)
        x = self.input_proj(x)  # (B, N, H)
        x = self.relu(x)

        if self.adaptive_adj:
            adj = self.get_adaptive_adj(topology_dropout_p)
        else:
            adj = adj_fixed

        x = self.relu(self.graph_conv_layer(x, adj, self.graph_conv1, degree_norm))
        x = self.dropout(x)
        x = self.relu(self.graph_conv_layer(x, adj, self.graph_conv2, degree_norm))
        x = self.dropout(x)
        out = self.output(x)  # (B, N, pred_len)
        return out


class DCRNNCell(nn.Module):
    """Diffusion convolution RNN cell (simplified single-step)."""
    def __init__(self, num_nodes, hidden_dim, diffusion_steps=2):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        input_size = (diffusion_steps * 2 + 1) * hidden_dim  # fwd + bwd + identity
        self.gate = nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2)
        self.update = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)

    def diffusion(self, x, adj):
        """Multi-step diffusion: sum of adj^k @ x for k=0..K."""
        out = x
        x_k = x
        for _ in range(self.diffusion_steps):
            x_k = torch.einsum("mn,bnh->bmh", adj, x_k)
            out = out + x_k
        return out / (self.diffusion_steps + 1)

    def forward(self, x, h, adj):
        # x, h: (B, N, H)
        x_diff = self.diffusion(x, adj)
        xh = torch.cat([x_diff, h], dim=-1)
        gates = torch.sigmoid(self.gate(xh))
        r, u = gates.chunk(2, dim=-1)
        c = torch.tanh(self.update(torch.cat([x_diff, r * h], dim=-1)))
        h_new = u * h + (1 - u) * c
        return h_new


class DCRNN(nn.Module):
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=64, dropout=0.3,
                 diffusion_steps=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)
        self.encoder = DCRNNCell(num_nodes, hidden_dim, diffusion_steps)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, pred_len)

    def forward(self, x, adj, **kwargs):
        # x: (B, T, N) or (B, T, N, 1)
        if x.dim() == 4:
            x = x.squeeze(-1)
        B, T, N = x.shape
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, N, T, 1)
        h = torch.zeros(B, N, self.hidden_dim, device=x.device)
        for t in range(T):
            inp = self.input_proj(x[:, :, t, :])  # (B, N, H)
            h = self.encoder(inp, h, adj)
        h = self.dropout(h)
        out = self.decoder(h)  # (B, N, pred_len)
        return out


class GRUBaseline(nn.Module):
    """Pure temporal GRU — no graph. Tests P2: graph-free model more shift-robust?"""
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(num_nodes, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_nodes * pred_len)
        self.num_nodes = num_nodes
        self.pred_len = pred_len

    def forward(self, x, adj=None, **kwargs):
        if x.dim() == 4:
            x = x.squeeze(-1)
        B, T, N = x.shape
        out, _ = self.gru(x)  # (B, T, H)
        out = self.dropout(out[:, -1, :])  # last step
        out = self.output(out).reshape(B, N, self.pred_len)
        return out


class MLPBaseline(nn.Module):
    """Flat MLP — no graph, no sequence structure. Absolute lower bound."""
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_nodes * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_nodes * pred_len),
        )
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.seq_len = seq_len

    def forward(self, x, adj=None, **kwargs):
        if x.dim() == 4:
            x = x.squeeze(-1)
        B, T, N = x.shape
        x_flat = x.reshape(B, T * N)
        out = self.net(x_flat).reshape(B, N, self.pred_len)
        return out


class MTGNNLight(nn.Module):
    """Lightweight MTGNN-inspired model (Wu et al., NeurIPS 2020).

    Key structural difference from GWNet: graph structure is learned via
    pairwise cosine similarity on a single embedding matrix with top-k sparsification,
    producing a sparse adaptive graph.  GWNet uses dense softmax(E1 @ E2^T) with two
    separate embedding matrices.  This sparser learned topology is our inductive
    baseline: if "adaptive adjacency hurts" is architecture-specific to GWNet's dense
    routing, MTGNNLight should not replicate the same fragility pattern.
    """
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=64, dropout=0.3,
                 top_k=10, embed_dim=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = min(top_k, num_nodes)
        # Single embedding matrix (unlike GWNet's E1, E2)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.input_proj = nn.Linear(seq_len, hidden_dim)
        self.graph_conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, pred_len)
        self.relu = nn.ReLU()

    def learn_adj(self):
        """Cosine-similarity graph with top-k sparsification (row-normalized)."""
        e = F.normalize(self.node_emb, p=2, dim=-1)  # (N, d)
        sim = e @ e.t()                               # (N, N) cosine similarity
        # Top-k neighbours per row
        _, topk_idx = sim.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(sim).scatter_(-1, topk_idx, 1.0)
        adj = F.relu(sim) * mask                      # non-negative, sparse
        row_sum = adj.sum(-1, keepdim=True).clamp(min=1e-8)
        return adj / row_sum                           # row-normalized

    def forward(self, x, adj_fixed=None, **kwargs):
        if x.dim() == 4:
            x = x.squeeze(-1)
        B, T, N = x.shape
        x = x.permute(0, 2, 1)          # (B, N, T)
        x = self.relu(self.input_proj(x))  # (B, N, H)
        adj = self.learn_adj()           # (N, N) — recomputed each forward
        x_agg = torch.einsum("mn,bnh->bmh", adj, x)
        x = self.relu(self.graph_conv1(x_agg))
        x = self.dropout(x)
        x_agg2 = torch.einsum("mn,bnh->bmh", adj, x)
        x = self.relu(self.graph_conv2(x_agg2))
        x = self.dropout(x)
        return self.output(x)           # (B, N, pred_len)


class GWNetStrong(GWNet):
    """Scaled-capacity GWNet for the P17 stronger-baseline control (Reviewer Fix v5).

    Relative to vanilla GWNet:
      * hidden_dim=128 (vs 64)           — passed via BACKBONE_CFG_OVERRIDES
      * node embedding dim = 20 (vs 10)  — set directly in __init__ below
      * early stopping patience = 30     — passed via BACKBONE_CFG_OVERRIDES

    Architecture is otherwise identical (adaptive-adjacency, 2 graph-conv layers),
    so P17 isolates the 'insufficient capacity' confound cleanly. A full published
    Wu et al. 2019 GWNet (dilated TCN stacks, residual + skip) would require a
    separate ~200 LoC rewrite; our scaled variant is the minimal-code reviewer fix
    and we describe it transparently in §3 Setup and Appendix G.
    """
    def __init__(self, num_nodes, seq_len, pred_len, hidden_dim=128, dropout=0.3):
        super().__init__(num_nodes, seq_len, pred_len,
                         hidden_dim=hidden_dim, dropout=dropout, adaptive_adj=True)
        # Overwrite node embeddings with 20-dim versions (double the vanilla GWNet)
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 20))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, 20))


BACKBONE_REGISTRY = {
    "stgcn": STGCN,
    "gwnet": GWNet,
    "gwnet_cm": GWNet,         # capacity-matched: same arch, patience=30 only
    "gwnet_strong": GWNetStrong,  # P17: hidden_dim=128, embed_dim=20, patience=30
    # gwnet_best: filled after gwnet_sweep.py Stage 2 completes;
    #             set to the best-config GWNetSweep variant for paper reporting.
    "dcrnn": DCRNN,
    "gru": GRUBaseline,
    "mlp": MLPBaseline,
    "mtgnn": MTGNNLight,
}

# Per-backbone training config overrides (applied on top of the default cfg in main())
BACKBONE_CFG_OVERRIDES = {
    "gwnet_cm":     {"early_stopping_patience": 30},
    "gwnet_strong": {"hidden_dim": 128, "early_stopping_patience": 30},
    # gwnet_best: populated after gwnet_sweep.py Stage 2 (see results/gwnet_sweep/)
}


# ── Training Loop ──────────────────────────────────────────────────────────────

def make_dataloader(X, Y, batch_size, shuffle):
    X_t = torch.FloatTensor(X)
    Y_t = torch.FloatTensor(Y)
    ds = torch.utils.data.TensorDataset(X_t, Y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, adj, device):
    model.train()
    total_loss = 0.0
    criterion = nn.HuberLoss()
    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b, adj)  # (B, N, pred_len)
        loss = criterion(pred, Y_b.permute(0, 2, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, adj, device, stats=None):
    """Evaluate model; preds and targets both returned as (total, N, pred_len)."""
    model.eval()
    preds, targets = [], []
    for X_b, Y_b in loader:
        X_b = X_b.to(device)
        pred = model(X_b, adj)             # (B, N, pred_len)
        preds.append(pred.cpu())
        # Y_b from make_windows: (B, pred_len, N) → permute to (B, N, pred_len)
        targets.append(Y_b.permute(0, 2, 1))
    preds   = torch.cat(preds,   dim=0)    # (total, N, pred_len)
    targets = torch.cat(targets, dim=0)    # (total, N, pred_len)

    # Denormalize if stats provided; mean/std shape (1, N) → reshape to (1, N, 1)
    if stats is not None:
        mean = torch.FloatTensor(stats["mean"]).reshape(1, -1, 1)  # (1, N, 1)
        std  = torch.FloatTensor(stats["std"]).reshape(1, -1, 1)
        preds   = preds   * std + mean
        targets = targets * std + mean

    mae  = (preds - targets).abs().mean().item()
    rmse = ((preds - targets) ** 2).mean().sqrt().item()
    mape = ((preds - targets).abs() / (targets.abs() + 1e-8)).mean().item() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def train_backbone(backbone_name, num_nodes, adj, train_loader, val_loader,
                   test_loader, cfg, device, stats):
    """Train a single backbone; return best val MAE and metrics."""
    model_cls = BACKBONE_REGISTRY[backbone_name]
    model = model_cls(
        num_nodes=num_nodes,
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, adj, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, adj, device, stats)
        val_mae = val_metrics["MAE"]

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"  [Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_MAE={val_mae:.4f}")

        if patience_counter >= cfg["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best checkpoint, evaluate on test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, adj, device, stats)
    return model, best_state, test_metrics


# ── Registry ───────────────────────────────────────────────────────────────────

def update_registry(registry_path, key, entry):
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {}
    registry[key] = entry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  Registry updated: {key}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+",
                        default=["dcrnn", "gwnet", "gru"],
                        choices=list(BACKBONE_REGISTRY.keys()),
                        help="Backbones to train. 'gwnet_cm' = GWNet with patience=30. "
                             "'gwnet_strong' = GWNet with hidden=128 & embed=20 (P17).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dataset", default="metr-la",
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Source dataset to train on. v5 adds 'pems04' (P18).")
    parser.add_argument("--output_dir", default="baselines/stgnn/")
    parser.add_argument("--registry", default="baselines/stgnn_protocol_registry.json")
    parser.add_argument("--data_dir", default=None,
                        help="Override data directory; defaults to DATASET_REGISTRY entry.")
    parser.add_argument("--adj_path", default=None,
                        help="Override adjacency pkl path; defaults to DATASET_REGISTRY entry.")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Optional global hidden_dim override (default uses frozen config "
                             "and per-backbone overrides in BACKBONE_CFG_OVERRIDES).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Resolve dataset loader + default paths
    loader_fn, default_data_dir, default_adj_path = DATASET_REGISTRY[args.dataset]
    data_dir = args.data_dir or default_data_dir
    adj_path = args.adj_path or default_adj_path

    # Frozen training config
    cfg = {
        "epochs": 100, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.3,
        "seq_len": 12, "pred_len": 12, "hidden_dim": 64, "batch_size": 64,
        "early_stopping_patience": 15,
    }
    if args.hidden_dim is not None:
        cfg["hidden_dim"] = args.hidden_dim

    print(f"Loading {args.dataset} dataset from {data_dir}...")
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), stats = loader_fn(
        data_dir, cfg["seq_len"], cfg["pred_len"]
    )
    num_nodes = X_tr.shape[2]
    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    print(f"  Num nodes: {num_nodes}")

    print(f"Loading adjacency matrix from {adj_path}...")
    adj = load_adjacency(adj_path, num_nodes, device)

    train_loader = make_dataloader(X_tr, Y_tr, cfg["batch_size"], shuffle=True)
    val_loader   = make_dataloader(X_va, Y_va, cfg["batch_size"], shuffle=False)
    test_loader  = make_dataloader(X_te, Y_te, cfg["batch_size"], shuffle=False)

    for backbone in args.backbones:
        # Apply per-backbone config overrides (e.g. gwnet_cm gets patience=30)
        backbone_cfg = {**cfg}
        if backbone in BACKBONE_CFG_OVERRIDES:
            backbone_cfg.update(BACKBONE_CFG_OVERRIDES[backbone])
            print(f"  [CFG OVERRIDE] {backbone}: {BACKBONE_CFG_OVERRIDES[backbone]}")

        per_seed_results = []
        for seed in args.seeds:
            key = f"{backbone}__{args.dataset}__seed{seed}"
            ckpt_path = os.path.join(args.output_dir, f"{key}.pt")

            if os.path.exists(ckpt_path):
                print(f"[SKIP] {key} already exists — backbone bank is frozen.")
                continue

            print(f"\n{'='*60}")
            print(f"Training {backbone.upper()} | seed={seed}")
            print(f"{'='*60}")
            set_seed(seed)
            t0 = time.time()

            model, state_dict, test_metrics = train_backbone(
                backbone, num_nodes, adj,
                train_loader, val_loader, test_loader,
                backbone_cfg, device, stats
            )

            elapsed = time.time() - t0
            print(f"  Test: MAE={test_metrics['MAE']:.4f}  RMSE={test_metrics['RMSE']:.4f}"
                  f"  MAPE={test_metrics['MAPE']:.2f}%  [{elapsed:.0f}s]")

            # Save frozen checkpoint
            torch.save({
                "backbone": backbone,
                "seed": seed,
                "state_dict": state_dict,
                "test_metrics": test_metrics,
                "config": backbone_cfg,
                "num_nodes": num_nodes,
            }, ckpt_path)
            print(f"  Checkpoint frozen: {ckpt_path}")

            per_seed_results.append(test_metrics)
            update_registry(args.registry, key, {
                "key": key,
                "backbone": backbone,
                "dataset": args.dataset,
                "seed": seed,
                "checkpoint": ckpt_path,
                "test_metrics": test_metrics,
                "status": "frozen",
            })

        if per_seed_results:
            maes = [r["MAE"] for r in per_seed_results]
            print(f"\n[{backbone.upper()}] {args.dataset} test MAE: "
                  f"{np.mean(maes):.4f} ± {np.std(maes):.4f}")

    print("\n=== Backbone Bank build complete ===")
    print(f"Registry: {args.registry}")


if __name__ == "__main__":
    main()
