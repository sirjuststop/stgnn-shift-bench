"""
EXP-2: Sensor Dropout Severity Sweep
Robustness curve: Δ_sensor at 10/20/30/40/50% zeromask dropout
for DCRNN, GWNet, GRU (3 seeds each).

Produces the key Figure 1 for the summer camp paper.
"""
import json, os
import numpy as np
import torch

from stgnn_backbone_bank import (
    BACKBONE_REGISTRY, load_metr_la, load_adjacency, make_dataloader, set_seed,
)
from stgnn_shift_eval_v2 import evaluate_sensor_zeromask


DROPOUT_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]
BACKBONES      = ["dcrnn", "gwnet", "gru"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "results/tier1/"
    os.makedirs(out_dir, exist_ok=True)

    with open("baselines/stgnn_protocol_registry.json") as f:
        registry = json.load(f)

    (_, _), (_, _), (X_te, Y_te), metr_stats = load_metr_la("data/metr-la")
    N = X_te.shape[2]
    adj_metr = load_adjacency("data/metr-la/adj_mx.pkl", N, device)

    all_results = {}

    print(f"\n{'='*70}")
    print("Sensor Dropout Severity Sweep")
    print(f"{'='*70}")
    header = f"{'Backbone':<10}" + "".join(f"  {int(r*100)}%{'':<4}" for r in DROPOUT_RATIOS)
    print(header)
    print("-"*70)

    for backbone_name in BACKBONES:
        bkeys = sorted([k for k in registry if k.startswith(backbone_name + "__metr-la__")])
        if not bkeys:
            continue

        # per-ratio: list of (seed, delta)
        ratio_deltas = {r: [] for r in DROPOUT_RATIOS}

        for key in bkeys:
            entry = registry[key]
            seed  = entry["seed"]
            ckpt  = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
            cfg   = ckpt["config"]
            set_seed(seed)

            model_cls = BACKBONE_REGISTRY[backbone_name]
            model = model_cls(
                num_nodes=N, seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
                hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            for ratio in DROPOUT_RATIOS:
                sd = evaluate_sensor_zeromask(
                    model, X_te, Y_te, adj_metr, device,
                    dropout_ratio=ratio, seed=42,
                )
                ratio_deltas[ratio].append(sd["delta_sensor"])

        row = f"{backbone_name:<10}"
        for ratio in DROPOUT_RATIOS:
            vals = ratio_deltas[ratio]
            row += f"  {np.mean(vals):>+6.4f}"
        print(row)

        all_results[backbone_name] = {
            str(r): {
                "mean": float(np.mean(ratio_deltas[r])),
                "std":  float(np.std(ratio_deltas[r])),
            }
            for r in DROPOUT_RATIOS
        }

    print("="*70)

    out_path = os.path.join(out_dir, "dropout_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick robustness check: is GWNet consistently worst?
    print("\nRobustness ranking check (GWNet > DCRNN at each ratio?):")
    for ratio in DROPOUT_RATIOS:
        gw = all_results.get("gwnet", {}).get(str(ratio), {}).get("mean", float("nan"))
        dc = all_results.get("dcrnn", {}).get(str(ratio), {}).get("mean", float("nan"))
        status = "✓" if gw > dc else "✗"
        print(f"  {int(ratio*100)}%: GWNet={gw:+.4f} DCRNN={dc:+.4f} {status}")


if __name__ == "__main__":
    main()
