"""GWNet fairness check: retrain with patience=30 (vs 15), verify sensor-dropout fragility persists.
Addresses Codex reviewer concern: is the 4.4x sensor-dropout gap an artifact of GWNet under-tuning?
<30 LoC core logic. Output: results/tier2/gwnet_fairness_check.json"""
import json, sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from stgnn_backbone_bank import (BACKBONE_REGISTRY, load_metr_la, load_adjacency,
    make_dataloader, train_backbone, set_seed)
from stgnn_shift_eval_v2 import evaluate_sensor_zeromask
import numpy as np, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {"epochs": 150, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.3,
       "seq_len": 12, "pred_len": 12, "hidden_dim": 64, "batch_size": 64,
       "early_stopping_patience": 30}  # patience=30 vs standard 15

print("Loading METR-LA..."); (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te), stats = load_metr_la("data/metr-la")
N = X_tr.shape[2]; adj = load_adjacency("data/metr-la/adj_mx.pkl", N, device)
tr_l = make_dataloader(X_tr, Y_tr, 64, True); va_l = make_dataloader(X_va, Y_va, 64, False)
te_l = make_dataloader(X_te, Y_te, 64, False)

results = []
for seed in [0, 1, 2]:
    set_seed(seed); t0 = time.time()
    model, state, test_m = train_backbone("gwnet", N, adj, tr_l, va_l, te_l, cfg, device, stats)
    delta = evaluate_sensor_zeromask(model, X_te, Y_te, adj, device, dropout_ratio=0.30)
    elapsed = time.time() - t0
    print(f"seed={seed}: in-dist={test_m['MAE']:.4f}  Δ_sensor={delta['delta_sensor']:+.4f}  [{elapsed:.0f}s]")
    results.append({"seed": seed, "in_dist_MAE": test_m["MAE"], "delta_sensor": delta["delta_sensor"],
                    "relative_sensitivity": delta["delta_sensor"] / test_m["MAE"]})

os.makedirs("results/tier2", exist_ok=True)
summary = {"config": "patience=30 epochs=150", "results": results,
           "mean_in_dist": float(np.mean([r["in_dist_MAE"] for r in results])),
           "mean_delta_sensor": float(np.mean([r["delta_sensor"] for r in results])),
           "mean_rel_sensitivity": float(np.mean([r["relative_sensitivity"] for r in results]))}
print(f"\nSummary: in-dist={summary['mean_in_dist']:.4f}  Δ_sensor={summary['mean_delta_sensor']:+.4f}")
print(f"Relative sensitivity: {summary['mean_rel_sensitivity']:.4f}")
json.dump(summary, open("results/tier2/gwnet_fairness_check.json", "w"), indent=2)
print("Saved: results/tier2/gwnet_fairness_check.json")
