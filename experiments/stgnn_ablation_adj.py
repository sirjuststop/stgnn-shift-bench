"""
EXP-1: GWNet Adaptive-Adjacency Ablation
Mechanism test: does removing the adaptive adj reduce sensor-dropout sensitivity?

Compares:
  (a) GWNet[adaptive] — trained with learned node embeddings (softmax(E1@E2^T))
  (b) GWNet[fixed]    — same trained weights but adaptive_adj=False; uses METR-LA fixed adj

Both evaluated under:
  - 30% zero-mask sensor dropout (Δ_sensor)
  - METR-LA→PEMS-BAY city transfer (Δ_city)

If GWNet[fixed] shows meaningfully lower Δ_sensor → adaptive adj is the mechanism.
"""
import json, os, pickle
import numpy as np
import torch

from stgnn_backbone_bank import (
    GWNet, load_metr_la, load_adjacency, make_dataloader, set_seed, _load_raw,
)
from stgnn_shift_eval_v2 import (
    load_pems_bay_subsets, evaluate_sensor_zeromask,
)


def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry_path = "baselines/stgnn_protocol_registry.json"
    adj_metr = "data/metr-la/adj_mx.pkl"
    adj_bay  = "data/pems-bay/adj_mx_bay.pkl"
    out_dir  = "results/tier1/"
    os.makedirs(out_dir, exist_ok=True)

    with open(registry_path) as f:
        registry = json.load(f)

    # Load METR-LA
    (_, _), (_, _), (X_te, Y_te), metr_stats = load_metr_la("data/metr-la")
    N = X_te.shape[2]
    adj_metr_t = load_adjacency(adj_metr, N, device)
    clean_loader = make_dataloader(X_te, Y_te, 64, shuffle=False)

    # PEMS-BAY subsets (5, same as v2)
    bay_subsets = load_pems_bay_subsets(
        "data/pems-bay", adj_bay, metr_stats, N, n_subsets=5,
    )

    gwnet_keys = sorted([k for k in registry if k.startswith("gwnet__metr-la__")])

    results = {}
    print(f"\n{'='*65}")
    print("GWNet Adaptive-Adj Ablation: adaptive vs fixed adjacency")
    print(f"{'='*65}")
    print(f"  {'Variant':<18} {'Δ_sensor':>10} {'city_tgt_MAE':>14}")
    print("  " + "-"*45)

    for key in gwnet_keys:
        entry = registry[key]
        seed  = entry["seed"]
        ckpt  = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
        cfg   = ckpt["config"]

        for variant, use_adaptive in [("adaptive", True), ("fixed", False)]:
            set_seed(seed)
            model = GWNet(
                num_nodes=N,
                seq_len=cfg["seq_len"],
                pred_len=cfg["pred_len"],
                hidden_dim=cfg["hidden_dim"],
                dropout=cfg["dropout"],
                adaptive_adj=use_adaptive,
            ).to(device)
            # Load weights; node_emb1/2 only exist in adaptive variant
            model.load_state_dict(ckpt["state_dict"], strict=False)
            model.eval()

            # Sensor dropout (zeromask)
            sd = evaluate_sensor_zeromask(model, X_te, Y_te, adj_metr_t, device)
            delta_sensor = sd["delta_sensor"]

            # City transfer (target-stat MAE, mean over 5 subsets)
            city_tgt_maes = []
            for sub in bay_subsets:
                adj_b = sub["adj_norm"].to(device)
                tgt_loader = make_dataloader(sub["X_tgt"], sub["Y_tgt"], 64, shuffle=False)
                from stgnn_backbone_bank import evaluate
                m = evaluate(model, tgt_loader, adj_b, device, stats=sub["tgt_stats"])
                city_tgt_maes.append(m["MAE"])

            city_tgt = float(np.mean(city_tgt_maes))
            label = f"gwnet[{variant}] s={seed}"
            results[f"{key}_{variant}"] = {
                "seed": seed, "variant": variant,
                "delta_sensor": delta_sensor,
                "city_tgt_MAE": city_tgt,
            }
            print(f"  {label:<18} {delta_sensor:>+10.4f} {city_tgt:>14.4f}")

    # Aggregate
    print(f"\n  {'Variant':<12} {'Δ_sensor mean±std':>20} {'city_tgt mean±std':>20}")
    print("  " + "-"*55)
    for variant in ["adaptive", "fixed"]:
        ds = [v["delta_sensor"] for v in results.values() if v["variant"] == variant]
        ct = [v["city_tgt_MAE"] for v in results.values() if v["variant"] == variant]
        print(f"  {variant:<12} {np.mean(ds):>+8.4f} ± {np.std(ds):.4f}   "
              f"{np.mean(ct):>8.4f} ± {np.std(ct):.4f}")

    # Mechanism check
    ds_adapt = [v["delta_sensor"] for v in results.values() if v["variant"]=="adaptive"]
    ds_fixed = [v["delta_sensor"] for v in results.values() if v["variant"]=="fixed"]
    reduction_pct = (np.mean(ds_adapt) - np.mean(ds_fixed)) / (abs(np.mean(ds_adapt)) + 1e-8) * 100
    threshold = 30.0
    status = "CONFIRMED" if reduction_pct >= threshold else "FAILED"
    print(f"\n  MECHANISM CHECK: fixed-adj reduces Δ_sensor by {reduction_pct:.1f}%")
    print(f"  (threshold ≥ {threshold}%) → [{status}]")

    out_path = os.path.join(out_dir, "ablation_adj_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "mechanism_reduction_pct": float(reduction_pct),
            "mechanism_status": status,
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
