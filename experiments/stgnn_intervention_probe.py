"""
Intervention Probe — Direction C: Shift-Aware STGNN
Tests Topology-Dropout (TD) and Degree Normalization (DN) on frozen GWNet checkpoints.
No re-training. Pure test-time intervention.

Usage:
  python experiments/stgnn_intervention_probe.py \
    --backbone gwnet \
    --registry baselines/stgnn_protocol_registry.json \
    --interventions td dn td_dn none \
    --shift city \
    --output_dir results/tier0/interventions/
"""
import argparse
import json
import os

import numpy as np
import torch

from stgnn_backbone_bank import (
    GWNet, load_metr_la, load_adjacency, make_dataloader, set_seed
)


# ── Intervention Evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_with_intervention(model, loader, adj, device,
                                 topology_dropout_p=0.0, degree_norm=False,
                                 n_mc_samples=10):
    """Evaluate GWNet with TD and/or DN interventions.
    TD is stochastic → average over n_mc_samples forward passes.
    """
    assert isinstance(model, GWNet), "Interventions only implemented for GWNet."
    model.eval()
    preds_list = []

    for _ in range(n_mc_samples if topology_dropout_p > 0 else 1):
        batch_preds = []
        for X_b, _ in loader:
            X_b = X_b.to(device)
            pred = model(X_b, adj,
                         topology_dropout_p=topology_dropout_p,
                         degree_norm=degree_norm)
            batch_preds.append(pred.cpu())
        preds_list.append(torch.cat(batch_preds, dim=0))

    preds = torch.stack(preds_list, dim=0).mean(dim=0)  # MC average

    targets_list = []
    for _, Y_b in loader:
        targets_list.append(Y_b.permute(0, 2, 1))  # (B, pred_len, N) → (B, N, pred_len)
    targets = torch.cat(targets_list, dim=0)

    mae = (preds - targets).abs().mean().item()
    rmse = ((preds - targets) ** 2).mean().sqrt().item()
    return {"MAE": mae, "RMSE": rmse}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="gwnet")
    parser.add_argument("--registry", default="baselines/stgnn_protocol_registry.json")
    parser.add_argument("--interventions", nargs="+",
                        default=["none", "td", "dn", "td_dn"],
                        choices=["none", "td", "dn", "td_dn"])
    parser.add_argument("--shift", default="city", choices=["city", "sensor"])
    parser.add_argument("--output_dir", default="results/tier0/interventions/")
    parser.add_argument("--td_p", type=float, default=0.15,
                        help="Edge dropout probability for Topology-Dropout")
    parser.add_argument("--n_mc", type=int, default=10,
                        help="MC samples for TD (stochastic intervention)")
    parser.add_argument("--metr_la_dir", default="data/metr-la")
    parser.add_argument("--pems_bay_dir", default="data/pems-bay")
    parser.add_argument("--adj_path_metr", default="data/metr-la/adj_mx.pkl")
    parser.add_argument("--adj_path_bay", default="data/pems-bay/adj_mx_bay.pkl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load registry
    with open(args.registry) as f:
        registry = json.load(f)

    backbone_keys = [k for k in registry
                     if k.startswith(args.backbone + "__metr-la__")]
    if not backbone_keys:
        print(f"No frozen checkpoints found for {args.backbone}. Run backbone bank first.")
        return

    # Load METR-LA
    (_, _), (_, _), (X_te, Y_te), metr_stats = load_metr_la(args.metr_la_dir)
    num_nodes_metr = X_te.shape[2]
    adj_metr = load_adjacency(args.adj_path_metr, num_nodes_metr, device)
    clean_loader = make_dataloader(X_te, Y_te, 64, shuffle=False)

    # Load shift data (city transfer)
    if args.shift == "city":
        try:
            import pickle as _pickle
            from stgnn_shift_eval import load_pems_bay
            X_shift, Y_shift, _ = load_pems_bay(args.pems_bay_dir, metr_stats)
            num_nodes_bay = X_shift.shape[2]
            # Subsample to match METR-LA model capacity (fixed seed=42)
            if num_nodes_bay != num_nodes_metr:
                rng_city = np.random.RandomState(42)
                keep_bay = sorted(rng_city.choice(num_nodes_bay, num_nodes_metr, replace=False).tolist())
                X_shift = X_shift[:, :, keep_bay]
                Y_shift = Y_shift[:, :, keep_bay]
                with open(args.adj_path_bay, "rb") as _f:
                    _, _, adj_bay_full = _pickle.load(_f, encoding="latin1")
                adj_bay_sub = adj_bay_full[np.ix_(keep_bay, keep_bay)]
                adj_shift = torch.FloatTensor(adj_bay_sub).to(device)
                deg_bay = adj_shift.sum(dim=1).clamp(min=1e-8)
                D_inv_sqrt_bay = deg_bay.pow(-0.5)
                adj_shift = D_inv_sqrt_bay.unsqueeze(1) * adj_shift * D_inv_sqrt_bay.unsqueeze(0)
            else:
                adj_shift = load_adjacency(args.adj_path_bay, num_nodes_metr, device)
            shift_loader = make_dataloader(X_shift, Y_shift, 64, shuffle=False)
            print(f"City transfer: METR-LA → PEMS-BAY (subsampled to {num_nodes_metr} nodes)")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
    else:
        # Sensor dropout
        from stgnn_shift_eval import apply_sensor_dropout
        import pickle
        X_sparse, keep_idx = apply_sensor_dropout(X_te, dropout_ratio=0.30, seed=42)
        Y_sparse = Y_te[:, :, keep_idx]
        with open(args.adj_path_metr, "rb") as f_adj:
            _, _, adj_full = pickle.load(f_adj, encoding="latin1")
        import numpy as np_
        adj_sub_np = adj_full[np_.ix_(keep_idx, keep_idx)]
        adj_sub = torch.FloatTensor(adj_sub_np).to(device)
        deg = adj_sub.sum(dim=1).clamp(min=1e-8)
        D_inv_sqrt = deg.pow(-0.5)
        adj_shift = D_inv_sqrt.unsqueeze(1) * adj_sub * D_inv_sqrt.unsqueeze(0)
        shift_loader = make_dataloader(X_sparse, Y_sparse, 64, shuffle=False)
        print(f"Sensor dropout: {len(keep_idx)}/{num_nodes_metr} sensors kept")

    # Intervention configs
    intervention_configs = {
        "none":   {"topology_dropout_p": 0.0,      "degree_norm": False},
        "td":     {"topology_dropout_p": args.td_p, "degree_norm": False},
        "dn":     {"topology_dropout_p": 0.0,       "degree_norm": True},
        "td_dn":  {"topology_dropout_p": args.td_p, "degree_norm": True},
    }

    all_results = {}

    for key in sorted(backbone_keys):
        entry = registry[key]
        seed = entry["seed"]
        ckpt = torch.load(entry["checkpoint"], map_location=device)
        cfg = ckpt["config"]

        model = GWNet(
            num_nodes=num_nodes_metr,
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            adaptive_adj=True,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        print(f"\n[GWNet seed={seed}]  Shift: {args.shift}")
        print(f"  {'Intervention':<12} {'Clean MAE':>10} {'Shift MAE':>10} {'Δ':>8} {'Recovery %':>12}")
        print("  " + "-" * 55)

        seed_results = {}

        # Get clean MAE (no intervention)
        clean_metrics = {"MAE": float("nan")}

        for interv in args.interventions:
            if interv not in intervention_configs:
                continue
            ic = intervention_configs[interv]
            set_seed(seed)

            # Clean evaluation (no intervention)
            @torch.no_grad()
            def eval_clean():
                model.eval()
                preds, targets = [], []
                for X_b, Y_b in clean_loader:
                    X_b = X_b.to(device)
                    pred = model(X_b, adj_metr,
                                 topology_dropout_p=0.0, degree_norm=False)
                    preds.append(pred.cpu())
                    targets.append(Y_b.permute(0, 2, 1))  # (B, pred_len, N) → (B, N, pred_len)
                preds = torch.cat(preds, 0)
                targets = torch.cat(targets, 0)
                return (preds - targets).abs().mean().item()

            clean_mae = eval_clean()

            shift_metrics = evaluate_with_intervention(
                model, shift_loader,
                adj_shift if args.shift == "city" else adj_shift,
                device,
                topology_dropout_p=ic["topology_dropout_p"],
                degree_norm=ic["degree_norm"],
                n_mc_samples=args.n_mc,
            )
            shift_mae = shift_metrics["MAE"]
            delta = shift_mae - clean_mae

            # Recovery relative to no-intervention delta
            if "none" in seed_results:
                baseline_delta = seed_results["none"]["delta"]
                recovery_pct = (baseline_delta - delta) / (abs(baseline_delta) + 1e-8) * 100
            else:
                recovery_pct = 0.0  # for "none" itself

            seed_results[interv] = {
                "clean_MAE": clean_mae,
                "shift_MAE": shift_mae,
                "delta": delta,
                "recovery_pct": recovery_pct,
                "td_p": ic["topology_dropout_p"],
                "degree_norm": ic["degree_norm"],
            }
            print(f"  {interv:<12} {clean_mae:>10.4f} {shift_mae:>10.4f} "
                  f"{delta:>8.4f} {recovery_pct:>11.1f}%")

        all_results[key] = seed_results

    # Summary across seeds
    print(f"\n\n=== INTERVENTION SUMMARY ({args.backbone.upper()}, shift={args.shift}) ===")
    print(f"  {'Intervention':<12} {'Δ mean':>10} {'Δ std':>8} {'Recovery mean':>14}")
    print("  " + "-" * 48)
    for interv in args.interventions:
        deltas = []
        recoveries = []
        for key_results in all_results.values():
            if interv in key_results:
                deltas.append(key_results[interv]["delta"])
                recoveries.append(key_results[interv]["recovery_pct"])
        if deltas:
            print(f"  {interv:<12} {np.mean(deltas):>10.4f} {np.std(deltas):>8.4f} "
                  f"{np.mean(recoveries):>13.1f}%")

    # P3 check
    if "none" in args.interventions and "td_dn" in args.interventions:
        none_deltas = [all_results[k]["none"]["delta"] for k in all_results
                       if "none" in all_results[k]]
        td_dn_deltas = [all_results[k]["td_dn"]["delta"] for k in all_results
                        if "td_dn" in all_results[k]]
        if none_deltas and td_dn_deltas:
            rec = (np.mean(none_deltas) - np.mean(td_dn_deltas)) / (abs(np.mean(none_deltas)) + 1e-8) * 100
            threshold = 15.0
            status = "CONFIRMED" if rec >= threshold else "FAILED"
            print(f"\nP3 CHECK: TD+DN recovery = {rec:.1f}% (threshold ≥ {threshold}%) → [{status}]")

    # Save
    output_path = os.path.join(args.output_dir, f"intervention_{args.backbone}_{args.shift}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
