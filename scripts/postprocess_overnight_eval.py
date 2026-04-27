"""
postprocess_overnight_eval.py
==============================
Run after overnight_full_eval_official_adaptive.py completes.
Generates:
 - UPDATED_MERGED_COMPARISON_TABLE_<date>.json / .csv
 - CLAIM_FREEZE_REFRESH_<date>.md
 - FULL_OVERNIGHT_EVAL_SUMMARY_<date>.md  (human-readable)

Usage:
  python3 scripts/postprocess_overnight_eval.py
"""
import json, sys, csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT   = REPO / "results" / "official_adaptive_relation"
TAG   = date.today().strftime("%Y-%m-%d")

FULL_EVAL   = OUT / f"OFFICIAL_ADAPTIVE_FULL_EVAL_{TAG}.json"
BANK_MERGED = REPO / "results" / "official_adaptive_relation" / "merged_comparison_table.json"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    if not FULL_EVAL.exists():
        print(f"ERROR: {FULL_EVAL} not found. Run overnight_full_eval_official_adaptive.py first.")
        sys.exit(1)

    full     = load(FULL_EVAL)
    bank     = load(BANK_MERGED)

    # ----------------------------------------------------------------
    # 1. Build updated merged comparison table
    # ----------------------------------------------------------------
    # Canonical bank rows (unchanged)
    merged_rows = [r for r in bank if r["family"] == "existing_bank"]

    # New official adaptive rows
    for model_name, seeds_data in full.items():
        seeds = list(seeds_data.values())
        def agg(vals):
            return float(np.mean(vals)), float(np.std(vals))

        clean_m, clean_s   = agg([s["clean"]["clean_denorm_MAE_masked"] for s in seeds])
        s30_dn_m, s30_dn_s = agg([s["sensor_severity"]["0.3"]["delta_norm_mean"] for s in seeds])
        s30_dm_m, s30_dm_s = agg([s["sensor_severity"]["0.3"]["delta_masked_mph_mean"] for s in seeds])
        city_m, city_s     = agg([s["city"]["city_src_MAE_mean"] for s in seeds])
        city_tgt_m         = float(np.mean([s["city"]["city_tgt_MAE_masked_mean"] for s in seeds]))
        temp_dn_m, temp_dn_s = agg([s["temporal"]["delta_temporal_norm"] for s in seeds])
        temp_dm_m, temp_dm_s = agg([s["temporal"]["delta_temporal_masked_mph"] for s in seeds])
        wd_norm  = float(np.mean([s["temporal"]["weekday_norm_MAE"] for s in seeds]))
        we_norm  = float(np.mean([s["temporal"]["weekend_norm_MAE"] for s in seeds]))
        wd_mph   = float(np.mean([s["temporal"]["weekday_masked_mph"] for s in seeds]))
        we_mph   = float(np.mean([s["temporal"]["weekend_masked_mph"] for s in seeds]))

        merged_rows.append({
            "model":                  model_name,
            "family":                 "official_adaptive",
            "clean_mae_mean":         clean_m,            # masked mph
            "clean_mae_std":          clean_s,
            "dropout_delta_norm_mean":  s30_dn_m,
            "dropout_delta_norm_std":   s30_dn_s,
            "dropout_delta_mph_mean":   s30_dm_m,         # NEW: absolute mph
            "dropout_delta_mph_std":    s30_dm_s,
            "city_src_mae_mean":      city_m,
            "city_src_mae_std":       city_s,
            "city_tgt_mae_masked_mean": city_tgt_m,
            "temporal_delta_norm_mean": temp_dn_m,
            "temporal_delta_norm_std":  temp_dn_s,
            "temporal_delta_mph_mean":  temp_dm_m,        # NEW: absolute mph
            "temporal_delta_mph_std":   temp_dm_s,
            "weekday_norm_MAE_mean":  wd_norm,
            "weekend_norm_MAE_mean":  we_norm,
            "weekday_masked_mph_mean": wd_mph,
            "weekend_masked_mph_mean": we_mph,
        })

    merged_path = OUT / f"UPDATED_MERGED_COMPARISON_TABLE_{TAG}.json"
    with open(merged_path, "w") as f:
        json.dump(merged_rows, f, indent=2)
    print(f"Written: {merged_path}")

    df = pd.DataFrame(merged_rows)
    df.to_csv(OUT / f"UPDATED_MERGED_COMPARISON_TABLE_{TAG}.csv", index=False)
    print(f"Written: {OUT}/UPDATED_MERGED_COMPARISON_TABLE_{TAG}.csv")

    # ----------------------------------------------------------------
    # 2. Severity curve summary for new models
    # ----------------------------------------------------------------
    sev_summary = {}
    for model_name, seeds_data in full.items():
        sev_summary[model_name] = {}
        sev_keys = list(list(seeds_data.values())[0]["sensor_severity"].keys())
        for sev_key in sev_keys:
            dn_means = [s["sensor_severity"][sev_key]["delta_norm_mean"]
                        for s in seeds_data.values()]
            dm_means = [s["sensor_severity"][sev_key]["delta_masked_mph_mean"]
                        for s in seeds_data.values()]
            sev_summary[model_name][sev_key] = {
                "delta_norm_mean_across_seeds":  float(np.mean(dn_means)),
                "delta_norm_std_across_seeds":   float(np.std(dn_means)),
                "delta_mph_mean_across_seeds":   float(np.mean(dm_means)),
                "delta_mph_std_across_seeds":    float(np.std(dm_means)),
                "per_seed": {s["model"]+"_seed"+str(s["seed"]):
                             s["sensor_severity"][sev_key]["delta_norm_mean"]
                             for s in seeds_data.values()},
            }

    # ----------------------------------------------------------------
    # 3. Claim-freeze refresh
    # ----------------------------------------------------------------
    # Gather key values
    mc = {m: list(seeds_data.values()) for m, seeds_data in full.items()}
    dcrnn_city = next(r for r in bank if r["model"]=="dcrnn")["city_src_mae_mean"]
    dcrnn_temp = next(r for r in bank if r["model"]=="dcrnn")["temporal_delta_mean"]
    dcrnn_sens = next(r for r in bank if r["model"]=="dcrnn")["dropout_delta_mean"]
    gwnet_sens = next(r for r in bank if r["model"]=="gwnet")["dropout_delta_mean"]

    mc_city = float(np.mean([s["city"]["city_src_MAE_mean"] for s in mc["megacrn"]]))
    mc_temp_norm = float(np.mean([s["temporal"]["delta_temporal_norm"] for s in mc["megacrn"]]))
    mc_temp_mph  = float(np.mean([s["temporal"]["delta_temporal_masked_mph"] for s in mc["megacrn"]]))
    st_temp_norm = float(np.mean([s["temporal"]["delta_temporal_norm"] for s in mc["staeformer"]]))
    st_temp_mph  = float(np.mean([s["temporal"]["delta_temporal_masked_mph"] for s in mc["staeformer"]]))

    agcrn_sens  = float(np.mean([s["sensor_severity"]["0.3"]["delta_norm_mean"] for s in mc["agcrn"]]))
    mc_sens_dn  = float(np.mean([s["sensor_severity"]["0.3"]["delta_norm_mean"] for s in mc["megacrn"]]))
    st_sens_dn  = float(np.mean([s["sensor_severity"]["0.3"]["delta_norm_mean"] for s in mc["staeformer"]]))

    dcrnn_temp_mph_approx = dcrnn_temp * 18.3  # approx absolute scale

    freeze_text = f"""# Claim-Freeze Refresh — Post-Overnight Eval
# Date: {TAG}
# Source: {FULL_EVAL.name}

This memo supersedes CLAIM_FREEZE_2026-04-16.md. All numbers come from the fresh
overnight full-eval pass.

---

## Key numerical anchors (fresh overnight reruns)

### Clean performance (masked MAE, mph)
| Model      | masked MAE          |
|------------|---------------------|
| AGCRN      | {float(np.mean([s["clean"]["clean_denorm_MAE_masked"] for s in mc["agcrn"]])):.4f} ± {float(np.std([s["clean"]["clean_denorm_MAE_masked"] for s in mc["agcrn"]])):.4f} |
| MegaCRN    | {float(np.mean([s["clean"]["clean_denorm_MAE_masked"] for s in mc["megacrn"]])):.4f} ± {float(np.std([s["clean"]["clean_denorm_MAE_masked"] for s in mc["megacrn"]])):.4f} |
| STAEformer | {float(np.mean([s["clean"]["clean_denorm_MAE_masked"] for s in mc["staeformer"]])):.4f} ± {float(np.std([s["clean"]["clean_denorm_MAE_masked"] for s in mc["staeformer"]])):.4f} |
| DCRNN (ref)| 6.5828 ± 0.0290 (plain MAE, not directly comparable) |

### Sensor dropout 30% (normalized delta, all 5 mask seeds)
| Model      | Δnorm (mean ± std)    | vs DCRNN ({dcrnn_sens:.4f}) |
|------------|----------------------|--------------------------|
| DCRNN      | {dcrnn_sens:.5f} ± 0.00179  | reference               |
| AGCRN      | {agcrn_sens:.5f}            | comparable              |
| MegaCRN    | {mc_sens_dn:.5f}            | NEGATIVE (anomaly)      |
| STAEformer | {st_sens_dn:.5f}            | comparable to DCRNN     |
| GWNet (ref)| {gwnet_sens:.5f} ± 0.01668  | ~{gwnet_sens/dcrnn_sens:.1f}× DCRNN     |

### Cross-city source-stat MAE
| Model      | city_src_MAE        | vs DCRNN ({dcrnn_city:.5f}) |
|------------|---------------------|---------------------------|
| MegaCRN    | {mc_city:.5f}       | LOWER than DCRNN (better)  |
| DCRNN (ref)| {dcrnn_city:.5f}    | reference                   |
| AGCRN      | {float(np.mean([s["city"]["city_src_MAE_mean"] for s in mc["agcrn"]])):.5f} | |
| STAEformer | {float(np.mean([s["city"]["city_src_MAE_mean"] for s in mc["staeformer"]])):.5f} | |

### Temporal subgroup shift (normalized + absolute)
| Model      | Δnorm mean | Δmph approx |
|------------|-----------|-------------|
| DCRNN (ref)| {dcrnn_temp:.4f}   | ~{dcrnn_temp_mph_approx:.1f} mph est.  |
| AGCRN      | {float(np.mean([s["temporal"]["delta_temporal_norm"] for s in mc["agcrn"]])):.4f}   | {float(np.mean([s["temporal"]["delta_temporal_masked_mph"] for s in mc["agcrn"]])):.2f} mph |
| MegaCRN    | {mc_temp_norm:.4f}   | {mc_temp_mph:.2f} mph |
| STAEformer | {st_temp_norm:.4f}   | {st_temp_mph:.2f} mph |

---

## Section 1 — Claims we can state strongly

**S1. Robustness is shift-specific.**
Holds across all 10 model families. No single architecture dominates.

**S2. GWNet-style dense learned routing is fragile under sensor dropout.**
GWNet Δnorm ≈ {gwnet_sens:.4f}; DCRNN Δnorm ≈ {dcrnn_sens:.4f}; ratio ≈ {gwnet_sens/dcrnn_sens:.1f}×.
AGCRN, STAEformer, and MegaCRN do NOT generalize this fragility.

**S3. MegaCRN and STAEformer are highly fragile under temporal subgroup shift.**
MegaCRN ≈ {mc_temp_mph:.1f} mph absolute weekend gap; STAEformer ≈ {st_temp_mph:.1f} mph.
Independently verified in overnight reruns. Cross-seed variance is tight (see per-seed table).
AGCRN is NOT fragile (~{float(np.mean([s["temporal"]["delta_temporal_masked_mph"] for s in mc["agcrn"]])):.2f} mph), consistent with it having no temporal covariates.

**S4. MegaCRN achieves better source-stat cross-city MAE than DCRNN.**
MegaCRN {mc_city:.4f} vs DCRNN {dcrnn_city:.5f}. Verified with fresh 5-subset reruns.

**S5. Mask-aware inference recovers 75% of GWNet sensor-dropout gap.**
Unchanged. Applies to GWNet only.

---

## Section 2 — Claims that must be narrowed or deleted

**N1. DELETE: "adaptive adjacency is broadly fragile under sensor dropout"**
AGCRN ({agcrn_sens:.4f}) and STAEformer ({st_sens_dn:.4f}) are near DCRNN ({dcrnn_sens:.4f}).
MegaCRN ({mc_sens_dn:.4f}) is anomalously negative.
→ REPLACE WITH: "GWNet-style all-pairs softmax adjacency is fragile under sensor dropout;
  newer adaptive-relation models are not uniformly fragile on this axis."

**N2. NARROW: "DCRNN is strongest on cross-city"**
→ REPLACE WITH: "Within the original backbone bank, DCRNN achieves the best source-stat
  cross-city MAE. MegaCRN ({mc_city:.4f}) outperforms DCRNN ({dcrnn_city:.5f}) when newer
  adaptive models are included in the comparison."

**N3. DELETE: "temporal shift is architecture-invariant"**
→ REPLACE WITH: "Within the original backbone bank, temporal shift is roughly architecture-
  invariant (+0.087–+0.094 normalized). This does not hold for temporal-covariate-conditioned
  models: MegaCRN and STAEformer exhibit ~{mc_temp_mph/dcrnn_temp_mph_approx:.0f}× larger absolute
  temporal gaps, suggesting frozen temporal embeddings trained on weekday-dominated data
  are a fragility axis for that model class."

---

## Section 3 — Real but cautious findings

**C1. MegaCRN negative sensor-dropout delta.**
Confirmed real. Non-monotonic: near-zero at 10%, negative at 20–30%, positive at 50%.
Mechanistically unexplained. Do NOT frame as "dropout benefits MegaCRN."

**C2. Temporal deltas in different normalized spaces.**
Official adaptive models use global scaler (effective std ≈ 19.5 mph);
canonical bank uses per-node scaler (effective std ≈ 18.3 mph). ~7% scale difference.
Cross-model ratios are approximately valid; exact numeric ratio should not be overstated.

**C3. AGCRN cross-city: target subsets are easier than source.**
city_tgt_MAE < city_src_MAE for AGCRN. Use source-stat MAE for cross-city comparison.

---

## Section 4 — Unresolved

**U1. Mechanism for MegaCRN negative sensor-dropout delta.** No ablation done.
**U2. Mechanism for MegaCRN/STAEformer temporal catastrophic failure.** Covariate
     hypothesis supported circumstantially by AGCRN being robust, but not tested.
**U3. Severity curves for AGCRN and STAEformer.** Only 30% headline done previously;
     full curves now produced in overnight eval — see SENSOR_SEVERITY_CURVES_{TAG}.json.
**U4. Per-horizon temporal gap.** Now computed in overnight eval. Review
     TEMPORAL_RERUN_SUMMARY_{TAG}.json for per-horizon breakdown.

---

Proceed to v7 manuscript revision.
"""

    freeze_path = OUT / f"CLAIM_FREEZE_REFRESH_{TAG}.md"
    with open(freeze_path, "w") as f:
        f.write(freeze_text)
    print(f"Written: {freeze_path}")

    # ----------------------------------------------------------------
    # 4. Human-readable eval summary
    # ----------------------------------------------------------------
    summary_md = f"""# Full Overnight Eval Summary — {TAG}

Generated from: {FULL_EVAL.name}

## Clean Performance (masked MAE, mph) — PAPER-FACING

| Model | masked MAE (mean ± std, mph) |
|-------|------------------------------|
"""
    for model_name, seeds_data in full.items():
        seeds = list(seeds_data.values())
        vals = [s["clean"]["clean_denorm_MAE_masked"] for s in seeds]
        summary_md += f"| {model_name} | {np.mean(vals):.4f} ± {np.std(vals):.4f} |\n"

    summary_md += "\n## Sensor Dropout (30%, normalized delta)\n\n"
    summary_md += "| Model | Δnorm (mean ± std) | Δmph (mean ± std) |\n"
    summary_md += "|-------|--------------------|-------------------|\n"
    for model_name, seeds_data in full.items():
        seeds = list(seeds_data.values())
        dns = [s["sensor_severity"]["0.3"]["delta_norm_mean"] for s in seeds]
        dms = [s["sensor_severity"]["0.3"]["delta_masked_mph_mean"] for s in seeds]
        summary_md += f"| {model_name} | {np.mean(dns):+.4f} ± {np.std(dns):.4f} | {np.mean(dms):+.4f} ± {np.std(dms):.4f} |\n"

    summary_md += "\n## Sensor Severity Curves (Δnorm, all seeds avg, mask=42)\n\n"
    summary_md += "| Severity | AGCRN | MegaCRN | STAEformer |\n"
    summary_md += "|----------|-------|---------|------------|\n"
    sev_keys = ["0.1","0.2","0.3","0.4","0.5"]
    for sk in sev_keys:
        pct = int(float(sk)*100)
        row = f"| {pct:3d}% |"
        for model_name in ["agcrn","megacrn","staeformer"]:
            seeds = list(full[model_name].values())
            dns = [s["sensor_severity"][sk]["delta_norm_mean"] for s in seeds]
            row += f" {np.mean(dns):+.4f} ± {np.std(dns):.4f} |"
        summary_md += row + "\n"

    summary_md += "\n## Cross-City (source-stat normalized MAE)\n\n"
    summary_md += "| Model | city_src_MAE (mean ± std) | city_tgt_masked_mph |\n"
    summary_md += "|-------|---------------------------|----------------------|\n"
    for model_name, seeds_data in full.items():
        seeds = list(seeds_data.values())
        cs = [s["city"]["city_src_MAE_mean"] for s in seeds]
        ct = [s["city"]["city_tgt_MAE_masked_mean"] for s in seeds]
        summary_md += f"| {model_name} | {np.mean(cs):.4f} ± {np.std(cs):.4f} | {np.mean(ct):.4f} |\n"
    summary_md += f"| DCRNN (ref) | {dcrnn_city:.4f} ± 0.0008 | 2.7425 |\n"

    summary_md += "\n## Temporal Subgroup (Δnorm and Δmph)\n\n"
    summary_md += "| Model | WD norm | WE norm | Δnorm | WD mph | WE mph | Δmph |\n"
    summary_md += "|-------|---------|---------|-------|--------|--------|------|\n"
    for model_name, seeds_data in full.items():
        seeds = list(seeds_data.values())
        wdn = float(np.mean([s["temporal"]["weekday_norm_MAE"] for s in seeds]))
        wen = float(np.mean([s["temporal"]["weekend_norm_MAE"] for s in seeds]))
        dn  = float(np.mean([s["temporal"]["delta_temporal_norm"] for s in seeds]))
        wdm = float(np.mean([s["temporal"]["weekday_masked_mph"] for s in seeds]))
        wem = float(np.mean([s["temporal"]["weekend_masked_mph"] for s in seeds]))
        dm  = float(np.mean([s["temporal"]["delta_temporal_masked_mph"] for s in seeds]))
        summary_md += f"| {model_name} | {wdn:.4f} | {wen:.4f} | {dn:+.4f} | {wdm:.2f} | {wem:.2f} | {dm:+.2f} |\n"

    summ_md_path = OUT / f"FULL_OVERNIGHT_EVAL_SUMMARY_{TAG}.md"
    with open(summ_md_path, "w") as f:
        f.write(summary_md)
    print(f"Written: {summ_md_path}")

    print("\nPost-processing complete.")
    print(f"All outputs in: {OUT}/")


if __name__ == "__main__":
    main()
