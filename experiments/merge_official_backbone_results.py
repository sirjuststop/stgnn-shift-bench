import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "results" / "reviewer_fix_v5" / "metr_la" / "shift_eval_v2_results.json"
OFFICIAL_PATH = ROOT / "results" / "official_adaptive_relation" / "official_adaptive_shift_eval.json"
OUT_JSON = ROOT / "results" / "official_adaptive_relation" / "merged_comparison_table.json"
OUT_CSV = ROOT / "results" / "official_adaptive_relation" / "merged_comparison_table.csv"


def summarize_existing(payload):
    rows = []
    for model_name, seeds in payload.items():
        clean = [v["city"]["clean_denorm_MAE"] for v in seeds.values()]
        sensor = [v["sensor_zeromask"]["delta_sensor"] for v in seeds.values()]
        frag = [v["sensor_zeromask"]["delta_sensor"] / max(v["city"]["clean_denorm_MAE"], 1e-8) for v in seeds.values()]
        city = [v.get("city_source_adj_control", v["city"])["city_src_MAE_mean"] for v in seeds.values()]
        rows.append({
            "model": model_name,
            "family": "existing_bank",
            "clean_mae_mean": float(np.mean(clean)),
            "clean_mae_std": float(np.std(clean)),
            "clean_mae_plain_mean": float(np.mean(clean)),
            "clean_mae_plain_std": float(np.std(clean)),
            "dropout_delta_mean": float(np.mean(sensor)),
            "dropout_delta_std": float(np.std(sensor)),
            "relative_fragility_mean": float(np.mean(frag)),
            "relative_fragility_std": float(np.std(frag)),
            "city_src_mae_mean": float(np.mean(city)),
            "city_src_mae_std": float(np.std(city)),
        })
    return rows


def summarize_official(payload):
    rows = []
    for model_name, seeds in payload.items():
        clean = [v["clean"].get("clean_denorm_MAE_masked", v["clean"]["clean_denorm_MAE"]) for v in seeds.values()]
        clean_plain = [v["clean"]["clean_denorm_MAE"] for v in seeds.values()]
        sensor = [v["sensor_multimask"]["delta_sensor_mean"] for v in seeds.values()]
        frag = [v["sensor_multimask"].get("relative_fragility_masked", v["sensor_multimask"]["relative_fragility"]) for v in seeds.values()]
        city = [v["city"]["city_src_MAE_mean"] for v in seeds.values()]
        rows.append({
            "model": model_name,
            "family": "official_adaptive",
            "clean_mae_mean": float(np.mean(clean)),
            "clean_mae_std": float(np.std(clean)),
            "clean_mae_plain_mean": float(np.mean(clean_plain)),
            "clean_mae_plain_std": float(np.std(clean_plain)),
            "dropout_delta_mean": float(np.mean(sensor)),
            "dropout_delta_std": float(np.std(sensor)),
            "relative_fragility_mean": float(np.mean(frag)),
            "relative_fragility_std": float(np.std(frag)),
            "city_src_mae_mean": float(np.mean(city)),
            "city_src_mae_std": float(np.std(city)),
        })
    return rows


def main():
    rows = []
    if BASELINE_PATH.exists():
        rows.extend(summarize_existing(json.loads(BASELINE_PATH.read_text())))
    if OFFICIAL_PATH.exists():
        rows.extend(summarize_official(json.loads(OFFICIAL_PATH.read_text())))
    rows = sorted(rows, key=lambda x: (x["family"], x["dropout_delta_mean"]))
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    with open(OUT_CSV, "w") as f:
        headers = [
            "model",
            "family",
            "clean_mae_mean",
            "clean_mae_std",
            "clean_mae_plain_mean",
            "clean_mae_plain_std",
            "dropout_delta_mean",
            "dropout_delta_std",
            "relative_fragility_mean",
            "relative_fragility_std",
            "city_src_mae_mean",
            "city_src_mae_std",
        ]
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
