"""Generate the hero robustness radar for the v15 polish wave.

A 3-axis radar chart over (sensor / cross-city / temporal) shifts with 8
backbones overlaid as polylines. Per-axis min-max normalization brings the
three differently-scaled normalized fragility metrics onto a shared [0,1]
scale (lower = more robust on that axis).

Optionally overlays a thinner secondary trace per backbone showing PEMS04-native
fragility for the 4 adaptive backbones, when
results/polish_v15/pems04_native_routing_gate.json is present (Step 1 output).
This reads only the ΔSensor axis since the gate covers only sensor shift.

Companion small panel (right): per-horizon temporal-subgroup MAE for
DCRNN / MegaCRN / STAEformer when paper-sourced inline numbers exist; the
companion is omitted gracefully if no source is available, leaving a clean
single-radar figure.

Outputs:
  paper/figures/robustness_radar.pdf
  paper/figures/robustness_radar.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Canonical headline numbers (from paper §4 Table 1 + Table 2) ─────────────
MODELS = [
    "DCRNN", "MTGNNLight", "STGCN", "GRU", "GWNet",  # canonical (5)
    "AGCRN", "MegaCRN", "STAEformer",                 # extended (3)
]
N_CANONICAL = 5

# Sensor: ΔSensor (norm) at 30% zero-mask, 5 masks × 3 seeds
SENSOR = np.array([0.020, 0.051, 0.078, 0.025, 0.088,
                   0.019, -0.048, 0.016])

# Cross-city: source-stat normalized MAE on PEMS-BAY
CITY = np.array([0.180, 0.244, 0.249, 0.278, 0.268,
                 0.206, 0.163, 0.223])

# Temporal: weekend − weekday normalized MAE
TEMPORAL = np.array([0.087, 0.075, 0.094, 0.093, 0.094,
                     0.061, 0.734, 0.750])

SHIFT_LABELS = [
    r"$\Delta_{\mathrm{sensor}}$" "\n(30% zero-mask)",
    "Cross-city MAE\n(src-stat, norm)",
    r"$\Delta_{\mathrm{temporal}}$" "\n(weekend − weekday)",
]

# Color per backbone — group-coded (fixed/sparse green, dense red, modern blues, gray)
PALETTE = {
    "DCRNN":      "#2ca02c",  # fixed graph - green
    "MTGNNLight": "#5fad56",  # sparse top-k - lighter green
    "STGCN":      "#7fc97f",  # fixed spectral - pale green
    "GRU":        "#888888",  # graph-free - gray
    "GWNet":      "#d62728",  # dense softmax - red
    "AGCRN":      "#1f77b4",  # adaptive - blue
    "MegaCRN":    "#9467bd",  # memory-meta - purple
    "STAEformer": "#17becf",  # attention - teal
}


# ── Per-axis min-max normalization (low = robust) ───────────────────────────
def per_axis_minmax(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values), lo, hi
    return (values - lo) / (hi - lo), lo, hi


sensor_n, s_lo, s_hi = per_axis_minmax(SENSOR)
city_n, c_lo, c_hi   = per_axis_minmax(CITY)
temp_n, t_lo, t_hi   = per_axis_minmax(TEMPORAL)
RAW_RANGES = [(s_lo, s_hi), (c_lo, c_hi), (t_lo, t_hi)]
NORM = np.stack([sensor_n, city_n, temp_n], axis=1)  # (8, 3)


# ── Optional PEMS04 overlay ──────────────────────────────────────────────────
def load_pems04_overlay():
    """Return dict {backbone: ΔSensor_pems04_norm_to_metr_axis} or None."""
    path = REPO_ROOT / "results" / "polish_v15" / "pems04_native_routing_gate.json"
    if not path.exists():
        return None
    with open(path) as f:
        gate = json.load(f)
    name_map = {"agcrn": "AGCRN", "megacrn": "MegaCRN",
                "mtgnn": "MTGNNLight", "gwnet": "GWNet"}
    overlay_raw = {}
    for k, v in gate.get("per_model", {}).items():
        if k in name_map:
            overlay_raw[name_map[k]] = v["delta_sensor_mean"]
    # normalize against the same METR-LA min/max so overlay shares the radial scale
    overlay_norm = {m: (val - s_lo) / max(s_hi - s_lo, 1e-12)
                    for m, val in overlay_raw.items()}
    return overlay_raw, overlay_norm, gate.get("pearson_rho_pems04_native")


# ── Plot ─────────────────────────────────────────────────────────────────────
def make_radar(ax, theta, values_per_model, linestyle, alpha, lw, marker):
    for i, m in enumerate(MODELS):
        v = list(values_per_model[i]) + [values_per_model[i][0]]  # close polygon
        ax.plot(theta, v, color=PALETTE[m], linestyle=linestyle, lw=lw,
                marker=marker, markersize=4, alpha=alpha, label=m)


def main():
    overlay = load_pems04_overlay()

    fig = plt.figure(figsize=(10, 5.0))
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    n_axes = 3
    theta = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    theta += theta[:1]  # close

    make_radar(ax, theta, NORM, linestyle="-", alpha=0.95, lw=1.6, marker="o")

    # Optional PEMS04 overlay on the sensor axis only (single point per adaptive model)
    if overlay is not None:
        overlay_raw, overlay_norm, rho = overlay
        # Plot a small "×" at the sensor-axis angle for each PEMS04-trained adaptive model.
        # Single legend entry (combined) — the per-model color comes from the polyline above.
        first = True
        for m, v_norm in overlay_norm.items():
            ax.plot([theta[0]], [np.clip(v_norm, 0.0, 1.05)],
                    color=PALETTE[m], marker="x", markersize=8, linestyle="None",
                    markeredgewidth=1.8,
                    label="PEMS04-native (×)" if first else None)
            first = False

    # Axis labels
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(SHIFT_LABELS, fontsize=10)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["robust", "", "0.5", "", "fragile"], fontsize=8, color="gray")
    ax.set_ylim(0.0, 1.05)
    ax.set_rlabel_position(20)
    ax.grid(True, alpha=0.4)
    ax.spines["polar"].set_alpha(0.4)

    # Compose legend (split canonical vs extended)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uniq.append((h, l))
    handles, labels = zip(*uniq)
    leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.18, 0.5),
                    fontsize=9, frameon=False, ncol=1)

    # Annotate raw-range mapping below the figure
    raw_note = (
        f"Per-axis min–max normalization (low = robust). Raw ranges: "
        f"$\\Delta_\\mathrm{{sensor}}\\in[{s_lo:+.3f},{s_hi:+.3f}]$;  "
        f"X-city$\\in[{c_lo:.3f},{c_hi:.3f}]$;  "
        f"$\\Delta_\\mathrm{{temporal}}\\in[{t_lo:+.3f},{t_hi:+.3f}]$."
    )
    if overlay is not None:
        rho = overlay[2]
        rho_str = "n/a" if rho is None or np.isnan(rho) else f"{rho:+.3f}"
        raw_note += (
            f"  PEMS04-native ΔSensor overlaid as ×; cross-city Pearson ρ = {rho_str}."
        )
    fig.text(0.02, 0.02, raw_note, fontsize=7, color="dimgray", wrap=True)

    fig.suptitle("Robustness across three deployment shifts (8 backbones)",
                 fontsize=12, fontweight="bold", y=0.99)

    out_pdf = REPO_ROOT / "paper" / "figures" / "robustness_radar.pdf"
    out_png = REPO_ROOT / "paper" / "figures" / "robustness_radar.png"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
