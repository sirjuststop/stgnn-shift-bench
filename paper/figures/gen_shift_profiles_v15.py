"""Generate the v15 grouped bar hero figure (replaces the radar).

3-panel grouped bar chart over sensor / cross-city / temporal shifts. Eight
backbones per panel, error bars, canonical (5) vs.\ extended (3) separated by
a vertical dashed line. The sensor panel additionally overlays PEMS04-native
ΔSensor as a second hatched bar on the four adaptive backbones (MTGNN,
GWNet, AGCRN, MegaCRN), making the two-city replication legible without a
separate figure.

Numbers come from §4 Tables 1–2 + results/polish_v15/pems04_native_routing_gate.json.
Error bars use the population-std convention from the canonical merged table.

Outputs:
  paper/figures/shift_profiles.pdf
  paper/figures/shift_profiles.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Headline numbers (from paper §4 Tables 1–2) ──────────────────────────────
MODELS = [
    "DCRNN", "MTGNNLight", "STGCN", "GRU", "GWNet",   # canonical (5)
    "AGCRN", "MegaCRN", "STAEformer",                  # extended (3)
]
N_CANONICAL = 5

SENSOR     = np.array([0.020, 0.051, 0.078, 0.025, 0.088,
                       0.019, -0.048, 0.015])
SENSOR_STD = np.array([0.002, 0.012, 0.007, 0.012, 0.017,
                       0.001, 0.027, 0.001])

CITY     = np.array([0.180, 0.244, 0.249, 0.278, 0.268,
                     0.206, 0.163, 0.223])
CITY_STD = np.array([0.001, 0.004, 0.008, 0.009, 0.006,
                     0.009, 0.003, 0.008])

TEMP     = np.array([0.087, 0.075, 0.093, 0.093, 0.094,
                     0.061, 0.734, 0.750])
TEMP_STD = np.array([0.004, 0.007, 0.003, 0.003, 0.009,
                     0.008, 0.019, 0.003])

# ── PEMS04-native ΔSensor overlay (on sensor panel only) ─────────────────────
def load_pems04():
    path = REPO_ROOT / "results" / "polish_v15" / "pems04_native_routing_gate.json"
    if not path.exists():
        return None
    with open(path) as f:
        gate = json.load(f)
    name_map = {"agcrn": "AGCRN", "megacrn": "MegaCRN",
                "mtgnn": "MTGNNLight", "gwnet": "GWNet"}
    means, stds = {}, {}
    for k, v in gate.get("per_model", {}).items():
        if k in name_map:
            means[name_map[k]] = v["delta_sensor_mean"]
            stds[name_map[k]]  = v["delta_sensor_std"]
    rho = gate.get("pearson_rho_pems04_native")
    return means, stds, rho


# Family-coded palette (avoids close-hue confusion on bar plots)
COLORS = {
    "DCRNN":      "#2ca02c",  # fixed graph - green
    "MTGNNLight": "#5fad56",  # sparse top-k - lighter green
    "STGCN":      "#7fc97f",  # fixed spectral - pale green
    "GRU":        "#888888",  # graph-free - gray
    "GWNet":      "#d62728",  # dense softmax - red
    "AGCRN":      "#1f77b4",  # adaptive - blue
    "MegaCRN":    "#9467bd",  # memory-meta - purple
    "STAEformer": "#17becf",  # attention - teal
}
BAR_COLORS = [COLORS[m] for m in MODELS]


def main():
    pems04 = load_pems04()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    x = np.arange(len(MODELS))
    bw = 0.65

    # ── Panel 1: Sensor dropout (with PEMS04 overlay) ───────────────────────
    ax1 = axes[0]
    if pems04 is not None:
        # Two-bar layout for the 4 adaptive backbones; single-bar for the rest
        adaptive_set = {"MTGNNLight", "GWNet", "AGCRN", "MegaCRN"}
        means_p, stds_p, rho = pems04
        # METR-LA bar (slightly left for the 4 adaptive cells, full width otherwise)
        for i, m in enumerate(MODELS):
            if m in adaptive_set:
                ax1.bar(x[i] - 0.18, SENSOR[i], 0.36, color=BAR_COLORS[i],
                        edgecolor="white", linewidth=0.5)
                ax1.errorbar(x[i] - 0.18, SENSOR[i], yerr=SENSOR_STD[i],
                             fmt="none", ecolor="black", capsize=2, linewidth=0.7)
                # PEMS04 hatched bar
                ax1.bar(x[i] + 0.18, means_p[m], 0.36, color=BAR_COLORS[i],
                        alpha=0.55, hatch="//", edgecolor="white", linewidth=0.5)
                ax1.errorbar(x[i] + 0.18, means_p[m], yerr=stds_p[m],
                             fmt="none", ecolor="black", capsize=2, linewidth=0.7)
            else:
                ax1.bar(x[i], SENSOR[i], bw, color=BAR_COLORS[i],
                        edgecolor="white", linewidth=0.5)
                ax1.errorbar(x[i], SENSOR[i], yerr=SENSOR_STD[i],
                             fmt="none", ecolor="black", capsize=2, linewidth=0.7)
    else:
        ax1.bar(x, SENSOR, bw, color=BAR_COLORS, edgecolor="white", linewidth=0.5)
        ax1.errorbar(x, SENSOR, yerr=SENSOR_STD, fmt="none", ecolor="black",
                     capsize=2, linewidth=0.7)

    ax1.set_ylabel(r"$\Delta_{\mathrm{sensor}}$ (norm)", fontsize=11)
    ax1.set_title("Sensor Dropout (30%)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8.5)
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax1.axvline(x=N_CANONICAL - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    # Panel-1 inline legend for the PEMS04 overlay (bottom-right)
    if pems04 is not None:
        rho_s = "n/a" if rho is None else f"{rho:+.2f}"
        ax1.legend(handles=[
            Patch(facecolor="lightgray", edgecolor="white", label="METR-LA"),
            Patch(facecolor="lightgray", alpha=0.55, hatch="//", edgecolor="white",
                  label=f"PEMS04 native (ρ={rho_s})"),
        ], fontsize=8, loc="upper left", frameon=False)

    # ── Panel 2: Cross-city ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(x, CITY, bw, color=BAR_COLORS, edgecolor="white", linewidth=0.5)
    ax2.errorbar(x, CITY, yerr=CITY_STD, fmt="none", ecolor="black",
                 capsize=2, linewidth=0.7)
    ax2.set_ylabel("Cross-city MAE (src-stat, norm)", fontsize=11)
    ax2.set_title("Cross-City Transfer", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8.5)
    ax2.axvline(x=N_CANONICAL - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    # ── Panel 3: Temporal subgroup shift ────────────────────────────────────
    ax3 = axes[2]
    ax3.bar(x, TEMP, bw, color=BAR_COLORS, edgecolor="white", linewidth=0.5)
    ax3.errorbar(x, TEMP, yerr=TEMP_STD, fmt="none", ecolor="black",
                 capsize=2, linewidth=0.7)
    ax3.set_ylabel(r"$\Delta_{\mathrm{temporal}}$ (norm)", fontsize=11)
    ax3.set_title("Temporal Subgroup Shift", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8.5)
    ax3.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax3.axvline(x=N_CANONICAL - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    # Canonical / Extended group annotations
    for ax in axes:
        ylim = ax.get_ylim()
        y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.05
        ax.text(2, y_pos, "Canonical", ha="center", fontsize=8,
                fontstyle="italic", color="dimgray")
        ax.text(6, y_pos, "Extended", ha="center", fontsize=8,
                fontstyle="italic", color="dimgray")

    plt.tight_layout()
    out_pdf = REPO_ROOT / "paper" / "figures" / "shift_profiles.pdf"
    out_png = REPO_ROOT / "paper" / "figures" / "shift_profiles.png"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
