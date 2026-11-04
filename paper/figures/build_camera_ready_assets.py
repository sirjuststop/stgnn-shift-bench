"""Regenerate revised tables and Figure 1 from existing submission/results.

Run with the repository's available Python with numpy/matplotlib installed.
No checkpoints are loaded and no experiment results are overwritten.
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
GENERATED = PAPER / "generated"
GENERATED.mkdir(exist_ok=True)
PACKAGED = "--from-packaged-inputs" in sys.argv[1:]
if any(arg != "--from-packaged-inputs" for arg in sys.argv[1:]):
    raise SystemExit("Usage: build_camera_ready_assets.py [--from-packaged-inputs]")
package = json.loads((PAPER / "inputs/asset_sources.json").read_text(encoding="utf-8")) if PACKAGED else None
SOURCES = [
    "submission_neurips2026_20260506.zip",
    "results/official_adaptive_relation/TEMPORAL_RERUN_SUMMARY_2026-04-16.json",
    "results/routing_sparsity_spectrum/ROUTING_SPARSITY_SPECTRUM_2026-04-17.json",
    "results/polish_v15/pems04_native_routing_gate.json",
    "results/polish_v15/mask_aware_ratio_sweep.json",
    "results/reviewer_fix_v4/shift_eval_v2_results.json",
    "comments.txt",
]


def load(rel):
    if PACKAGED:
        return package["result_inputs"][rel]
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def write_tex(name, content):
    (GENERATED / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def correlation(x, y):
    a = [v-statistics.mean(x) for v in x]
    b = [v-statistics.mean(y) for v in y]
    return sum(u*v for u, v in zip(a, b)) / math.sqrt(sum(u*u for u in a)*sum(v*v for v in b))


def pair(mean, std):
    return rf"${mean:+.3f}\pm{std:.3f}$"


temporal_raw = load(SOURCES[1])
display = {"agcrn": "AGCRN", "megacrn": "MegaCRN", "staeformer": "STAEformer"}
temporal = {}
for m, seeds in temporal_raw.items():
    n = [v["delta_temporal_norm"] for v in seeds.values()]
    p = [v["delta_temporal_masked_mph"] for v in seeds.values()]
    assert len(n) == len(p) == 3
    temporal[m] = {"n_seeds": len(n), "norm_mean": statistics.mean(n),
        "norm_std": statistics.pstdev(n), "mph_mean": statistics.mean(p), "mph_std": statistics.pstdev(p)}

rows = [display[m] + " & " + pair(v["norm_mean"], v["norm_std"]) + " & "
        + pair(v["mph_mean"], v["mph_std"]) + r" \\" for m, v in temporal.items()]
write_tex("temporal_absolute.tex", r"""% Generated from TEMPORAL_RERUN_SUMMARY_2026-04-16.json; population std across 3 seeds.
\begin{table}[h]
\centering
\small
\caption{Weekday/weekend gap under two reporting conventions. The same
checkpoints and test windows are used in both columns; normalized MAE uses
all targets, while masked mph excludes zero targets. Negative gaps mean
lower weekend error. Mean $\pm$ population std over three seeds.}
\label{tab:temporal_absolute}
\begin{tabular}{lcc}
\toprule
\textbf{Model} & \textbf{Normalized/plain gap} & \textbf{Masked-mph gap} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

routing = load(SOURCES[2])
models = ["agcrn", "megacrn", "mtgnn", "gwnet"]
x = [routing[m]["max_mass_mean"] for m in models]
y = [routing[m]["fragility_delta_sensor"] for m in models]
rho = correlation(x, y)
loo = {m: correlation(x[:i]+x[i+1:], y[:i]+y[i+1:]) for i, m in enumerate(models)}
perms = [correlation(x, p) for p in itertools.permutations(y)]
perm_p = sum(abs(r) >= abs(rho)-1e-12 for r in perms) / len(perms)
assert abs(rho-0.9788268581749475) < 1e-12 and abs(perm_p-2/24) < 1e-12
names = {**display, "mtgnn": "MTGNN-Light", "gwnet": "GWNet"}
write_tex("routing_loo.tex", r"""% Generated from the original routing-spectrum estimates, n=4 architectures.
\begin{table}[h]
\centering
\small
\caption{Leave-one-out Pearson correlations for the original four-model
routing-spectrum estimates. The full-sample Pearson is $+0.978827$; the exact
absolute-value-tail permutation test over all 24 permutations gives $p=0.0833$.}
\label{tab:routing_loo}
\begin{tabular}{lc}
\toprule
\textbf{Excluded model} & \textbf{Remaining-model Pearson} \\
\midrule
""" + "\n".join(names[m] + rf" & ${v:+.6f}$ \\" for m, v in loo.items()) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

mask = load(SOURCES[4])["ratio_sweep"]["r30"]
write_tex("mask_aware_rows.tex", r"""% Seed means over five masks; source: results/polish_v15/mask_aware_ratio_sweep.json
\begin{tabular}{lcc}
\toprule
\textbf{Variant} & \textbf{$\DeltaSensor$ (30\%, multi-mask)} & \textbf{Recovery} \\
\midrule
"""
    + "GWNet[standard] & " + pair(mask["delta_standard_mean"], mask["delta_standard_std"]) + r" & --- \\" + "\n"
    + "GWNet[mask-aware] & " + pair(mask["delta_aware_mean"], mask["delta_aware_std"])
    + rf" & ${mask['recovery_mean']:.1f}\%$ \\" + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}")

fp_raw = load(SOURCES[5])["gwnet"]
fp_stats = {}
fp_rows = []
for cell, title, rates in [("fp0_fn0", "Perfect mask", r"$0\%/0\%$"),
                          ("fp5_fn10", "Conservative detector", r"$5\%/10\%$"),
                          ("fp10_fn20", "Noisy detector", r"$10\%/20\%$")]:
    vals = [v["mask_fp_fn"][cell]["delta_from_perfect"] for v in fp_raw.values()]
    fp_stats[cell] = {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "median": statistics.median(vals)}
    v = fp_stats[cell]
    fp_rows.append(title + " & " + rates + " & " + pair(v["mean"], v["std"]) + rf" & ${v['median']:+.3f}$ \\")
write_tex("fpfn_rows.tex", r"""% Paired primary-mask penalties from reviewer_fix_v4; no recovery-denominator mixing.
\begin{tabular}{lccc}
\toprule
\textbf{Detector scenario} & \textbf{FP / FN} & \textbf{Mean $\pm$ std} & \textbf{Median} \\
\midrule
""" + "\n".join(fp_rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}")

# Transcribe tables actually preserved in the authors' replies. These are
# reported aggregates, kept separate from the statistics recomputed above.
if PACKAGED:
    corruption_rows = package["reported_aggregates"]["corruptions"]
    severity_rows = package["reported_aggregates"]["severity"]
    fewshot_rows = package["reported_aggregates"]["fewshot"]
else:
    comment_lines = (ROOT / "comments.txt").read_text(encoding="utf-8").splitlines()
    comment_text = "\n".join(comment_lines)
    assert "At n=6 the correlation is ρ = +0.974" in comment_text
    assert "smallest leave-one-out value is +0.961" in comment_text
    corruption_rows = []
    for line in comment_lines[102:108]:
        parts = line.split()
        assert len(parts) == 6, parts
        corruption_rows.append({"model": parts[0], "deltas_mph": [float(v) for v in parts[1:]]})
    severity_rows = []
    for line in comment_lines[92:97]:
        match = re.fullmatch(r"(.+?)\s+\(([^)]+)\)\s+(.+)", line)
        assert match is not None, line
        values = [float(v) for v in match.group(3).split()]
        assert len(values) == 5
        severity_rows.append({"model": match.group(1), "routing_mass_label": match.group(2), "deltas_mph": values})
    fewshot_rows = []
    for line in comment_lines[245:249]:
        parts = line.split()
        assert len(parts) == 5, parts
        fewshot_rows.append({"model": parts[0], "mae_mph": [float(v) for v in parts[1:]]})

def reported_table(filename, caption, label, headers, rows):
    text = r"\begin{table}[h]" + "\n" + r"\centering\small" + "\n"
    text += r"\caption{" + caption + "}\n" + r"\label{" + label + "}\n"
    text += r"\begin{tabular}{l" + "c"*(len(headers)-1) + "}\n" + r"\toprule" + "\n"
    text += " & ".join(r"\textbf{" + h + "}" for h in headers) + r" \\" + "\n" + r"\midrule" + "\n"
    text += "\n".join(" & ".join(row) + r" \\" for row in rows)
    text += "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
    write_tex(filename, text)

reported_table("corruption_summary.tex",
    "Additional sensor corruptions on the canonical bank and GWNet[Strong] control: 30\\% affected channels, healthy-sensor split. Entries are MAE increases in mph relative to clean inputs.",
    "tab:corruption_summary", ["Backbone", "Zero", "Stale", "Noise", "Bias", "Intermittent"],
    [[v["model"]]+[f"${n:+.2f}$" for n in v["deltas_mph"]] for v in corruption_rows])
reported_table("severity_extended.tex",
    "Physical-unit zero-fill severity comparison, healthy-sensor split. GTS and DGCRN extend the concentration range; routing-mass values are rounded. DCRNN uses a fixed graph.",
    "tab:severity_extended", ["Backbone", "Max mass", r"10\%", r"20\%", r"30\%", r"40\%", r"50\%"],
    [[v["model"], "---" if v["routing_mass_label"]=="fixed graph" else v["routing_mass_label"]]
     +[f"${n:+.2f}$" for n in v["deltas_mph"]] for v in severity_rows])
reported_table("fewshot_summary.tex",
    "Limited target-data adaptation: target-test MAE in mph on one fixed 207-node PEMS-BAY subset, using target normalization. This is separate from the five-subset source-normalized main comparison.",
    "tab:fewshot_summary", ["Model", "$k=0$", "$k=50$", "$k=200$", "$k=800$"],
    [[v["model"]]+[f"${n:.2f}$" for n in v["mae_mph"]] for v in fewshot_rows])
native_reported = [
    {"dataset": "METR-LA", "quantity": "Speed", "models": 6, "rho": .974, "p": .003},
    {"dataset": "PEMS-BAY", "quantity": "Speed", "models": 6, "rho": .75, "p": .014},
    {"dataset": "PEMS04", "quantity": "Flow", "models": 6, "rho": .63, "p": .06},
]
# For n=6, the two-sided Pearson t test (df=4) has the closed form
# 1 - 3*abs(r)/2 + abs(r)**3/2. Use the reported rounded coefficients.
native_t_checks = [{"dataset": v["dataset"], "rho": v["rho"], "n": 6,
                   "df": 4, "two_sided_t_p": 1 - 1.5*abs(v["rho"]) + .5*abs(v["rho"])**3}
                  for v in native_reported]
reported_table("routing_extended.tex",
    r"Six-architecture native-source routing-concentration correlations. Approximate two-sided Pearson $t$ tests ($p_t$, 4 degrees of freedom) use rounded correlations. Test details: Appendix~\ref{sec:appendix_routing_extended}.",
    "tab:routing_extended", ["Source dataset", "Quantity", "$n$", "Pearson", r"Approx. $p_t$"],
    [[v["dataset"], v["quantity"], str(v["models"]), f"${v['rho']:+.3f}$", f"${c['two_sided_t_p']:.3f}$"] for v, c in zip(native_reported, native_t_checks)])
if PACKAGED:
    reported = package["reported_aggregates"]
    assert reported["native_routing"] == native_reported
else:
    reported = {"source": "comments.txt", "source_sha256": hashlib.sha256((ROOT/"comments.txt").read_bytes()).hexdigest(),
        "origin": "Authors' rebuttal; transcribed aggregate results, not rerun in this revision",
        "line_ranges_1_based": {"corruption": [103,108], "severity": [93,97], "fewshot": [246,249], "routing": [174,188]},
        "corruptions": corruption_rows, "severity": severity_rows, "fewshot": fewshot_rows,
        "native_routing": native_reported, "metr_la_six_model_loo": {"without_gwnet": .973, "minimum": .961}}
(GENERATED / "rebuttal_reported.json").write_text(json.dumps(reported, indent=2)+"\n", encoding="utf-8")

if PACKAGED:
    submitted_rows = package["submitted_sensor_city_values"]
else:
    with zipfile.ZipFile(ROOT / SOURCES[0]) as z:
        original = z.read("submission_neurips2026_20260506/sections/4_taxonomy.tex").decode("utf-8")
    patterns = {"dcrnn": r"DCRNN \(fixed diffusion\)", "mtgnn": r"MTGNNLight \(sparse cosine-sim\)",
        "stgcn": r"STGCN \(fixed spectral\)", "gru": r"GRU\s+\(graph-free\)",
        "gwnet": r"GWNet \(dense adaptive adj\)", "agcrn": r"AGCRN", "megacrn": r"MegaCRN", "staeformer": r"STAEformer"}
    submitted_rows = {}
    for m, pattern in patterns.items():
        match = re.search(r"^"+pattern+r"\s*&(.+?)\\\\", original, re.M)
        if match is None:
            raise ValueError("Submitted table row not found: " + m)
        cells = match.group(1).split("&")
        values = [[float(n) for n in re.findall(r"[+-]?\d+(?:\.\d+)?", cell)] for cell in cells]
        submitted_rows[m] = {"sensor": values[1][0], "sensor_std": values[1][1],
                             "city": values[2][0], "city_std": values[2][1]}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
order = ["dcrnn", "mtgnn", "stgcn", "gru", "gwnet", "agcrn", "megacrn", "staeformer"]
labels = ["DCRNN", "MTGNN-Light", "STGCN", "GRU", "GWNet", "AGCRN", "MegaCRN", "STAEformer"]
colors = {"dcrnn": "#3D8F62", "mtgnn": "#76B48B", "stgcn": "#559E72", "gru": "#969AA0",
          "gwnet": "#C95559", "agcrn": "#4E9DBE", "megacrn": "#9274B5", "staeformer": "#42A6A4"}
fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.15), gridspec_kw={"width_ratios": [1.15, 1.05, 1]})
pems = load(SOURCES[3])["per_model"]
for ax, metric, title, ylabel in [(axes[0], "sensor", "Sensor dropout (30%)", r"$\Delta_{\rm sensor}$ (normalized)"),
                                  (axes[1], "city", "Cross-city transfer", "Source-normalized MAE")]:
    pos = np.arange(len(order))
    vals = [submitted_rows[m][metric] for m in order]
    errors = [submitted_rows[m][metric+"_std"] for m in order]
    ax.bar(pos, vals, yerr=errors, width=.66, color=[colors[m] for m in order],
           error_kw={"elinewidth": .7, "capsize": 1.7, "capthick": .7})
    ax.axvline(4.5, color="#888888", linestyle="--", linewidth=.7)
    ax.axhline(0, color="#333333", linewidth=.6)
    ax.set_xticks(pos, labels, rotation=58, ha="right", fontsize=8.2)
    ax.set_title(title, fontsize=10.4, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9.2)
    ax.tick_params(axis="y", labelsize=8.5)
    if metric == "sensor":
        for m, v in pems.items():
            i = order.index(m)
            ax.bar(i+.18, v["delta_sensor_mean"], width=.27, facecolor="white", edgecolor=colors[m],
                   hatch="////", linewidth=.7, yerr=v["delta_sensor_std"],
                   error_kw={"elinewidth": .6, "capsize": 1.2})
        ax.set_ylim(-.082, .17)
        ax.legend(handles=[Patch(facecolor="#777777", label="METR-LA primary"),
                           Patch(facecolor="white", edgecolor="#777777", hatch="////", label="PEMS04 native")],
                  loc="upper left", frameon=False, fontsize=6.9, handlelength=1.2)
    else:
        ax.set_ylim(0, .31)

ax = axes[2]
extended = list(display)
vals = [temporal[m]["mph_mean"] for m in extended]
err = [temporal[m]["mph_std"] for m in extended]
ax.bar(np.arange(3), vals, width=.57, color=[colors[m] for m in extended], yerr=err,
       error_kw={"elinewidth": .8, "capsize": 2})
ax.axhline(0, color="#333333", linewidth=.8)
ax.set_xticks(np.arange(3), [display[m] for m in extended], rotation=35, ha="right", fontsize=8.2)
ax.set_ylim(-.86, 1.02)
ax.set_title("Weekday/weekend (mph)", fontsize=10.4, fontweight="bold")
ax.set_ylabel(r"$\mathrm{MAE}_{\rm weekend}-\mathrm{MAE}_{\rm weekday}$", fontsize=9.2)
ax.tick_params(axis="y", labelsize=8.5)
for i, v in enumerate(vals):
    ax.text(i, v+(.075 if v>=0 else -.09), f"{v:+.2f}", ha="center", va="bottom" if v>=0 else "top", fontsize=9)
fig.tight_layout(w_pad=1.4)
fig.savefig(PAPER / "figures/shift_profiles.pdf", bbox_inches="tight")
fig.savefig(PAPER / "figures/shift_profiles.png", dpi=180, bbox_inches="tight")
plt.close(fig)

primary_y = [submitted_rows[m]["sensor"] for m in models]
primary_rho = correlation(x, primary_y)
output = {"date_local": "2026-11-04", "kind": "Reaggregation and presentation of existing results; no training or inference",
    "source_sha256": package["original_source_sha256"] if PACKAGED else {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
    "temporal": temporal, "routing": {"rho": rho, "leave_one_out": loo, "exact_abs_tail_p": perm_p,
        "primary_mask_sensitivity_rho": primary_rho, "primary_mask_deltas": dict(zip(models, primary_y)),
        "max_mass_population_std": {m: statistics.pstdev(v['max_mass_mean'] for v in routing[m]['per_seed']) for m in models},
        "six_model_t_checks_from_rounded_reported_rho": native_t_checks},
    "mask_aware_r30": mask, "fpfn_penalties": fp_stats, "submitted_sensor_city_values": submitted_rows}
(GENERATED / "metrics.json").write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"generated": str(GENERATED), "rho": rho, "permutation_p": perm_p,
                  "temporal_mph": {m: v["mph_mean"] for m, v in temporal.items()}}))
