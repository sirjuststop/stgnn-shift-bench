"""Generate three-shift failure profile grouped bar chart for v7 paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
# Canonical bank (from Table 1 / main taxonomy)
# Extended models (from overnight eval)
models = [
    'DCRNN', 'MTGNNLight', 'STGCN', 'GRU', 'GWNet',  # canonical
    'AGCRN', 'MegaCRN', 'STAEformer'  # extended
]

# Sensor dropout delta (normalized, 30%; primary-mask protocol)
sensor = [0.020, 0.051, 0.078, 0.025, 0.088,
          0.019, -0.048, 0.015]
sensor_std = [0.002, 0.012, 0.007, 0.012, 0.017,
              0.001, 0.027, 0.001]

# Cross-city source-stat normalized MAE
city = [0.180, 0.244, 0.249, 0.278, 0.268,
        0.206, 0.163, 0.223]
city_std = [0.001, 0.004, 0.008, 0.009, 0.006,
            0.009, 0.003, 0.008]

# Temporal delta (normalized)
temporal = [0.087, 0.075, 0.093, 0.093, 0.094,
            0.061, 0.734, 0.750]
temporal_std = [0.004, 0.007, 0.003, 0.003, 0.009,
                0.008, 0.019, 0.003]

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# Colors: canonical = blues/grays, extended = warm colors
colors_canonical = ['#4472C4', '#6B9BD2', '#8DB4E0', '#A9A9A9', '#E74C3C']
colors_extended = ['#F39C12', '#E67E22', '#9B59B6']
colors = colors_canonical + colors_extended

x = np.arange(len(models))
bar_width = 0.65

# Panel 1: Sensor dropout
ax1 = axes[0]
bars1 = ax1.bar(x, sensor, bar_width, color=colors, edgecolor='white', linewidth=0.5)
ax1.errorbar(x, sensor, yerr=sensor_std, fmt='none', ecolor='black', capsize=2, linewidth=0.8)
ax1.set_ylabel('$\\Delta_{\\mathrm{sensor}}$ (norm)', fontsize=11)
ax1.set_title('Sensor Dropout (30%)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
# Add vertical separator between canonical and extended
ax1.axvline(x=4.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)

# Panel 2: Cross-city
ax2 = axes[1]
bars2 = ax2.bar(x, city, bar_width, color=colors, edgecolor='white', linewidth=0.5)
ax2.errorbar(x, city, yerr=city_std, fmt='none', ecolor='black', capsize=2, linewidth=0.8)
ax2.set_ylabel('Cross-city MAE (src-stat, norm)', fontsize=11)
ax2.set_title('Cross-City Transfer', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
ax2.axvline(x=4.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)

# Panel 3: Temporal
ax3 = axes[2]
bars3 = ax3.bar(x, temporal, bar_width, color=colors, edgecolor='white', linewidth=0.5)
ax3.errorbar(x, temporal, yerr=temporal_std, fmt='none', ecolor='black', capsize=2, linewidth=0.8)
ax3.set_ylabel('$\\Delta_{\\mathrm{temporal}}$ (norm)', fontsize=11)
ax3.set_title('Temporal Subgroup Shift', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
ax3.axvline(x=4.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)

# Add "Canonical" and "Extended" labels
for ax in axes:
    ylim = ax.get_ylim()
    y_pos = ylim[1] * 0.92
    ax.text(2, y_pos, 'Canonical', ha='center', fontsize=8, fontstyle='italic', color='#4472C4')
    ax.text(6, y_pos, 'Extended', ha='center', fontsize=8, fontstyle='italic', color='#E67E22')

plt.tight_layout()
plt.savefig('/root/autodl-tmp/xile/benchmark_first_graph_gen/paper/figures/shift_profiles.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/root/autodl-tmp/xile/benchmark_first_graph_gen/paper/figures/shift_profiles.png',
            dpi=300, bbox_inches='tight')
print("Figure saved to paper/figures/shift_profiles.pdf and .png")
