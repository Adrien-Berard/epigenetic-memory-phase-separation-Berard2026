"""
src_figS3.py
----------
Supplementary Figure S3: small time-series panels and RGB scan results from all_results.csv.
"""

# Local dataset root (Zenodo download); override with BERARD_DATA_ROOT.
DATA_ROOT = Path(os.environ.get("BERARD_DATA_ROOT", "./data"))

import os
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import numpy as np
import matplotlib as mpl

mpl.rcParams["font.family"] = "serif"
mpl.rcParams['font.size'] = 9
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7

# ── paths ────────────────────────────────────────────────────────────────────
base_root_dir = DATA_ROOT / "SPombe_MatRegion_Model" / "Finep2Swi6Scan26_03_26"
result_csv    = os.path.join(base_root_dir, "all_results.csv")

result_df = pd.read_csv(result_csv)
result_df = result_df[result_df["noise"] == 500]
lookup_df = result_df.set_index(["p2", "swi6"])

# ── parameter grids ──────────────────────────────────────────────────────────
p2_values         = sorted([0.00035, 0.00026, 0.00024, 0.00042,
                             0.00038, 0.00032, 0.00029])
swi6_values       = sorted(np.linspace(160, 490, 7, dtype=int).tolist())
variants          = ['FullA', 'FullM']
noise             = 500

NR = len(swi6_values)   # rows    per panel
NC = len(p2_values)     # columns per panel

pattern = re.compile(
    r"sim_p2_(?P<p2>[0-9\.eE\-\+]+)_swi6_(?P<swi6>[0-9]+)"
)

# ── colour helpers ───────────────────────────────────────────────────────────
def compute_rgb(a, m):
    green = np.array([0, 1, 0])
    blue  = np.array([00, 0, 1])
    red  = np.array([1, 0, 0])
    rgb = (a * m * green + a * (1-m) * blue +
           (1-a) * m * red)
    return np.clip(rgb, 0, 1)

# ── layout constants (in figure-fraction units) ──────────────────────────────
LEFT   = 0.13   # left edge of the subplot grid
RIGHT  = 0.98
BOTTOM = 0.10   # bottom edge of LOWER panel
TOP    = 0.97   # top    edge of UPPER panel
VGAP   = 0.07   # gap between the two panels
PANEL_H = (TOP - BOTTOM - VGAP) / 2   # height of each panel in fig-fraction

# bottom edges of the two panels (panel 0 = top = FullA)
panel_bottom = [BOTTOM + PANEL_H + VGAP, BOTTOM]

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 11))

for idx, variant in enumerate(variants):

    pb = panel_bottom[idx]   # bottom of this panel in fig coords

    # ── inner GridSpec for this panel ────────────────────────────────────────
    gs = gridspec.GridSpec(
        NR, NC,
        figure=fig,
        left=LEFT, right=RIGHT,
        bottom=pb, top=pb + PANEL_H,
        wspace=0.12, hspace=0.12,
    )
    axes = np.array([[fig.add_subplot(gs[i, j])
                      for j in range(NC)]
                     for i in range(NR)])

    # ── load simulation data ─────────────────────────────────────────────────
    sim = {}
    variant_dir = os.path.join(base_root_dir, f"Noise{noise}", variant)
    if variant == 'FullM':
        print(variant_dir)
    if os.path.exists(variant_dir):
        for folder in os.listdir(variant_dir):
            m = pattern.match(folder)
            if m:
                p2   = float(m.group("p2"))
                swi6 = float(m.group("swi6"))
                p2 = float(m.group("p2"))
                fp   = os.path.join(variant_dir, folder, "types1.dat")
                if os.path.exists(fp):
                    sim[(p2, swi6)] = pd.read_csv(
                        fp, comment='#', names=['A','U','M','Swi6','Swi6M'])

    # ── fill subplots ─────────────────────────────────────────────────────────
    for i, swi6 in enumerate(swi6_values[::-1]):
        for j, p2 in enumerate(p2_values):
            ax  = axes[i, j]
            key = (p2, float(swi6))

            if key in sim:
                df = sim[key]
                try:
                    fhA, fhM = lookup_df.loc[(p2, swi6), ["all_tauA_startA","all_tauM_startM"]]
                except KeyError:
                    fhA, fhM = np.nan, np.nan

                rgb = compute_rgb(fhA, fhM)
                ax.imshow(np.ones((2,2,3)) * rgb,
                          extent=[0,20000,0,80], aspect='auto',
                          alpha=0.25, zorder=0)
                ax.plot(df['U'].iloc[::250], color='gold',  lw=0.7, zorder=2)
                ax.plot(df['A'].iloc[::250], color='blue',  lw=0.7, zorder=2)
                ax.plot(df['M'].iloc[::250], color='red',   lw=0.7, zorder=2)

            ax.set_xticks([0, 10000])
            ax.set_yticks([0, 80])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, lw=0.4)

    # ── panel label (a) / (b) ─────────────────────────────────────────────────
    label = '(a)' if idx == 0 else '(b)'
    fig.text(LEFT - 0.16, pb + PANEL_H + 0.01, label,
             fontsize=12, va='top')

    # ════════════════════════════════════════════════════════════════════════
    # OUTER AXIS – Swi6 row labels  (one invisible axis spanning the panel)
    # Position: [left, bottom, width, height]  all in figure fraction
    # ════════════════════════════════════════════════════════════════════════
    ax_swi6 = fig.add_axes([LEFT - 0.09, pb, 0.001, PANEL_H])
    ax_swi6.set_ylim(-0.5, NR - 0.5)
    ax_swi6.set_yticks(range(NR))
    ax_swi6.set_yticklabels(
        [str(v) for v in swi6_values],   # top row = highest swi6
        fontsize=8
    )
    ax_swi6.tick_params(axis='y', length=0, pad=3)
    ax_swi6.set_xticks([])
    for sp in ax_swi6.spines.values():
        sp.set_visible(False)

    # ── Swi6 axis label (only once, left of the axis) ────────────────────────
    ax_swi6.set_ylabel("Total Swi6 count", fontsize=11, labelpad=28)
    plt.annotate(
        '', 
        xy=(LEFT - 0.08, 0.5),        # arrow head (top)
        xytext=(LEFT - 0.08, 0.13),   # tail (bottom)
        xycoords='figure fraction',
        arrowprops=dict(arrowstyle='->', lw=0.8)
    )
    
    plt.annotate(
            '', 
            xy=(LEFT - 0.08, 0.97),        # arrow head (top)
            xytext=(LEFT - 0.08, 0.6),   # tail (bottom)
            xycoords='figure fraction',
            arrowprops=dict(arrowstyle='->', lw=0.8)
        )
    # ── nucleosomal-count y-tick labels on leftmost column ───────────────────
    for i in range(NR):
        axes[i, 0].set_yticks([0, 40])
        axes[i, 0].set_yticklabels(['0','40'], fontsize=8)

# ════════════════════════════════════════════════════════════════════════════
# OUTER AXIS – k2 column labels  (one invisible axis below both panels)
# ════════════════════════════════════════════════════════════════════════════
ax_k2 = fig.add_axes([LEFT, BOTTOM - 0.07, RIGHT - LEFT, 0.001])
ax_k2.set_xlim(-0.5, NC - 0.5)
ax_k2.set_xticks(range(NC))
ax_k2.set_xticklabels(
    [rf"$\mathregular{{{p2/1e-3:.2e}}}$" for p2 in p2_values],
    fontsize=9
)
ax_k2.tick_params(axis='x', length=0, pad=3)
ax_k2.set_yticks([])
for sp in ax_k2.spines.values():
    sp.set_visible(False)
ax_k2.set_xlabel(r"$k_2\ (\tau_{\mathrm{LJ}}^{-1})$", fontsize=11, labelpad=12)
plt.annotate(
    '',
    xy=(1.01, BOTTOM - 0.064),     # arrow head (right)
    xytext=(0.2, BOTTOM - 0.064), # tail (left)
    xycoords='figure fraction',
    arrowprops=dict(arrowstyle='->', lw=0.8)
)

# ── shared time label centred below the bottom panel ─────────────────────────
# show x-tick numbers on the bottom row of the LOWER panel only
# bottom_axes = np.array([[fig.add_subplot(gs[i, j])
#                          for j in range(NC)]
#                         for i in range(NR)])  # reuse last gs (FullM)
for j in range(NC):
    axes[NR-1, j].set_xticklabels(['0', r'$10^4$'], fontsize=8)

fig.text(0.56, BOTTOM - 0.03,
         r"Time ($\tau_{\mathrm{LJ}}$)",
         ha='center', va='top', fontsize=11)

plt.savefig("SuppTimeseriesScan_combined_0605_timefrac.pdf", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved SuppTimeseriesScan_combined.pdf")