"""
src_fig2.py
----------
Build Figure 2 from Rg/types time series, model PDFs, snapshots, and displacement histograms.
"""

import argparse
import csv
import string
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
try:
    from matplotlib.figure import Figure
    import matplotlib.svg as msvg
    SVG_AVAILABLE = True
except ImportError:
    SVG_AVAILABLE = False
from pdf2image import convert_from_path
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Matplotlib rcParams
# ---------------------------------------------------------------------------
# A4 usable width  ≈ 170 mm  ≈ 6.69 in
# A4 usable height ≈ 257 mm  ≈ 10.12 in  (with 2 cm margins top/bottom)

MPL_RC = {
    "font.family":        "serif",
    "font.size":          8,   # 8 × 1.4
    "axes.labelsize":     6,   # 9 × 1.4
    "axes.titlesize":     8,   # 8 × 1.4
    "xtick.labelsize":    8,   # 8 × 1.4
    "ytick.labelsize":    8,   # 8 × 1.4
    "legend.fontsize":    8.8,    # 7 × 1.4
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "lines.linewidth":    1.2,
    "figure.dpi":         500,
    "savefig.dpi":        500,
    "savefig.bbox":       "tight",
}


A4_WIDTH  = 7    
A4_HEIGHT = 6  

# -- CONFIG -------------------------------------------------------------------
LEFT_R2      = "left_r2.dat"
LEFT_TYPES   = "left_types.dat"
LEFT_DUMP    = "left_dump.lammpstrj"


SNAPSHOT_PS      = "PS.png"
SNAPSHOT_NO_PS      = "NO_PS.png"
SNAPSHOT_ZOOM_NO_PS = "zoom_NO_PS.png"
SNAPSHOT_ZOOM_PS = "zoom_PS.png"
FIGURE2A_PDF    = "Figure2_model_a.pdf"
FIGURE2E_PDF    = "Figure2_model_e.pdf"
# Defne polymers (list of (first_id, last_id) tuples).
POLYMERS = list(range(1, 81))

TS_STRIDE    = 50     # time-series downsampling stride
KYMO_STRIDE  = 1     # kymograph downsampling stride
TYPES_STEP   = 1000  # types1.dat is written every this many timesteps
DUMP_STEP    = 10000 # dump is written every this many timesteps
TYPES_PER_DUMP = DUMP_STEP // TYPES_STEP   # = 10

TYPE_COLORS = {
    1: "#2166AC",   # A  — blue
    2: "#F4C300",   # U  — yellow
    3: "#D6001C",   # M  — red
}
TYPE_LABELS = {1: "A", 2: "U", 3: "M"}
TYPE_CMAP   = mcolors.ListedColormap([TYPE_COLORS[k] for k in sorted(TYPE_COLORS)])

SWI6M_COLOR = "#1A9641"   # green
SWI6_COLOR  = '#CC79A7'   # pink
RG_COLOR    = "#777777"   # grey

# ═══════════════════════════════════════════════════════════════════════════
# PBC asymptote: for uniform distribution in [-L/2, L/2]³ the mean ⟨r²⟩ = L²/4
MSD_PBC_LIMIT = (BOX_SIZE / 2) ** 2


# ══════════════════════════════════════════════════════════════════════════
#  TRAJECTORY IO
# ══════════════════════════════════════════════════════════════════════════
def parse_r2(filepath):
    ts, vals = [], []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts.append(int(parts[0]))
            vals.append(float(parts[1]))
    return np.array(ts, dtype=np.int64), np.array(vals, dtype=float)

def _try_load_image(path):
    """Return image array or None if file is missing."""
    try:
        return mpimg.imread(path)
    except FileNotFoundError:
        warnings.warn(f"Snapshot not found: {path} — showing placeholder.")
        return None


def add_inset(ax, img, loc="lower right", size="45%", pad=0.2, edgecolor="white"):
    if img is None:
        return None

    fig = ax.figure

    ax_in = inset_axes(
        ax,
        width=size,
        height=size,
        loc=loc,
        borderpad=pad,
        axes_kwargs={"zorder": 10}
    )

    ax_in.imshow(img, aspect="equal", zorder=10)
    ax_in.set_xticks([])
    ax_in.set_yticks([])

    for spine in ax_in.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor(edgecolor)

    ax_in.patch.set_alpha(0.00)
    ax_in.set_zorder(10)
    fig.add_axes(ax_in)
    return ax_in


def crop_img(img, f=0.05):
    h, w = img.shape[:2]
    dy, dx = int(h * f), int(w * f)
    return img[dy:h-dy, dx:w-dx]


def plot_all(timesteps, ids_list, arrays,
             rg_ts=None,    rg_vals=None,
             types_df=None, types_ts=None,
             timeline_events=None,
             SNAPSHOT_PS=SNAPSHOT_PS,
             SNAPSHOT_NO_PS=SNAPSHOT_NO_PS,
             SNAPSHOT_ZOOM_PS=SNAPSHOT_ZOOM_PS,
             SNAPSHOT_ZOOM_NO_PS=SNAPSHOT_ZOOM_NO_PS,
             outfile=None):

    with mpl.rc_context(MPL_RC):
        _plot_inner(timesteps, ids_list, arrays,
                    rg_ts=rg_ts,         rg_vals=rg_vals,
                    types_df=types_df,   types_ts=types_ts,
                    SNAPSHOT_PS=SNAPSHOT_PS,
                    SNAPSHOT_NO_PS=SNAPSHOT_NO_PS,
                    SNAPSHOT_ZOOM_PS=SNAPSHOT_ZOOM_PS,
                    SNAPSHOT_ZOOM_NO_PS=SNAPSHOT_ZOOM_NO_PS,
                    outfile=outfile)

def _plot_inner(timesteps, ids_list, arrays,
                rg_ts, rg_vals, types_df, types_ts, SNAPSHOT_PS, SNAPSHOT_NO_PS, SNAPSHOT_ZOOM_PS, SNAPSHOT_ZOOM_NO_PS, outfile):

    ts      = np.array(timesteps)
    ts_ts   = ts[::TS_STRIDE]
    ts_kymo = ts[::KYMO_STRIDE]

    arrays_ts   = [arr[::TS_STRIDE]   for arr in arrays]
    arrays_kymo = [arr[::KYMO_STRIDE] for arr in arrays]

    counts_list = []

    bounds = [0.5, 1.5, 2.5, 3.5]
    norm   = mcolors.BoundaryNorm(bounds, TYPE_CMAP.N)

    # ==========================================================
    # AXES
    # ==========================================================
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
    gs = gridspec.GridSpec(
        4, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=[2.3, 1.2, 1.2, 1.2],
        hspace=0.035,
        wspace=0.225
    )

    # LEFT COLUMN — shared x-axis for b, c, d
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_b)
    ax_d = fig.add_subplot(gs[3, 0], sharex=ax_b)

    # RIGHT COLUMN
    ax_e = fig.add_subplot(gs[0, 1])

    # snapshots: split right column rows 1-2 into 1 row × 2 cols
    subgs_snap = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1:3, 1], hspace=0.125
    )
    ax_f_nops = fig.add_subplot(subgs_snap[0, 0])  # NO_PS left
    ax_f_ps   = fig.add_subplot(subgs_snap[0, 1])  # PS right

    ax_g = fig.add_subplot(gs[3, 1])

    # ==========================================================
    # (a) (e) MODELS — no box
    # ==========================================================
    ax_a.imshow(crop_img(np.array(_load_pdf_as_image(FIGURE2A_PDF)), f=0.0125))
    ax_a.set_axis_off()
    _label_panel(ax_a, 0)

    ax_e.imshow(crop_img(np.array(_load_pdf_as_image(FIGURE2E_PDF)), f=0.02))
    ax_e.set_axis_off()
    _label_panel(ax_e, 4, 0,0.84)

    # remove spines/border around the pdf axes
    for ax_pdf in [ax_a, ax_e]:
        for spine in ax_pdf.spines.values():
            spine.set_visible(False)

    # ==========================================================
    # (f) SNAPSHOTS — NO_PS left (zoom top-right), PS right (zoom top-left)
    # ==========================================================
    def show_snapshot(ax, path, inset_path, inset_loc):
        img = _try_load_image(path)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)   # no box
        if img is not None:
            img = crop_img(img, 0.103)
            ax.imshow(img, aspect="equal")
            if inset_path is not None:
                inset_img = _try_load_image(inset_path)
                color = COL_NPS if "NPS" in inset_loc or inset_loc == "upper right" else COL_PS
                add_inset(ax, inset_img, loc=inset_loc, edgecolor=color)
        else:
            ax.set_facecolor("#e8e8e8")
            ax.text(0.5, 0.5, f"{path}\n(not found)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888", style="italic")

    show_snapshot(ax_f_nops, SNAPSHOT_NO_PS, SNAPSHOT_ZOOM_NO_PS, "upper right")
    show_snapshot(ax_f_ps,   SNAPSHOT_PS,    SNAPSHOT_ZOOM_PS,    "upper left")

    _label_panel(ax_f_nops, 5)

    # ==========================================================
    # (b) COUNT TYPES — shared x
    # ==========================================================
    _label_panel(ax_b, 1)

    # shift x to start at 0
    x_offset = types_ts[0]
    x_b = types_ts - x_offset

    for i, t in enumerate(['A', 'U', 'M'], start=1):
        ax_b.plot(x_b[::200], types_df[t].iloc[::200],
                  color=TYPE_COLORS[i],
                  label=TYPE_LABELS[i],
                  linewidth=1.0)

    ax_b.set_ylabel("Count\nnucleosomal\ntype", fontsize=7, rotation=90)
    ax_b.set_ylim(0, 80)
    ax_b.set_yticks([0, 40, 80])
    ax_b.set_yticklabels([0, 40, 80],fontsize=6.5)
    ax_b.grid(alpha=0.20, linewidth=0.4)
    # ax_b.legend(frameon=False, fontsize=9)
    ax_b.tick_params(axis='both', which='major',
                    direction='in', bottom=True, left=True, top=False, right=False,
                    labelbottom=False, labelleft=True)
    ax_b.xaxis.set_tick_params(labelbottom=False)
    ax_b.minorticks_off()
    plt.setp(ax_b.get_xticklabels(), visible=False)

    # ==========================================================
    # (c) SWI6* — shared x
    # ==========================================================
    _label_panel(ax_c, 2)
    x_c = types_ts - x_offset

    if types_df is not None:
        ax_c.plot(x_c[::50], types_df["Swi6M"].values[::50],
                  color=SWI6M_COLOR, lw=1.2, label="Swi6*")
        # ax_c.plot(x_c[::50], types_df["Swi6"].values[::50],
        #           color=SWI6_COLOR,  lw=1.0, label="Swi6")
        ax_c.set_ylabel("Count Swi6*", fontsize=7
                        )
        ax_c.set_yticks([20,40])
        ax_c.set_yticklabels([20,40],fontsize=6.5)
        ax_c.grid(alpha=0.2)
                # --- First group: Types ---
        type_handles = [
            Patch(color=TYPE_COLORS[k], label=TYPE_LABELS[k])
            for k in sorted(TYPE_COLORS)
        ]

        # --- Second group: SWI6 / SWI6M ---
        other_handles = [
            Patch(color=SWI6M_COLOR, label="Swi6*"),
        ]

        # Combine all handles
        handles = type_handles + other_handles

        # Draw legend (2 columns)
        ax_c.legend(handles=handles, ncol=2, fontsize=6)
        # make sure ticks are visible
        ax_c.minorticks_off()
        ax_c.tick_params(axis='both', which='major',
                        direction='in', bottom=True, left=True, top=False, right=False,
                        labelbottom=False, labelleft=True)
        ax_c.xaxis.set_tick_params(labelbottom=False)

    plt.setp(ax_c.get_xticklabels(), visible=False)

    # ==========================================================
    # (d) Rg — shared x, x-axis shown here
    # ==========================================================
    _label_panel(ax_d, 3)
    x_d = rg_ts - x_offset

    if rg_ts is not None:
        ax_d.plot(x_d[::100], rg_vals[::100],
                  color=RG_COLOR, lw=1.2)

    ax_d.set_yticks([2, 5])
    ax_d.set_yticklabels([2, 5],fontsize=6.5)
    ax_d.set_ylabel(r"$R_g^2(\sigma^2)$", fontsize=7)
    ax_d.grid(alpha=0.2)

    
    x_span = x_b[-1] - x_b[0]
    tick_vals = [0, x_span / 2, x_span]
    ax_d.set_xticks(tick_vals)
    ax_d.set_xticklabels([rf'$0$',rf'$1.5$',rf'$3 \times 10^4$'], fontsize=6.5)
    ax_d.set_xlabel(r"Time ($\tau_{\mathrm{LJ}}$)", fontsize=7)
    # make sure ticks are visible
    ax_d.tick_params(axis='both', which='major',
                     direction='in', bottom=True, left=True, top=False, right=False,
                     labelbottom=True, labelleft=True)
    ax_d.xaxis.set_tick_params(labelbottom=True)
    ax_d.minorticks_off()

    # ==========================================================
    # (g) DISPLACEMENTS
    # ==========================================================
    ax_g = plot_from_cache(ax_g)
    _label_panel(ax_g, 6, -0.16)

    # make sure ticks are visible
    ax_g.tick_params(axis='both', which='major',
                     direction='in', bottom=True, left=True, top=False, right=False,
                     labelbottom=True, labelleft=True)
    ax_g.xaxis.set_tick_params(labelbottom=True)
    ax_g.grid(alpha=0.2)
    ax_g.minorticks_off()

    # ==========================================================
    # SAVE
    # ==========================================================
    plt.savefig("figure2_0605_rayleigh.pdf", dpi=500)
    print("Saved")
# ══════════════════════════════════════════════════════════════════════════
#  DISPLACEMENT DISTRIBUTION  –  BIC-penalised Rayleigh mixture
# ══════════════════════════════════════════════════════════════════════════

def plot_from_cache(ax_g, cache_prefix="disp"):

    bins = np.linspace(0, 3, 50)

    for system, color in [("NPS", COL_NPS), ("PS", COL_PS)]:

        data = np.load(f"{cache_prefix}_{system}_raw.npy")

        ax_g.hist(
            data,
            bins=bins,
            density=True,
            alpha=0.5,
            color=color,
            label="Early" if system == "NPS" else "Late"
        )
        ax_g.set_xticks([0,1,2,3])
        ax_g.set_xticklabels([0,1,2,3],fontsize=6.5)
        ax_g.set_yticks([0,0.75,1.5])
        ax_g.set_yticklabels([0,0.75,1.5],fontsize=6.5)
        ax_g.set_ylabel('Density',rotation = 90, fontsize=7)
        ax_g.set_xlabel(r'$|\Delta r|(\sigma)$', fontsize=7)
        ax_g.legend()
        
    return ax_g

# ══════════════════════════════════════════════════════════════════════════
# VI.  FIGURE 
# ══════════════════════════════════════════════════════════════════════════
# -- PRX PANEL LABEL HELPER ---------------------------------------------------

def _label_panel(ax, idx, x=-0.25, y=0.88): #x=-0.12 before
    """ panel label slightly outside top-left of axes."""
    label = f"({string.ascii_lowercase[idx]})"
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        clip_on=False,
        zorder=10,
    )

def _load_pdf_as_image(pdf_path,dpi=500):
    # Convert PDF → list of PIL images
    pages = convert_from_path(pdf_path, dpi=dpi)

    # Take first page (or loop if multiple)
    img = pages[0]
    return img





# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    # # ── SPT trajectories ──────────────────────────────────────────
    # all_atoms = sorted(set(SPT_ATOMS))

    # print(f"\nLoading {FILE_NPS} …")
    # traj_n_all, times_n_all = parse_lammpstrj(FILE_NPS, all_atoms)

    # print(f"Loading {FILE_PS} …")
    # traj_p_all, times_p_all = parse_lammpstrj(FILE_PS, all_atoms)

    # if traj_n_all.shape[0] == 0 or traj_p_all.shape[0] == 0:
    #     print("[ERROR] One or both trajectories empty."); return

    # id_to_col = {a: i for i, a in enumerate(sorted(all_atoms))}
    # spt_cols  = [id_to_col[a] for a in sorted(SPT_ATOMS)]

    # traj_n_cut, _ = cut_trajectory(traj_n_all, times_n_all, T_EQ,  T_MID)
    # traj_p_cut, _ = cut_trajectory(traj_p_all, times_p_all, T_MID, T_END)

    # print(f"\n  NPS window: {len(traj_n_cut)} frames  [t={T_EQ:.0e} … {T_MID:.0e}]")
    # print(f"  PS  window: {len(traj_p_cut)} frames  [t={T_MID:.0e} … {T_END:.0e}]")

    # traj_n = traj_n_cut[:, spt_cols, :]
    # traj_p = traj_p_cut[:, spt_cols, :]

    # compute_and_save_displacement_analysis(traj_n, traj_p)

    # ── Left-column data ──────────────────────────────────────────
    # 1. Rg — has explicit timesteps, use as ground-truth timeline
    rg_ts_full, rg_vals_full = parse_r2(LEFT_R2)
    rg_dt = int(np.median(np.diff(rg_ts_full)))
    print(f"\nReading {LEFT_R2} -> {len(rg_ts_full)} points, dt={rg_dt}")

    # 2. Types — no timestep column; reconstruct from rg span
    #    parse_types needs a dummy timestep array of the right length;
    #    we pass rg_ts_full so its length is used for alignment
    types_df_full_raw = pd.read_csv(LEFT_TYPES, comment="#",
                                    names=["A", "U", "M", "Swi6", "Swi6M"])
    types_df_full_raw = types_df_full_raw.iloc[1:].reset_index(drop=True)

    n_types = len(types_df_full_raw)
    rg_tmin, rg_tmax = rg_ts_full[0], rg_ts_full[-1]
    types_dt = (rg_tmax - rg_tmin) // (n_types - 1)
    types_ts_full = np.arange(n_types, dtype=np.int64) * types_dt + rg_tmin
    print(f"Reading {LEFT_TYPES} -> {n_types} windows, "
          f"inferred dt={types_dt} "
          f"(span {rg_tmin}–{rg_tmax})")
    TMIN = 30000000  
    TMAX = 60000000
    # 3. Restrict everything to [TMIN, TMAX]
    mask_rg    = (rg_ts_full    >= TMIN) & (rg_ts_full    <= TMAX)
    mask_types = (types_ts_full >= TMIN) & (types_ts_full <= TMAX)

    rg_ts    = rg_ts_full[mask_rg]
    rg_vals  = rg_vals_full[mask_rg]
    types_ts = types_ts_full[mask_types]
    types_df = types_df_full_raw.iloc[mask_types].reset_index(drop=True)

    print(f"  Restricted to [{TMIN}, {TMAX}]: "
          f"{len(rg_ts)} rg points, {len(types_ts)} types windows")

    # timesteps & dummy arrays for _plot_inner (types_df drives the plots)
    timesteps = rg_ts.copy()
    ids_list  = POLYMERS
    arrays    = []   # no kymograph data needed
    plot_all(timesteps, ids_list, arrays,
             rg_ts=rg_ts,        rg_vals=rg_vals,
             types_df=types_df,  types_ts=types_ts,
             timeline_events=None,
             SNAPSHOT_PS=SNAPSHOT_PS,
             SNAPSHOT_NO_PS=SNAPSHOT_NO_PS,
             SNAPSHOT_ZOOM_PS=SNAPSHOT_ZOOM_PS,
             SNAPSHOT_ZOOM_NO_PS=SNAPSHOT_ZOOM_NO_PS,
             outfile=None)


if __name__ == "__main__":
    main()