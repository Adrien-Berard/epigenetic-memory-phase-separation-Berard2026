"""
src_figS2.py
----------
Build supplementary Figure S2 from LAMMPS polymer dump and Rg/types.
"""

import argparse
import csv
import string
import warnings

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

# ---------------------------------------------------------------------------
# Matplotlib rcParams
# ---------------------------------------------------------------------------
# A4 usable width  ≈ 170 mm  ≈ 6.69 in
# A4 usable height ≈ 257 mm  ≈ 10.12 in  (with 2 cm margins top/bottom)

MPL_RC = {
    "font.family":        "serif",
    "font.size":          9,   # 8 × 1.4
    "axes.labelsize":     9,   # 9 × 1.4
    "axes.titlesize":     9,   # 8 × 1.4
    "xtick.labelsize":    8,   # 8 × 1.4
    "ytick.labelsize":    8,   # 8 × 1.4
    "legend.fontsize":    8,    # 7 × 1.4
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

# Single-column A4 dimensions in inches
# One-column journal format ≈ 86 mm wide; full A4 page height
A4_WIDTH  = 6.5    # 86 mm — single column
A4_HEIGHT = 8  # 297 mm

# -- CONFIG -------------------------------------------------------------------
DUMP_FILE     = "id_and_type.dat"
R2_FILE       = "r2.dat"
TIMELINE_FILE = "replication_timeline.dat"
TYPES_FILE    = "types1.dat"


# Define polymers (list of (first_id, last_id) tuples).
POLYMERS = [
    (1, 80)
]

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

PHASE_STYLES = {
    "G1":      {"color": "#74C476", "alpha": 0.35, "label": "G1"},
    "S":       {"color": "#9E9AC8", "alpha": 0.35, "label": "S-phase"},
    "G2":      {"color": "#4292C6", "alpha": 0.35, "label": "G2"},
    "Mitosis": {"color": "#EF6548", "alpha": 0.55, "label": "Mitosis"},
}


# -- PRX PANEL LABEL HELPER ---------------------------------------------------

def _label_panel(ax, idx, x=-0.24, y=0.88): #x=-0.12 before
    """ panel label slightly outside top-left of axes."""
    label = f"({string.ascii_lowercase[idx]})"
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        clip_on=False,
        zorder=10,
    )

# -- PARSERS ------------------------------------------------------------------

def parse_dump(filepath):
    timesteps, frames = [], []
    current_ts = None
    n_atoms = 0
    with open(filepath) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            current_ts = int(lines[i + 1].strip())
            i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("ITEM: ATOMS"):
            header   = line.split()[2:]
            col_id   = header.index("id")
            col_type = header.index("type")
            frame = {}
            for _ in range(n_atoms):
                i += 1
                parts = lines[i].strip().split()
                frame[int(parts[col_id])] = int(parts[col_type])
            timesteps.append(current_ts)
            frames.append(frame)
            i += 1
        else:
            i += 1
    return timesteps, frames


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


def parse_types(filepath, dump_timesteps):
    df = pd.read_csv(filepath, comment="#",
                     names=["A", "U", "M", "Swi6", "Swi6M"])
    df = df.iloc[1:].reset_index(drop=True)
    w = TYPES_PER_DUMP
    swi6m_max  = df["Swi6M"].rolling(w, min_periods=w).max()
    other_mean = df[["A", "U", "M", "Swi6M", "Swi6"]].rolling(w, min_periods=w).mean()
    df_roll = other_mean.copy()
    df_roll["Swi6M_max"] = swi6m_max
    keep   = np.arange(w - 1, len(df), w)
    df_out = df_roll.iloc[keep].reset_index(drop=True)
    n_dump = len(dump_timesteps)
    n_out  = len(df_out)
    n_min  = min(n_dump, n_out)
    n_max  = max(n_dump, n_out)
    round_ratio = round(n_max / n_min)
    if n_out != n_dump:
        print(f"  [info] types1.dat gives {n_out} windows, dump has {n_dump} frames "
              f"— using every {round_ratio} points")
    df_out = df_out.iloc[::round_ratio].reset_index(drop=True)
    ts_out = np.array(dump_timesteps, dtype=np.int64)
    return ts_out, df_out[["A", "U", "M", "Swi6", "Swi6M"]]


def parse_timeline(filepath):
    events = []
    with open(filepath, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            events.append({
                "step":  int(row["step"]),
                "event": row["event"].strip(),
                "cycle": int(row["cycle"]),
            })
    events.sort(key=lambda e: e["step"])
    return events


# -- PHASE-BAND HELPERS -------------------------------------------------------

def _find_next(events, from_idx, target_event, cycle):
    for j in range(from_idx + 1, len(events)):
        if events[j]["event"] == target_event and events[j]["cycle"] == cycle:
            return events[j]["step"]
    return None


def _find_next_event_step(events, from_idx, target_event):
    for j in range(from_idx + 1, len(events)):
        if events[j]["event"] == target_event:
            return events[j]["step"]
    return None


def build_arrays(frames, polymers):
    ids_list = [list(range(p[0], p[1] + 1)) for p in polymers]
    n = len(frames)
    arrays = [np.zeros((n, len(ids)), dtype=np.int32) for ids in ids_list]
    for t, frame in enumerate(frames):
        for p_idx, ids in enumerate(ids_list):
            for j, aid in enumerate(ids):
                arrays[p_idx][t, j] = frame.get(aid, 0)
    return ids_list, arrays


def compute_counts(arr):
    return {t: np.sum(arr == t, axis=1) for t in [1, 2, 3]}



# -- PLOTTING -----------------------------------------------------------------

def plot_all(timesteps, ids_list, arrays,
             rg_ts=None,    rg_vals=None,
             types_df=None, types_ts=None,
             outfile=None):

    with mpl.rc_context(MPL_RC):
        _plot_inner(timesteps, ids_list, arrays,
                    rg_ts=rg_ts,         rg_vals=rg_vals,
                    types_df=types_df,   types_ts=types_ts,
                    outfile=outfile)


def _plot_inner(timesteps, ids_list, arrays,
                rg_ts, rg_vals, types_df, types_ts, outfile):

    ts      = np.array(timesteps)
    ts_ts   = ts[::TS_STRIDE]
    ts_kymo = ts[::KYMO_STRIDE]

    arrays_ts   = [arr[::TS_STRIDE]   for arr in arrays]
    arrays_kymo = [arr[::KYMO_STRIDE] for arr in arrays]
    counts_list = [compute_counts(arr) for arr in arrays_ts]

    bounds = [0.5, 1.5, 2.5, 3.5]
    norm   = mcolors.BoundaryNorm(bounds, TYPE_CMAP.N)
    n_polymers = len(arrays)

    all_ts = list(ts)
    if rg_ts is not None:  all_ts += list(rg_ts)


    # ------------------------------------------------------------------
    # Layout:
    #   row 0   : kymograph
    #   row 1…  : polymer type counts
    #   row …   : Swi6M
    #   row …   : Rg
    #
    # All data rows (rows 2+) share the same height ratio so spacing
    # between (b)↔(c) and (c)↔(d) etc. matches the time-series panels.
    # ------------------------------------------------------------------

    # Data-row definitions  (tag, height_ratio)
    data_rows = []
    data_rows.append(("kymo",   1.0))
    for i in range(n_polymers):
        data_rows.append((f"polymer_{i}", 1.0))
    if types_df is not None:
        data_rows.append(("swi6m", 1.0))
    if rg_ts is not None:
        data_rows.append(("rg", 1.0))

    # Height ratios for GridSpec rows
    # We want the svg row, snapshot row, and data rows to all feel balanced.
    # Use the same unit height for snapshot and data rows so spacing is uniform.
    DATA_H   = 1.0   # unit height for each data panel


    height_ratios = [DATA_H] * len(data_rows)

    # Fit everything onto A4 with sensible margins
    # left_margin accounts for y-axis labels (≈12 mm) + panel label overhang
    fig_width  = A4_WIDTH
    # Scale figure height to fill A4 while preserving relative proportions
    content_h  = sum(height_ratios)
    # target usable height ≈ 257 mm → 10.12 in; add top+bottom margin
    margin_v   = 0.6   # inches top + bottom
    fig_height = max(A4_HEIGHT, content_h * 1.05 + margin_v)

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(fig_width, fig_height))


    
    # ============================================================
    # OUTER GRID
    # ============================================================



    gs_data = GridSpec(
        len(data_rows),
        1,
        hspace=0.15,
        top=0.97,
        bottom=0.05,
    )

    data_axs = {}
    ref_ax = None

    for row_idx, (tag, _) in enumerate(data_rows):
        ax = fig.add_subplot(gs_data[row_idx], sharex=ref_ax)

        # shrink horizontally
        pos = ax.get_position()
        ax.set_position([
            0.22,          # larger left margin
            pos.y0,
            0.75,          # narrower width
            pos.height
        ])

        if ref_ax is None:
            ref_ax = ax

        data_axs[tag] = ax
        

    # Hide x-tick labels on all data panels except the last
    last_data_tag = data_rows[-1][0]
    for tag, _ in data_rows:
        ax = data_axs[tag]
        if tag != last_data_tag:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
            ax.set_xlabel(r"Time ($\tau_{\mathrm{LJ}}$)", fontsize=8)
            ax.set_xticks([0,1e7,2e7])
            ax.set_xticklabels([rf'$0$',rf'$1$',rf'$2 \times 10^5$'], fontsize=8)

    # Panel label counter: svg=(a)=0, snap=(b)=1, then 2,3,…
    panel_idx = 0

    # ---- Kymograph --------------------------------------------------
    ax_kymo  = data_axs["kymo"]
    y_offset = 0
    for i, (arr, ids) in enumerate(zip(arrays_kymo, ids_list)):
        n_beads = arr.shape[1]
        ax_kymo.imshow(
            arr.T,
            aspect="auto",
            origin="upper",
            cmap=TYPE_CMAP,
            norm=norm,
            interpolation="nearest",
            extent=[ts_kymo[0], ts_kymo[-1],
                    y_offset + n_beads + 0.5,
                    y_offset + 0.5],
        )
        if i < n_polymers - 1:
            ax_kymo.axhline(y=y_offset + n_beads + 0.5,
                            color="black", linewidth=1.5)
        y_offset += n_beads
    ax_kymo.set_ylim(y_offset + 0.5, 0.5)
    ax_kymo.set_ylabel("Nucleosome\n position", fontsize=8, rotation=90)
    ax_kymo.set_yticks([80,40,0])
    ax_kymo.set_yticklabels([0,40,80])
    _label_panel(ax_kymo, panel_idx)
    panel_idx += 1

    # ---- Polymer type-count panels ----------------------------------
    for i in range(n_polymers):
        tag    = f"polymer_{i}"
        ax     = data_axs[tag]
        counts = counts_list[i]
        ids    = ids_list[i]


        for t in [1, 2, 3]:
            ax.plot(ts_ts, counts[t],
                    color=TYPE_COLORS[t],
                    label=TYPE_LABELS[t],
                    linewidth=1.0)

        ax.set_ylabel("Count\nnucleosomal\ntype", fontsize=8, rotation=90)
        ax.set_ylim(0, 80)
        ax.set_yticks([0,40,80])
        ax.grid(alpha=0.20, linewidth=0.4)
        _label_panel(ax, panel_idx)
        panel_idx += 1

    # ---- Swi6M panel ------------------------------------------------
    if types_df is not None and types_ts is not None:
        ax = data_axs["swi6m"]
        ax.plot(types_ts[::50], types_df["Swi6M"].values[::50],
                color=SWI6M_COLOR, linewidth=1.0, label="Swi6M")
        ax.set_ylabel("Count\n Swi6*", fontsize=8)
        ax.set_yticks([0,15,30])
        ax.grid(alpha=0.20, linewidth=0.4)
        # --- First group: Types ---
        type_handles = [
            Patch(color=TYPE_COLORS[k], label=TYPE_LABELS[k])
            for k in sorted(TYPE_COLORS)
        ]

        # --- Second group: SWI6 / SWI6M ---
        other_handles = [
            Patch(color=SWI6M_COLOR, label="Swi6*"),
            Patch(color=SWI6_COLOR,  label="Swi6"),
        ]

        # Combine all handles
        handles = type_handles + other_handles

        # Draw legend (2 columns)
        ax.legend(handles=handles, ncol=2, fontsize=7)
        
        _label_panel(ax, panel_idx)
        panel_idx += 1

    # ---- Rg panel ---------------------------------------------------
    if rg_ts is not None:
        ax = data_axs["rg"]

        ax.plot(rg_ts[::500], rg_vals[::500],
                color=RG_COLOR, linewidth=1.0, label=r"$R_g$")
        ax.set_ylabel(r"$R^{2}_{g}$ ($\sigma^2$)", fontsize=8)
        ax.grid(alpha=0.20, linewidth=0.4)

        # ticks = np.array([0, 5e7, 1e8, 1.5e8, 2e8])
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
        ax.set_xticks([0,1e8,2e8])
        ax.set_xticklabels([rf'$0$',rf'$1$',rf'$2 \times 10^5$'], fontsize=6)
        _label_panel(ax, panel_idx)
        panel_idx += 1

    # No tight_layout — we set explicit margins in GridSpec above
    if outfile:
        plt.savefig(outfile, dpi=500)
        print(f"Figure saved to {outfile}")


# -- MAIN ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot LAMMPS polymer dump (PRX style).")
    ap.add_argument("dump",             nargs="?", default=DUMP_FILE)
    ap.add_argument("--rg",             default=R2_FILE)
    ap.add_argument("--timeline",       default=TIMELINE_FILE)
    ap.add_argument("--types",          default=TYPES_FILE)
    ap.add_argument("--out",            default=None,
                    help="Save figure to this path instead of displaying it")
    args = ap.parse_args()

    print(f"Reading {args.dump} ...")
    timesteps, frames = parse_dump(args.dump)
    print(f"  -> {len(timesteps)} frames, steps {timesteps[0]}–{timesteps[-1]}")

    ids_list, arrays = build_arrays(frames, POLYMERS)
    for i, ids in enumerate(ids_list):
        print(f"  -> Polymer {i+1}: {len(ids)} beads")

    rg_ts, rg_vals = None, None
    try:
        rg_ts, rg_vals = parse_r2(args.rg)
        print(f"Reading {args.rg} -> {len(rg_ts)} data points")
    except FileNotFoundError:
        print(f"  [warn] {args.rg} not found — skipping Rg panel")

    types_df, types_ts = None, None
    try:
        types_ts, types_df = parse_types(args.types, timesteps)
        print(f"Reading {args.types} -> {len(types_df)} windows, "
              f"steps {types_ts[0]}–{types_ts[-1]}")
    except FileNotFoundError:
        print(f"  [warn] {args.types} not found — skipping Swi6M panel")

    timeline_events = None
    try:
        timeline_events = parse_timeline(args.timeline)
        print(f"Reading {args.timeline} -> {len(timeline_events)} events")
    except FileNotFoundError:
        print(f"  [warn] {args.timeline} not found — skipping timeline panel")

    plot_all(
        timesteps, ids_list, arrays,
        rg_ts=rg_ts,         rg_vals=rg_vals,
        types_df=types_df,   types_ts=types_ts,
        outfile=args.out,
    )


if __name__ == "__main__":
    main()