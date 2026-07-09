"""
src_fig_S6.py
----------------
Supplementary Figure S6: DNA-replication and cell-cycle kymographs.

Usage:
    python src_fig_S6.py
    python src_fig_S6.py --out figS6.pdf
    python src_fig_S6.py --window 550000000 650000000
"""

import argparse
import csv
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.patches import Wedge
from matplotlib.patches import Patch
from PIL import Image          

# ── FILE PATHS ────────────────────────────────────────────────────────────────
M_NOISE_1 = 'FigS6/Same_cycle_length_M/id_and_type.dat'
A_NOISE_1 = 'FigS6/Same_cycle_length/id_and_type.dat'
BOTH_NOISE_2 = 'FigS6/Same_cycle_noise500/id_and_type.dat'

# ── SIMULA TION CONSTANTS ──────────────────────────────────────────────────────
SIM_START      = 1000000   # timestep of row 0 in types1.dat
TYPES_STEP     = 500
DUMP_STEP      = 10000
TYPES_PER_DUMP = DUMP_STEP // TYPES_STEP   # = 20

# Polymer bead IDs for the cell-cycle dump
CC_POLYMERS = [(1, 80)]


# ── COLOURS ───────────────────────────────────────────────────────────────────
TYPE_COLORS = {1: "#2166AC", 2: "#F4C300", 3: "#D6001C"}
TYPE_LABELS = {1: "A", 2: "U", 3: "M"}
TYPE_CMAP   = mcolors.ListedColormap([TYPE_COLORS[k] for k in sorted(TYPE_COLORS)])
KYMO_NORM   = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], TYPE_CMAP.N)

SWI6M_COLOR = "#1A9641"
RG_COLOR    = "#777777"

PHASE_STYLES = {
    "G1":      {"color": "#74C476", "alpha": 0.35, "label": "G1"},
    "S":       {"color": "#9E9AC8", "alpha": 0.35, "label": "S-phase"},
    "G2":      {"color": "#4292C6", "alpha": 0.35, "label": "G2"},
    "Mitosis": {"color": "#EF6548", "alpha": 0.55, "label": "Mitosis"},
}

# ── STYLE ─────────────────────────────────────────────────────────────────────
PRX_RC = {
    "font.family":       "serif",
    "font.size":         7,
    "axes.labelsize":    7,
    "axes.titlesize":    7,
    "xtick.labelsize":   5,
    "ytick.labelsize":   5,
    "legend.fontsize":   6,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "0.7",
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "lines.linewidth":   1.2,
    "figure.dpi":        500,
    "savefig.dpi":       500,
    "savefig.bbox":      "tight",
}
plt.rcParams.update(PRX_RC)

A4_WIDTH  = 7.5
A4_HEIGHT = 6


# ═══════════════════════════════════════════════════════════════════════════════
# PARSERS  
# ═══════════════════════════════════════════════════════════════════════════════

def parse_dump(filepath):
    """Return (timesteps np.int64, frames list-of-dicts) for the full file."""
    timesteps, frames = [], []
    with open(filepath) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            current_ts = int(lines[i + 1].strip()); i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i + 1].strip()); i += 2
        elif line.startswith("ITEM: ATOMS"):
            header   = line.split()[2:]
            col_id   = header.index("id")
            col_type = header.index("type")
            frame = {}
            for _ in range(n_atoms):
                i += 1
                parts = lines[i].split()
                frame[int(parts[col_id])] = int(parts[col_type])
            timesteps.append(current_ts)
            frames.append(frame)
            i += 1
        else:
            i += 1
    return np.array(timesteps, dtype=np.int64), frames


def parse_r2(filepath, dump_timesteps):
    ts, vals = [], []
    with open(filepath) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            t, v = line.split()
            ts.append(int(t)); vals.append(float(v))
    return np.interp(dump_timesteps,
                     np.array(ts, dtype=np.int64),
                     np.array(vals, dtype=float))


def parse_types(filepath, dump_timesteps, sim_start=SIM_START):
    df = pd.read_csv(filepath, comment="#", names=["A", "U", "M", "Swi6", "Swi6M"])
    df["timestep"] = sim_start + np.arange(len(df), dtype=np.int64) * TYPES_STEP
    w = TYPES_PER_DUMP
    df["block"] = np.arange(len(df)) // w
    agg = df.groupby("block", sort=True).agg(
        A     =("A",     "mean"),
        U     =("U",     "mean"),
        M     =("M",     "mean"),
        Swi6  =("Swi6",  "mean"),
        Swi6M =("Swi6M", "max"),
    ).reset_index(drop=True).dropna()
    agg["timestep"] = sim_start + (agg.index + 1) * DUMP_STEP
    ts_sub   = agg["timestep"].values.astype(np.int64)
    dump_ts  = np.array(dump_timesteps, dtype=np.int64)
    idx      = np.searchsorted(ts_sub, dump_ts, side="left").clip(0, len(ts_sub) - 1)
    idx_left = (idx - 1).clip(0, len(ts_sub) - 1)
    nearest  = np.where(
        np.abs(dump_ts - ts_sub[idx_left]) < np.abs(dump_ts - ts_sub[idx]),
        idx_left, idx)
    return pd.DataFrame({col: agg[col].values[nearest]
                         for col in ["A", "U", "M", "Swi6", "Swi6M"]})


def parse_timeline(filepath):
    events = []
    with open(filepath, newline="") as fh:
        for row in csv.DictReader(fh):
            events.append({"step":  int(row["step"]),
                           "event": row["event"].strip(),
                           "cycle": int(row["cycle"])})
    events.sort(key=lambda e: e["step"])
    return events


def build_arrays(frames, polymers):
    ids_list = [list(range(p[0], p[1] + 1)) for p in polymers]
    n        = len(frames)
    arrays   = [np.zeros((n, len(ids)), dtype=np.int32) for ids in ids_list]
    for t, frame in enumerate(frames):
        for p_idx, ids in enumerate(ids_list):
            for j, aid in enumerate(ids):
                arrays[p_idx][t, j] = frame.get(aid, 0)
    return ids_list, arrays


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE-BAND HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_phase_bands(events, x_min, x_max):
    def find_next(evs, i, evt, cyc):
        for j in range(i + 1, len(evs)):
            if evs[j]["event"] == evt and evs[j]["cycle"] == cyc:
                return evs[j]["step"]
    def find_next_any(evs, i, evt):
        for j in range(i + 1, len(evs)):
            if evs[j]["event"] == evt:
                return evs[j]["step"]

    bands = []
    for i, ev in enumerate(events):
        name, start, cycle = ev["event"], ev["step"], ev["cycle"]
        if name == "G2_start":
            end = find_next(events, i, "G2_end", cycle)
            if end: bands.append({"phase": "G2", "start": start, "end": end, "cycle": cycle})
        elif name == "Mitosis_start":
            end = find_next(events, i, "Mitosis_end", cycle)
            if end: bands.append({"phase": "Mitosis", "start": start, "end": end, "cycle": cycle})
        elif name == "G1_start":
            end = find_next_any(events, i, "G2_start") or x_max
            bands.append({"phase": "G1", "start": start, "end": end, "cycle": cycle})

    bands.sort(key=lambda b: b["start"])
    filled, prev = [], events[0]["step"] if events else x_min
    for b in bands:
        if b["start"] > prev:
            filled.append({"phase": "S", "start": prev, "end": b["start"], "cycle": -1})
        filled.append(b)
        prev = max(prev, b["end"])
    if prev < x_max:
        filled.append({"phase": "S", "start": prev, "end": x_max, "cycle": -1})

    return [
        {**b, "start": max(b["start"], x_min), "end": min(b["end"], x_max)}
        for b in filled if b["end"] > x_min and b["start"] < x_max
    ]


def draw_phase_bands(ax, bands):
    for b in bands:
        s = PHASE_STYLES.get(b["phase"], {"color": "grey", "alpha": 0.2})
        ax.axvspan(b["start"], b["end"], color=s["color"], alpha=s["alpha"], lw=0)


def phase_legend_patches():
    return [Patch(color=v["color"], alpha=max(v["alpha"], 0.6), label=v["label"])
            for v in PHASE_STYLES.values()]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def crop_img(img, f=0.08):
    h, w = img.shape[:2]
    dy, dx = int(h * f), int(w * f)
    # crop bottom only
    return img[dy:h-dy, dx:w-dx]


def load_pdf_as_image(path):
    """Rasterise first page of a PDF to an RGBA numpy array via pdf2image."""
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(path, dpi=500)
        return crop_img(np.array(pages[0]),0.2)
    except Exception as e:
        print(f"  [warn] could not rasterise {path}: {e}")
        return None

def cell_cycle_model(ax):
    PHASE_STYLES = {
        "G1":      {"color": "#74C476", "alpha": 0.35, "label": "G1 (10%)"},
        "G2":      {"color": "#4292C6", "alpha": 0.35, "label": "G2 (70%)"},
        "Mitosis": {"color": "#EF6548", "alpha": 0.55, "label": "Mitosis (20%)"},
    }

    cycle = {
        "Mitosis": 0.20,
        "G2": 0.70,
        "G1": 0.10


    }

    ax.set_aspect("equal")

    # IMPORTANT: prevent cropping
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    radius = 1.05
    width = 0.35

    start_angle = -(414+180)

    boundary_angles = {}

    for phase, frac in cycle.items():
        end_angle = start_angle + frac * 360

        wedge = Wedge(
            (0, 0),
            radius,
            start_angle,
            end_angle,
            width=width,
            facecolor=PHASE_STYLES[phase]["color"],
            alpha=PHASE_STYLES[phase]["alpha"],
            edgecolor="white"
        )
        ax.add_patch(wedge)

        mid = (start_angle + end_angle) / 2
        x = 0.8 * np.cos(np.deg2rad(mid))
        y = 0.8 * np.sin(np.deg2rad(mid))
        if phase == 'G1':
            ax.text(x-0.4, y, PHASE_STYLES[phase]["label"], ha="center", va="center",
        bbox=dict(facecolor='white', alpha=0.6,boxstyle="round,pad=0.4"),fontsize=PRX_RC["axes.labelsize"]-1)
        else: 
            ax.text(x, y, PHASE_STYLES[phase]["label"], ha="center", va="center",
        bbox=dict(facecolor='white', alpha=0.6,boxstyle="round,pad=0.4"),fontsize=PRX_RC["axes.labelsize"]-1)

        boundary_angles[phase] = start_angle
        start_angle = end_angle

    # -------------------------
    # S-phase spike (visual marker)
    # -------------------------
    
    s_angle = boundary_angles["G1"]  
    print(s_angle)

    x0, y0 = 0, 0
    x1 = 1.1 * np.cos(np.deg2rad(s_angle))
    y1 = 1.1 * np.sin(np.deg2rad(s_angle))
    x2 = (1.1-0.45) * np.cos(np.deg2rad(s_angle))
    y2 = (1.1-0.45) * np.sin(np.deg2rad(s_angle))

    ax.plot([x2, x1], [y2, y1], color="#9E9AC8", linewidth=3)
    ax.scatter([x2, x1], [y2, y1], color="#9E9AC8", s=60)

    ax.text(
        0.8* np.cos(np.deg2rad(s_angle-20)) + 0.4,
        0.8 * np.sin(np.deg2rad(s_angle-20)),
        "S (~ 0.1%)",
        bbox=dict(facecolor='white', alpha=0.6,boxstyle="round,pad=0.4"),
        ha="center",
        va="center",
        color='black',fontsize=PRX_RC["axes.labelsize"]-1
    )
    ax.set_box_aspect(1)

def dna_rep_model(ax):
    PHASE_STYLES = {
    'G1 + G2 + M': {"color": "#E0A8E7", "alpha": 0.55, "label": 'G1 + G2 + M (100%)'}
    }

    cycle = {
        'G1 + G2 + M':1.0


    }

    ax.set_aspect("equal")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    radius = 1.05
    width = 0.35

    start_angle = 90

    boundary_angles = {}

    for phase, frac in cycle.items():
        end_angle = start_angle + frac * 360

        wedge = Wedge(
            (0, 0),
            radius,
            start_angle,
            end_angle,
            width=width,
            facecolor=PHASE_STYLES[phase]["color"],
            alpha=PHASE_STYLES[phase]["alpha"],
            edgecolor="white"
        )
        ax.add_patch(wedge)

        mid = (start_angle + end_angle) / 2
        x = 0.8 * np.cos(np.deg2rad(mid))
        y = 0.8 * np.sin(np.deg2rad(mid))

        ax.text(x, y, PHASE_STYLES[phase]["label"], ha="center", va="center",
        bbox=dict(facecolor='white', alpha=0.6,boxstyle="round,pad=0.4"),fontsize=PRX_RC["axes.labelsize"]-1)

        boundary_angles[phase] = start_angle
        start_angle = end_angle

    # -------------------------
    # S-phase spike (visual marker)
    # -------------------------

    s_angle = 90 

    x0, y0 = 0, 0
    x1 = 1.1 * np.cos(np.deg2rad(s_angle))
    y1 = 1.1 * np.sin(np.deg2rad(s_angle))
    x2 = (1.1-0.45) * np.cos(np.deg2rad(s_angle))
    y2 = (1.1-0.45) * np.sin(np.deg2rad(s_angle))

    ax.plot([x2, x1], [y2, y1], color="#9E9AC8", linewidth=3)
    ax.scatter([x2, x1], [y2, y1], color="#9E9AC8", s=60)

    ax.text(
        0.8* np.cos(np.deg2rad(s_angle-20)),
        0.8 * np.sin(np.deg2rad(s_angle-20)),
        "S (~ 0.1%)",
        bbox=dict(facecolor='white', alpha=0.6,boxstyle="round,pad=0.4"),
        ha="center",
        va="center",
        color='black',fontsize=PRX_RC["axes.labelsize"]-1
    )
    ax.set_box_aspect(1)

def draw_kymograph(ax, timesteps, arrays, ids_list, ylabel="Nucleosome\nposition"):
    y_offset = 0
    n_poly   = len(arrays)
    for i, (arr, ids) in enumerate(zip(arrays, ids_list)):
        n_beads = arr.shape[1]
        ax.imshow(arr.T, aspect="auto", origin="upper",
                  cmap=TYPE_CMAP, norm=KYMO_NORM, interpolation="nearest",
                  extent=[timesteps[0], timesteps[-1],
                          y_offset + n_beads + 0.5, y_offset + 0.5])
        if i < n_poly - 1:
            ax.axhline(y_offset + n_beads + 0.5, color="black", lw=1.5)
        y_offset += n_beads
    ax.set_ylim(y_offset + 0.5, 0.5)
    ax.set_ylabel(ylabel, fontsize=PRX_RC["axes.labelsize"])
    # ax.set_xlabel(r"Time ($\tau_{LJ}$)", fontsize=PRX_RC["axes.labelsize"])

    ax.set_yticks([1,40,80])
    ax.set_yticklabels([80,40,1])

def label_ax(ax, letter, x=-0.10, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=PRX_RC["axes.titlesize"],
            va="bottom", ha="right")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(outfile=None):

    # ── Load DNA-replication kymograph (full sim) ──────────────────────────
    print("Loading DNA-replication dump …")
    dna_A_ts, dna_A_frames = parse_dump(A_NOISE_1)
    dna_A_ids, dna_A_arrays = build_arrays(dna_A_frames, CC_POLYMERS)
    print(f"  -> {len(dna_A_ts)} frames, {dna_A_ts[0]}–{dna_A_ts[-1]}")


    print("Loading DNA-replication dump …")
    dna_M_ts, dna_M_frames = parse_dump(M_NOISE_1)
    dna_M_ids, dna_M_arrays = build_arrays(dna_M_frames, CC_POLYMERS)
    print(f"  -> {len(dna_M_ts)} frames, {dna_M_ts[0]}–{dna_M_ts[-1]}")

    print("Loading DNA-replication dump …")
    dna_both_ts, dna_both_frames = parse_dump(BOTH_NOISE_2)
    dna_both_ids, dna_both_arrays = build_arrays(dna_both_frames, CC_POLYMERS)
    print(f"  -> {len(dna_both_ts)} frames, {dna_both_ts[0]}–{dna_both_ts[-1]}")
    # ══════════════════════════════════════════════════════════════════════
    # BUILD FIGURE
    # Layout:
    #   outer_gs  2 rows × 2 cols  (top = kymo+model, bottom = zoom panels)
    #   zoom_gs   sub-gridspec inside bottom strip, 4 rows × 1 col, spans full width
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))

    outer = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1, 1, 1],   # DNA kymo/model | CC kymo/model | zoom
        width_ratios=[1,0.3],
        hspace=0.24,
        wspace=0.02,
    )

    # ── (a) DNA replication kymograph ─────────────────────────────────────
    ax_a = fig.add_subplot(outer[0, 0])
    
    draw_kymograph(ax_a, dna_A_ts, dna_A_arrays, dna_A_ids,
                   ylabel="Nucleosome\nposition")
    ax_a.set_xticks([1001000,50e7])
    ax_a.set_xlim(1001000,996930000)
    ax_a.set_xticklabels([])
    ax_a.set_title("100 replication cycles", fontsize=PRX_RC["axes.titlesize"])
    label_ax(ax_a, "(a)")

    # ── (c) DNA replication model ──────────────────────────────────────────
    ax_b = fig.add_subplot(outer[1, 1])
    dna_rep_model(ax_b)
    ax_b.axis("off")
    label_ax(ax_b, "(c)", x = 0.2)

    # ── (b)  kymograph (full sim) ───────────────────────────────
    ax_c = fig.add_subplot(outer[1, 0])
    draw_kymograph(ax_c, dna_M_ts, dna_M_arrays, dna_M_ids,
                   ylabel="Nucleosome\nposition")
    ax_c.set_xticks([1001000,50e7])
    ax_c.set_xticklabels([])
    ax_c.set_xlim(1001000,996930000)
    label_ax(ax_c, "(b)")

    # ── (d)  kymograph (full sim) ───────────────────────────────
    ax_d = fig.add_subplot(outer[2, 0])
    draw_kymograph(ax_d, dna_both_ts, dna_both_arrays, dna_both_ids,
                   ylabel="Nucleosome\nposition")
    ax_d.set_xticks([1001000,50e7])
    ax_d.set_xticklabels(['0',rf'$5 \times 10^5$'])
    ax_d.set_xlim(1001000,996930000)
    ax_d.set_xlabel(r"Time ($\tau_{LJ}$)", fontsize=PRX_RC["axes.labelsize"])
    label_ax(ax_d, "(d)")


    if outfile:
        fig.savefig(outfile)
        print(f"Saved → {outfile}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Master figure")
    ap.add_argument("--out",    default=None,
                    help="Output file (PDF/PNG/SVG).  Default: show interactively.")
    args = ap.parse_args()
    make_figure(outfile=args.out)