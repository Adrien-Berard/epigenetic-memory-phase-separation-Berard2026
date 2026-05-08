"""
src_master_figure.py
--------------------

Layout  (4 rows × 4 columns, outer GridSpec):

  Row 0  [col 0]         (a) model PDF (spans rows 0-2)
         [cols 1-3]      (b) snapshot fixed:  pre | meeting | post

  Row 1  [col 0]         (a) model PDF continued
         [cols 1-3]      (c) snapshot diffusive: pre | meeting | post

  Row 2  [col 0]         (a) model PDF continued
         [cols 1-3]      (d) diffusive types count timeseries (spans 3 cols)

  Row 3  [cols 0-1]      (e) contact map fixed (full sim, Hi-C style)
         [cols 2-3]      (f) contact map diffusive: pre | meeting | post
                             (inner 1×3 sub-gridspec)

Usage:
    python src_master_figure.py --out figure2.pdf
    python src_master_figure.py --out figure2.pdf --cutoff 2.5 --n-avg 500
"""

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.patches import Patch, Wedge
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False

warnings.filterwarnings("ignore")

# ── STYLE ─────────────────────────────────────────────────────────────────────
PRX_RC = {
    "font.family":        "serif",
    "font.size":          6.5,
    "axes.labelsize":     6.5,
    "axes.titlesize":     6.5,
    "xtick.labelsize":    5,
    "ytick.labelsize":    5,
    "legend.fontsize":    6.5,
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
plt.rcParams.update(PRX_RC)

A4_WIDTH  = 7.5
A4_HEIGHT = 7   # 4 rows needs a bit more height

# ── FILE PATHS ─────────────────────────────────────────────────────────────────
Fixed_types_dat  = '/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160Fixed/types1.dat'
Diffusive_types_dat  = '/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160/types1.dat'
Fixed_types_dump = '/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160Fixed/dump.lammpstrj'
Diffusive_types_dump = '/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160/dump.lammpstrj'

MODEL_PDF = '/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/model_2poly.pdf'

SNAPSHOT_FIXED_PRE     = "/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160Fixed/fixed_2poy_pre.png"
SNAPSHOT_FIXED_MEETING = "/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160Fixed/fixed_2poy_meeting.png"
SNAPSHOT_FIXED_POST    = "/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160Fixed/fixed_2poy_post.png"

SNAPSHOT_DIFF_PRE      = "/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160/diffusive_2poy_pre.png"
SNAPSHOT_DIFF_MEETING  = "/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160/diffusive_2poy_meeting.png"
SNAPSHOT_DIFF_POST     = "/home/adrien/SPombe_MatRegion_Model/2PolymersPAPER/sim_p2_0.00025_noise_500_swi6_400_nuc_160/diffusive_2poy_post.png"

# ── SIMULATION CONFIG ──────────────────────────────────────────────────────────
SIM_START      = 1000000
TYPES_STEP     = 500
DUMP_STEP      = 10000
TYPES_PER_DUMP = DUMP_STEP // TYPES_STEP   # 20

CHROMATIN        = list(range(1,  81))
HETEROCHROMATIN  = list(range(81, 241))
TWO_POLYMERS     = list(range(1,  241))

TS_STRIDE   = 50
KYMO_STRIDE = 1

# Contact map defaults (overridable via CLI)
CONTACT_CUTOFF   = 3.0     # σ
N_FRAMES_AVG     = 200     # frames used to average contact map

# Time windows for diffusive contact sub-panels
T_EQ      = 10_001_000
T_MEETING = 121_000_000   # row 240000 × 500 steps + SIM_START
T_END     = 152_620_000   # last frame in dump

# ── COLOURS ────────────────────────────────────────────────────────────────────
TYPE_COLORS = {1: "#2166AC", 2: "#F4C300", 3: "#D6001C", 11: "#bf396a"}
TYPE_LABELS = {1: "A", 2: "U", 3: "M", 11: "H"}
TYPE_CMAP   = mcolors.ListedColormap([TYPE_COLORS[k] for k in sorted(TYPE_COLORS)])

SWI6M_COLOR = "#1A9641"
SWI6_COLOR  = "#CC79A7"
RG_COLOR    = "#777777"

# Hi-C colourmap (white → red, log scale)
FRUIT_PUNCH_HIC = mcolors.LinearSegmentedColormap.from_list(
    "hic", ["white", "#fbb4ae", "#c0392b", "#2c0000"])


# ══════════════════════════════════════════════════════════════════════════════
# PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf_page(path, dpi=400):
    """Return first page of a PDF as an RGBA numpy array, or None on failure."""
    if not Path(path).exists():
        print(f"  [warn] PDF not found: {path}")
        return None
    if not PDF2IMAGE_OK:
        print(f"  [warn] pdf2image not installed. Run: pip install pdf2image && sudo apt install poppler-utils")
        return None
    try:
        pages = convert_from_path(path, dpi=dpi)
        arr = np.array(pages[0])
        # # Crop 3% from each edge
        # h, w = arr.shape[:2]
        # cy, cx = int(0.01*h), int(0.01*w)
        # arr = arr[cy:h-cy, cx:w-cx]
        print(f"  [ok] PDF loaded: {path}  ({arr.shape[1]}×{arr.shape[0]} px, cropped)")
        return arr
    except Exception as e:
        print(f"  [warn] pdf2image failed on {path}: {e}")
        print(f"         Check poppler is installed: sudo apt install poppler-utils")
        return None


def load_png(path):
    """Return an image array or None."""
    try:
        arr = mpimg.imread(path)
        h, w = arr.shape[:2]
        cy, cx = int(0.09*h), int(0.09*w)
        arr = arr[cy:h-cy, cx:w-cx]
        return arr
    except Exception as e:
        print(f"  [warn] could not load {path}: {e}")
        return None


def parse_dump_full(filepath):
    """Read entire LAMMPS dump; return (timesteps, frames)."""
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


def parse_types(filepath, dump_timesteps, sim_start=SIM_START):
    """
    Parse types1.dat, align to dump_timesteps by nearest-neighbour.
    Non-overlapping blocks → max(Swi6M) per block avoids sawtooth bleed.
    """
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
    ts_sub  = agg["timestep"].values.astype(np.int64)
    dump_ts = np.array(dump_timesteps, dtype=np.int64)
    idx      = np.searchsorted(ts_sub, dump_ts, side="left").clip(0, len(ts_sub) - 1)
    idx_left = (idx - 1).clip(0, len(ts_sub) - 1)
    nearest  = np.where(
        np.abs(dump_ts - ts_sub[idx_left]) < np.abs(dump_ts - ts_sub[idx]),
        idx_left, idx)
    return pd.DataFrame({col: agg[col].values[nearest]
                         for col in ["A", "U", "M", "Swi6", "Swi6M"]})


# ── Contact map machinery ──────────────────────────────────────────────────────

def index_trajectory(filepath):
    """Fast single-pass index: returns list of {timestep, offset, n_atoms}."""
    path = Path(filepath)
    index = []
    reading_ts = reading_natoms = False
    current_ts = current_offset = None
    with open(path, "rb") as fh:
        while True:
            offset = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="replace").strip()
            if line == "ITEM: TIMESTEP":
                current_offset = offset; reading_ts = True; reading_natoms = False
            elif reading_ts:
                current_ts = int(line); reading_ts = False
            elif line == "ITEM: NUMBER OF ATOMS":
                reading_natoms = True
            elif reading_natoms:
                index.append({"timestep": current_ts,
                               "offset":   current_offset,
                               "n_atoms":  int(line)})
                reading_natoms = False
    print(f"  Indexed {len(index)} frames in {filepath}")
    return index


def _parse_frame_at(fh, frame_info):
    """Parse one frame starting at its stored byte offset."""
    fh.readline()                           # ITEM: TIMESTEP
    timestep = int(fh.readline().strip())
    fh.readline()                           # ITEM: NUMBER OF ATOMS
    n_atoms  = int(fh.readline().strip())
    fh.readline()                           # ITEM: BOX BOUNDS
    box = np.array([list(map(float, fh.readline().split())) for _ in range(3)])
    L   = box[:, 1] - box[:, 0]
    header = fh.readline()
    cols   = header.split()[2:]
    id_col = cols.index("id")
    try:    x_col = cols.index("xs"); scaled = True
    except: x_col = cols.index("x");  scaled = False
    y_col, z_col = x_col + 1, x_col + 2
    coords = {}
    for _ in range(n_atoms):
        parts = fh.readline().split()
        aid   = int(parts[id_col])
        xyz   = np.array([float(parts[x_col]),
                          float(parts[y_col]),
                          float(parts[z_col])])
        if scaled:
            xyz = box[:, 0] + xyz * L
        coords[aid] = xyz
    return timestep, box, coords


def _contact_matrix(coords, polymer_ids, cutoff, box):
    ids = sorted(pid for pid in polymer_ids if pid in coords)
    L   = box[:, 1] - box[:, 0]
    pos = np.array([coords[i] for i in ids])
    diff = pos[:, None, :] - pos[None, :, :]
    diff -= np.round(diff / L) * L
    dist  = np.sqrt((diff ** 2).sum(-1))
    mat   = (dist <= cutoff).astype(np.float32)
    np.fill_diagonal(mat, 0)
    return mat, ids


def average_contact_map(filepath, polymer_ids, cutoff, n_frames,
                         t_min=None, t_max=None):
    """
    Load up to n_frames evenly-spaced frames (optionally within [t_min, t_max])
    and return the averaged contact matrix.
    """
    index = index_trajectory(filepath)

    # Filter by time window if requested
    if t_min is not None or t_max is not None:
        index = [f for f in index
                 if (t_min is None or f["timestep"] >= t_min)
                 and (t_max is None or f["timestep"] <= t_max)]

    if not index:
        print("  [warn] No frames found in time window — returning zeros")
        n = len(polymer_ids)
        return np.zeros((n, n)), polymer_ids

    n_pick = min(n_frames, len(index))
    pick   = np.round(np.linspace(0, len(index) - 1, n_pick)).astype(int)
    pick   = sorted(set(pick.tolist()))

    avg_mat = None
    ids_out = None
    with open(filepath, "r") as fh:
        for pos in tqdm(pick, desc=f"  Contact map ({len(pick)} frames)"):
            fh.seek(index[pos]["offset"])
            _, box, coords = _parse_frame_at(fh, index[pos])
            mat, ids = _contact_matrix(coords, polymer_ids, cutoff, box)
            if avg_mat is None:
                avg_mat = mat.astype(np.float64)
                ids_out = ids
            else:
                avg_mat += mat

    avg_mat /= len(pick)
    print(f"  Contact map: {avg_mat.shape[0]} beads, cutoff={cutoff}σ")
    return avg_mat, ids_out


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_image(ax, img, title="", letter=None,aspect='equal'):
    """Show a numpy image array on ax, or a placeholder if None."""
    if img is not None:
        ax.imshow(img,aspect=aspect)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.margins(0)
        ax.set_frame_on(False)
    else:
        ax.text(0.5, 0.5, title or "image not found",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="grey")
    ax.axis("off")
    # if title:
        # ax.set_title(title, fontsize=8, pad=3)
    if letter:
        ax.text(-0.04, 0.98, letter, transform=ax.transAxes,
                fontsize=PRX_RC["axes.titlesize"],
                va="bottom", ha="right")


# Region badge colours: ① pre-meeting  ② post-meeting
REGION_COLORS = ["#2166AC", "#D6001C", "#c2bbba"]   # blue=pre, red=post, grey=other
REGION_LABELS = [" Pre ", "Post ",'Fixed']

def badge(ax, label, color, x=0.16, y=0.83):
    """Stamp a coloured circled number in axes-fraction coords."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=7, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.18", facecolor=color,
                      edgecolor="white", linewidth=0.8, alpha=0.75))


def draw_hic(ax, avg_mat, title="", cbar=False, vmin=-4, vmax=0):
    """Draw a log10 Hi-C style contact map with optional shared colour scale."""
    log_mat = np.log10(avg_mat + 1e-3)
    if vmin is None: vmin = np.percentile(log_mat, 2)
    if vmax is None: vmax = np.percentile(log_mat, 98)
    im = ax.imshow(log_mat, cmap='Reds', vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest", aspect="equal")
    ax.set_xlabel("Nucleosome position", fontsize=PRX_RC["axes.labelsize"])
    ax.set_ylabel("Nucleosome\nposition", fontsize=PRX_RC["axes.labelsize"])
    ax.set_xticks([0, 80, 160, 240], ["1", "80", "160", "240"])
    ax.set_yticks([0, 80, 160, 240], ["1", "80", "160", "240"])
    # if title:
    #     ax.set_title(title, fontsize=8, pad=3)
    return im   # caller decides where/whether to put a colourbar


def draw_types_timeseries(ax, timesteps, types_df, title=""):
    """Plot A/U/M + Swi6M counts on ax."""
    ts = np.array(timesteps)
    counts = {t: np.zeros(len(ts)) for t in [1, 2, 3]}
    # types_df already aligned to timesteps
    for t in [1, 2, 3]:
        col = {1: "A", 2: "U", 3: "M"}[t]
        if col in types_df.columns:
            counts[t] = types_df[col].values[::TS_STRIDE]

    ts_ds = ts[::TS_STRIDE]
    for t in [1, 2, 3]:
        if t == 3:
            ax.plot(ts_ds, counts[t] - 160, color=TYPE_COLORS[t],
                label=TYPE_LABELS[t], lw=1.2) # just plot the 1st polymer
        else:
            ax.plot(ts_ds, counts[t], color=TYPE_COLORS[t],
                label=TYPE_LABELS[t], lw=1.2)
    ax.set_xlabel(r"Time ($\tau_{LJ}$)", fontsize=PRX_RC["axes.labelsize"],labelpad = 3)
    ax.set_xticks([0, 5e7, 10e7, 15e7], ["0", "5", "10", rf"$15 \times 10^4$"])
    ax.set_ylabel("Count\n nucleosomal \n type", fontsize=PRX_RC["axes.labelsize"])
    # ax.legend(fontsize=7, loc="upper right", ncol=4)
    ax.grid(alpha=0.25)
    # if title:
    #     ax.set_title(title, fontsize=8, pad=3)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(cutoff=CONTACT_CUTOFF, n_avg=N_FRAMES_AVG, outfile=None):

    # ── 1. Load all data ───────────────────────────────────────────────────
    print("Loading model PDF …")
    img_model = load_pdf_page(MODEL_PDF, dpi=300)

    print("Loading snapshots …")
    snaps = {
        "fixed_pre":     load_png(SNAPSHOT_FIXED_PRE),
        "fixed_meeting": load_png(SNAPSHOT_FIXED_MEETING),
        "fixed_post":    load_png(SNAPSHOT_FIXED_POST),
        "diff_pre":      load_png(SNAPSHOT_DIFF_PRE),
        "diff_meeting":  load_png(SNAPSHOT_DIFF_MEETING),
        "diff_post":     load_png(SNAPSHOT_DIFF_POST),
    }

    print("Loading diffusive dump for timeseries …")
    diff_ts, diff_frames = parse_dump_full(Diffusive_types_dump)
    print(f"  -> {len(diff_ts)} frames, {diff_ts[0]}–{diff_ts[-1]}")
    diff_types = parse_types(Diffusive_types_dat, diff_ts)

    print("Computing contact maps …")
    # (e) Fixed — full simulation
    mat_fixed, ids_fixed = average_contact_map(
        Fixed_types_dump, TWO_POLYMERS, cutoff, n_avg)

    # (f) Diffusive — pre and post only
    mat_diff_pre, _  = average_contact_map(
        Diffusive_types_dump, TWO_POLYMERS, cutoff, n_avg,
        t_min=T_EQ, t_max=T_MEETING)
    mat_diff_post, _ = average_contact_map(
        Diffusive_types_dump, TWO_POLYMERS, cutoff, n_avg,
        t_min=T_MEETING, t_max=T_END)

    # Common log scale across all three contact maps
    all_mats  = [mat_fixed, mat_diff_pre, mat_diff_post]
    log_mats  = [np.log10(m + 1e-3) for m in all_mats]
    vmin_global = -3
    vmax_global = 0

    # ── 2. Build figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))

    # Outer grid: 4 rows × 4 columns
    # Col widths: model col slightly wider; snapshot cols equal
    # outer = GridSpec(
    #     4, 4,
    #     figure=fig,
    #     height_ratios=[1.0, 1.0, 1.0, 1.2],   # rows 0-2 equal; row 3 contact maps
    #     width_ratios=[1.45, 1.0, 1.0, 1.0],
    #     hspace=0.45,
    #     wspace=0,
    # )
    
    outer = GridSpec(2, 1, height_ratios=[1.3, 1], hspace=0.1)

    # Top block (rows 0–2 in your original layout)
    top = GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=outer[0],
        height_ratios=[1, 1],
        width_ratios=[1.45, 1, 1, 1],
        hspace=0,   # tighter spacing for snapshots + timeseries
        wspace=0
    )

    # Bottom block (row 3 contact maps)
    bottom = GridSpecFromSubplotSpec(
        2, 5,
        subplot_spec=outer[1],
        height_ratios=[1, 1],
        width_ratios=[0.75,1, 1, 1, 0.06],
        hspace=0.45,
        wspace=0.05
    )

    # ── (a) Model PDF — spans rows 0, 1, 2 in column 0 ───────────────────
    # ax_model = fig.add_subplot(outer[0:3, 0])
    ax_model = fig.add_subplot(top[0:2, 0])
    draw_image(ax_model, img_model, letter="(a)")
    # ax_model.set_title("Model", fontsize=PRX_RC["axes.titlesize"], pad=4)

    # ── (b) Fixed snapshots — row 0, cols 1-3 ─────────────────────────────
    snap_titles_fixed   = ["Fixed — pre", "Fixed — meeting", "Fixed — post"]
    snap_keys_fixed     = ["fixed_pre", "fixed_meeting", "fixed_post"]
    letters_b           = ["(b)", "", ""]          # label only the first panel

    for col_offset, (key, title, letter) in enumerate(
            zip(snap_keys_fixed, snap_titles_fixed, letters_b)):
        ax = fig.add_subplot(top[0, col_offset + 1])
        draw_image(ax, snaps[key], title=title, letter=letter if col_offset == 0 else None, aspect='equal')
        # ① pre  ② post  (no badge on meeting col)
        badge(ax, REGION_LABELS[2], REGION_COLORS[2])   # fixed region badge 

    # ── (c) Diffusive snapshots — row 1, cols 1-3 ─────────────────────────
    snap_titles_diff = ["Diffusive — pre", "Diffusive — meeting", "Diffusive — post"]
    snap_keys_diff   = ["diff_pre", "diff_meeting", "diff_post"]

    for col_offset, (key, title) in enumerate(zip(snap_keys_diff, snap_titles_diff)):
        ax = fig.add_subplot(top[1, col_offset + 1])
        draw_image(ax, snaps[key], title=title,
                   letter="(c)" if col_offset == 0 else None, aspect='equal')
        if col_offset == 0: badge(ax, REGION_LABELS[0], REGION_COLORS[0])
        if col_offset == 2: badge(ax, REGION_LABELS[1], REGION_COLORS[1])

    # ── (d) Diffusive types timeseries — row 2, cols 1-3 (merged) ─────────
    ax_ts = fig.add_subplot(bottom[0, 1:])    # spans cols 1, 2, 3
    draw_types_timeseries(ax_ts, diff_ts, diff_types,
                          title="Diffusive — nucleosome type counts")
    ax_ts.axvline(T_MEETING, color="black", lw=1.1, ls="--", alpha=0.7)
    x_pre  = 0.5 * (T_MEETING - T_EQ)       / (diff_ts[-1] - T_EQ)
    x_post = (T_MEETING - T_EQ + 0.5 * (diff_ts[-1] - T_MEETING)) / (diff_ts[-1] - T_EQ)
    badge(ax_ts, REGION_LABELS[0], REGION_COLORS[0], x=x_pre,  y=0.92)
    badge(ax_ts, REGION_LABELS[1], REGION_COLORS[1], x=x_post, y=0.92)
    ax_ts.text(-0.03, 1.02, "(d)", transform=ax_ts.transAxes,
               fontsize=PRX_RC["axes.titlesize"],
               va="bottom", ha="right")

    # ── Row 3: 3 equal contact maps + 1 shared colourbar ─────────────────
    # Use an inner 1×4 sub-gridspec spanning the full width of row 3:
    #   col 0 = fixed (full sim)
    #   col 1 = diffusive pre
    #   col 2 = diffusive post
    #   col 3 = narrow colourbar
    hic_gs = GridSpecFromSubplotSpec(
        1, 4, subplot_spec=bottom[1, 1:],
        width_ratios=[1, 1, 1, 0.06],
        wspace=0.32)

    hic_axes  = [fig.add_subplot(hic_gs[0, j]) for j in range(3)]
    hic_data   = [mat_fixed,    mat_diff_pre,     mat_diff_post]
    hic_titles = ["Fixed (full sim)", "Diffusive — pre", "Diffusive — post"]
    hic_letters = ["(e)", "(f)", ""]

    # Map contact panels to region badges: j=0 fixed (no badge), j=1 pre ①, j=2 post ②
    hic_badge = [2, 0, 1]   # index into REGION_LABELS/COLORS, None = no badge
    last_im = None
    for j, (ax, mat, title, letter) in enumerate(
            zip(hic_axes, hic_data, hic_titles, hic_letters)):
        last_im = draw_hic(ax, mat, title=title,
                           vmin=vmin_global, vmax=vmax_global)
        if letter:
            ax.text(-0.34, 0.98, letter, transform=ax.transAxes,
                    fontsize=PRX_RC["axes.titlesize"],
                    va="bottom", ha="right")
        if hic_badge[j] is not None:
            badge(ax, REGION_LABELS[hic_badge[j]], REGION_COLORS[hic_badge[j]],
                  x=1, y=0.65)
        if j > 0:
            ax.set_ylabel("")   # avoid duplicate y-labels

    # Shared colourbar in the dedicated narrow column
    ax_cbar = fig.add_subplot(hic_gs[0, 3])
    plt.colorbar(last_im, cax=ax_cbar, label=r"$log_{10}(P_{contact})$", ticks=[-3, -2, -1, 0])

    # ── Save / show ────────────────────────────────────────────────────────
    if outfile:
        fig.savefig(outfile, dpi=PRX_RC["savefig.dpi"], bbox_inches="tight")
        print(f"Saved → {outfile}")
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Master figure — 2-polymer model")
    ap.add_argument("--out",    default=None,
                    help="Output path (PDF/PNG/SVG). Default: show interactively.")
    ap.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF,
                    help=f"Contact distance cutoff in σ  (default: {CONTACT_CUTOFF})")
    ap.add_argument("--n-avg",  type=int,   default=N_FRAMES_AVG,
                    help=f"Frames averaged per contact map (default: {N_FRAMES_AVG})")
    args = ap.parse_args()

    make_figure(cutoff=args.cutoff, n_avg=args.n_avg, outfile=args.out)