"""
src_supp_figure.py
------------------
Layout:
  Swi6=200  : timeseries row + contact row (pre | post | cbar)
  Swi6=600  : timeseries row + contact row (pre | post | cbar)
  Swi6=1000 : timeseries row + contact row (pre | switch | post-switch | post | cbar)

All contact maps share a global colour scale.
Cache: contact matrices are stored as .npy files next to the dump so they are
computed only once. Delete the .npy files to force recomputation.

Usage:
    python src_supp_figure.py --out supp2poly.pdf
    python src_supp_figure.py --out supp2poly.pdf --cutoff 3.0 --n-avg 100
"""

import argparse
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── STYLE ─────────────────────────────────────────────────────────────────────
PRX_RC = {
    "font.family":        "serif",
    "font.size":          8,
    "axes.labelsize":     8,
    "axes.titlesize":     8,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    8,
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

A4_WIDTH  = 7.5   # wider to fit 4 contact maps in row 3
A4_HEIGHT = 9

# ── FILE PATHS ─────────────────────────────────────────────────────────────────
BASE = '/home/adrien/SPombe_MatRegion_Model/March2026/diffusive'

swi6_200_dump  = f'{BASE}/sim_p2_0.00025_noise_500_swi6_200_nuc_80/dump.lammpstrj'
swi6_600_dump  = f'{BASE}/sim_p2_0.00025_noise_500_swi6_600_nuc_80/dump.lammpstrj'
swi6_1000_dump = f'{BASE}/sim_p2_0.00025_noise_500_swi6_1000_nuc_80/dump.lammpstrj'

swi6_200_types  = f'{BASE}/sim_p2_0.00025_noise_500_swi6_200_nuc_80/types1.dat'
swi6_600_types  = f'{BASE}/sim_p2_0.00025_noise_500_swi6_600_nuc_80/types1.dat'
swi6_1000_types = f'{BASE}/sim_p2_0.00025_noise_500_swi6_1000_nuc_80/types1.dat'

# ── SIMULATION CONFIG ──────────────────────────────────────────────────────────
SIM_START      = 1_000_000
TYPES_STEP     = 500
DUMP_STEP      = 10_000
TYPES_PER_DUMP = DUMP_STEP // TYPES_STEP   # 20

TWO_POLYMERS = list(range(1, 161))

TS_STRIDE      = 20
CONTACT_CUTOFF = 3.0
N_FRAMES_AVG   = 100

# Time windows — nothing beyond T_END is ever opened or displayed
T_EQ               = 1_000_000
T_END              = 101_000_000   # hard ceiling for ALL files

T_MEETING_200      = 15_750_000
T_MEETING_600      = 60_500_000
T_MEETING_1000     = 37_500_000
T_SWITCH_START_1000 = 24_500_000
T_SWITCH_END_1000   = 27_500_000

# ── COLOURS ───────────────────────────────────────────────────────────────────
TYPE_COLORS = {1: "#2166AC", 2: "#F4C300", 3: "#D6001C"}
TYPE_LABELS = {1: "A", 2: "U", 3: "M"}

# Badges for the 4 windows of swi6=1000:
#   0=pre   1=switch-start→end   2=post-switch→meeting   3=post-meeting
REGION_COLORS = ["#2166AC", "#9E9AC8", "#F4A582", "#D6001C"]
REGION_LABELS = ["Pre", "Switch", "Post-sw", "Post"]

# For the 2-window sims (200, 600) only indices 0 and 3 are used
PRE_COLOR  = REGION_COLORS[0]
POST_COLOR = REGION_COLORS[3]
PRE_LABEL  = REGION_LABELS[0]
POST_LABEL = REGION_LABELS[3]

VMIN_GLOBAL = -3
VMAX_GLOBAL =  0


# ══════════════════════════════════════════════════════════════════════════════
# CACHE  — contact matrices stored as .npy alongside the dump
# Key encodes filepath + window + cutoff so stale caches are never reused.
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(dump_filepath, t_min, t_max, cutoff, n_frames):
    key = f"{dump_filepath}|{t_min}|{t_max}|{cutoff}|{n_frames}"
    h   = hashlib.md5(key.encode()).hexdigest()[:10]
    return Path(dump_filepath).parent / f".cmap_{h}.npy"


def load_cached(dump_filepath, t_min, t_max, cutoff, n_frames):
    p = _cache_path(dump_filepath, t_min, t_max, cutoff, n_frames)
    if p.exists():
        print(f"  [cache] loading {p.name}")
        return np.load(str(p))
    return None


def save_cached(mat, dump_filepath, t_min, t_max, cutoff, n_frames):
    p = _cache_path(dump_filepath, t_min, t_max, cutoff, n_frames)
    np.save(str(p), mat)
    print(f"  [cache] saved {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_dump_full(filepath, t_max=T_END):
    """Read dump frames with timestep <= t_max (never reads beyond T_END)."""
    timesteps, frames = [], []
    with open(filepath) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            current_ts = int(lines[i + 1].strip()); i += 2
            if current_ts > t_max:   # stop early — no data beyond ceiling
                break
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
    df = pd.read_csv(filepath, comment="#", names=["A", "U", "M", "Swi6", "Swi6M"])
    df["timestep"] = sim_start + np.arange(len(df), dtype=np.int64) * TYPES_STEP
    # Discard rows beyond T_END immediately — never parse beyond the ceiling
    df = df[df["timestep"] <= T_END].reset_index(drop=True)
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


def index_trajectory(filepath, t_max=T_END):
    """Index LAMMPS dump up to t_max — never reads frames beyond T_END."""
    path  = Path(filepath)
    index = []
    reading_ts = reading_natoms = False
    current_ts = current_offset = None
    with open(path, "rb") as fh:
        while True:
            offset = fh.tell()
            raw    = fh.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="replace").strip()
            if line == "ITEM: TIMESTEP":
                current_offset = offset
                reading_ts     = True
                reading_natoms = False
            elif reading_ts:
                current_ts = int(line)
                reading_ts = False
                if current_ts > t_max:
                    break   # done — no frames beyond ceiling
            elif line == "ITEM: NUMBER OF ATOMS":
                reading_natoms = True
            elif reading_natoms:
                index.append({"timestep": current_ts,
                               "offset":   current_offset,
                               "n_atoms":  int(line)})
                reading_natoms = False
    print(f"  Indexed {len(index)} frames (≤ t={t_max}) in {path.name}")
    return index


def _parse_frame_at(fh, frame_info):
    fh.readline()
    fh.readline()   # timestep value
    fh.readline()
    n_atoms = int(fh.readline().strip())
    fh.readline()
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
    return None, box, coords


def _contact_matrix(coords, polymer_ids, cutoff, box):
    ids  = sorted(pid for pid in polymer_ids if pid in coords)
    L    = box[:, 1] - box[:, 0]
    pos  = np.array([coords[i] for i in ids])
    diff = pos[:, None, :] - pos[None, :, :]
    diff -= np.round(diff / L) * L
    dist  = np.sqrt((diff ** 2).sum(-1))
    mat   = (dist <= cutoff).astype(np.float32)
    np.fill_diagonal(mat, 0)
    return mat, ids


def average_contact_map(dump_filepath, polymer_ids, cutoff, n_frames,
                         t_min=None, t_max=None):
    """
    Average contact map from a LAMMPS dump file.
    Results are cached as .npy files — recomputed only when parameters change.
    t_max is clamped to T_END so nothing beyond the ceiling is ever read.
    """
    # Clamp window to T_END
    t_max_safe = min(t_max, T_END) if t_max is not None else T_END
    t_min_safe = t_min if t_min is not None else T_EQ

    # Check cache first
    cached = load_cached(dump_filepath, t_min_safe, t_max_safe, cutoff, n_frames)
    if cached is not None:
        return cached, polymer_ids

    index = index_trajectory(dump_filepath, t_max=t_max_safe)
    index = [f for f in index if f["timestep"] >= t_min_safe]

    if not index:
        print("  [warn] No frames in time window — returning zeros")
        n = len(polymer_ids)
        return np.zeros((n, n)), polymer_ids

    n_pick = min(n_frames, len(index))
    pick   = sorted(set(np.round(
                np.linspace(0, len(index) - 1, n_pick)).astype(int).tolist()))

    avg_mat = None
    ids_out = None
    with open(dump_filepath, "r") as fh:
        for pos in tqdm(pick, desc=f"  Contact map ({len(pick)} frames)"):
            fh.seek(index[pos]["offset"])
            _, box, coords = _parse_frame_at(fh, index[pos])
            mat, ids = _contact_matrix(coords, polymer_ids, cutoff, box)
            avg_mat  = mat.astype(np.float64) if avg_mat is None else avg_mat + mat
            if ids_out is None:
                ids_out = ids

    avg_mat /= len(pick)
    print(f"  Contact map: {avg_mat.shape[0]} beads, cutoff={cutoff}σ")
    save_cached(avg_mat, dump_filepath, t_min_safe, t_max_safe, cutoff, n_frames)
    return avg_mat, ids_out


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def badge(ax, label, color, x=0.12, y=0.90):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=color,
                      edgecolor="white", linewidth=0.7, alpha=0.88))


def draw_hic(ax, avg_mat, vmin=VMIN_GLOBAL, vmax=VMAX_GLOBAL):
    log_mat = np.log10(avg_mat + 1e-3)
    im = ax.imshow(log_mat, cmap="Reds", vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest", aspect="equal")
    ax.set_xlabel("Nucleosome", fontsize=PRX_RC["axes.labelsize"])
    ax.set_ylabel("Nucleosome", fontsize=PRX_RC["axes.labelsize"])
    n   = avg_mat.shape[0]
    mid = n // 2
    ax.set_xticks([0, mid, n - 1], ["1", str(mid), str(n)])
    ax.set_yticks([0, mid, n - 1], ["1", str(mid), str(n)])
    return im


def draw_timeseries(ax, timesteps, types_df, t_meetings, panel_label,
                    t0_display=None, t1_display=None):
    """
    t_meetings : list of timestep values where vertical lines are drawn.
    Badges are placed between consecutive lines (and at start/end).
    t0_display / t1_display : x-axis limits (default: first/last timestep).
    """
    ts   = np.array(timesteps)[::TS_STRIDE]
    t0d  = t0_display if t0_display is not None else ts[0]
    t1d  = t1_display if t1_display is not None else ts[-1]

    for t, col in [(1, "A"), (2, "U"), (3, "M")]:
        vals = types_df[col].values[::TS_STRIDE]
        if t == 3:
            vals = vals - (len(TWO_POLYMERS) // 2)
        ax.plot(ts, vals, color=TYPE_COLORS[t], label=TYPE_LABELS[t], lw=1.2)

    for tm in t_meetings:
        ax.axvline(tm, color="black", lw=1.0, ls="--", alpha=0.65)

    ax.set_xlim(t0d, t1d)
    ax.set_xticks([0,0.5e8, 1e8])
    ax.set_xticklabels( ["0", "0.5", rf"$1 \times 10^5$"])
    ax.set_ylabel("Nucleosomal\ntype count", fontsize=PRX_RC["axes.labelsize"])
    ax.set_xlabel(r"Time ($\tau_{LJ}$)", fontsize=PRX_RC["axes.labelsize"], labelpad=3)
    ax.grid(alpha=0.25)

    # Place a badge in the centre of each window
    boundaries = [t0d] + list(t_meetings) + [t1d]
    span = t1d - t0d
    n_regions = len(boundaries) - 1
    for k in range(n_regions):
        mid_t = (boundaries[k] + boundaries[k + 1]) / 2
        x_ax  = (mid_t - t0d) / span
        col   = REGION_COLORS[k] if n_regions > 2 else (PRE_COLOR if k == 0 else POST_COLOR)
        lbl   = REGION_LABELS[k] if n_regions > 2 else (POST_LABEL if k == 1 else PRE_LABEL)
        if lbl == "Post-sw":
            badge(ax, lbl, col, x=x_ax + 0.03, y=0.45)
        elif lbl == "Switch":
            badge(ax, lbl, col, x=x_ax + 0.01, y=1.02)
        else:
            badge(ax, lbl, col, x=x_ax + 0.02, y=0.90)

    ax.text(-0.10, 1.04, panel_label, transform=ax.transAxes,
            fontsize=PRX_RC["axes.titlesize"],
            va="bottom", ha="right")


def draw_contact_row(fig, subplot_spec, mats, badge_indices, panel_label):
    """
    mats         : list of averaged contact matrices (2 or 4)
    badge_indices: list of ints indexing into REGION_COLORS/LABELS per panel
    """
    n_maps = len(mats)
    inner  = GridSpecFromSubplotSpec(
        1, n_maps + 1, subplot_spec=subplot_spec,
        width_ratios=[1] * n_maps + [0.07],
        wspace=0.28)

    last_im = None
    for j, (mat, bi) in enumerate(zip(mats, badge_indices)):
        ax = fig.add_subplot(inner[0, j])
        last_im = draw_hic(ax, mat)
        badge(ax, REGION_LABELS[bi], REGION_COLORS[bi], x=0.12, y=0.75)
        if j > 0:
            ax.set_ylabel("")
        if j == 0:
            ax.text(-0.22, 1.08, panel_label, transform=ax.transAxes,
                    fontsize=PRX_RC["axes.titlesize"], 
                    va="bottom", ha="right")

    # Shared colourbar
    ax_cb = fig.add_subplot(inner[0, n_maps])
    plt.colorbar(last_im, cax=ax_cb,
                 label=r"$\log_{10}(P_\mathrm{contact})$",
                 ticks=[VMIN_GLOBAL, -2, -1, VMAX_GLOBAL])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(cutoff=CONTACT_CUTOFF, n_avg=N_FRAMES_AVG, outfile=None):

    # ── Load timeseries (capped at T_END) ─────────────────────────────────
    print("Loading dumps (capped at T_END) …")
    ts200,  _ = parse_dump_full(swi6_200_dump,  t_max=T_END)
    ts600,  _ = parse_dump_full(swi6_600_dump,  t_max=T_END)
    ts1000, _ = parse_dump_full(swi6_1000_dump, t_max=T_END)
    print(f"  200: {len(ts200)} frames  |  600: {len(ts600)} frames  |  1000: {len(ts1000)} frames")

    types200  = parse_types(swi6_200_types,  ts200)
    types600  = parse_types(swi6_600_types,  ts600)
    types1000 = parse_types(swi6_1000_types, ts1000)

    # ── Contact maps (cached, t_max always ≤ T_END) ───────────────────────
    print("Computing / loading contact maps …")

    mat_pre_200,  _ = average_contact_map(swi6_200_dump, TWO_POLYMERS, cutoff, n_avg,
                                           t_min=T_EQ,         t_max=T_MEETING_200)
    mat_post_200, _ = average_contact_map(swi6_200_dump, TWO_POLYMERS, cutoff, n_avg,
                                           t_min=T_MEETING_200, t_max=T_END)

    mat_pre_600,  _ = average_contact_map(swi6_600_dump, TWO_POLYMERS, cutoff, n_avg,
                                           t_min=T_EQ,         t_max=T_MEETING_600)
    mat_post_600, _ = average_contact_map(swi6_600_dump, TWO_POLYMERS, cutoff, n_avg,
                                           t_min=T_MEETING_600, t_max=T_END)

    # swi6=1000: 4 windows
    mat_1000_pre,      _ = average_contact_map(swi6_1000_dump, TWO_POLYMERS, cutoff, n_avg,
                                                t_min=T_EQ,               t_max=T_SWITCH_START_1000)
    mat_1000_switch,   _ = average_contact_map(swi6_1000_dump, TWO_POLYMERS, cutoff, n_avg,
                                                t_min=T_SWITCH_START_1000, t_max=T_SWITCH_END_1000)
    mat_1000_postsw,   _ = average_contact_map(swi6_1000_dump, TWO_POLYMERS, cutoff, n_avg,
                                                t_min=T_SWITCH_END_1000,   t_max=T_MEETING_1000)
    mat_1000_post,     _ = average_contact_map(swi6_1000_dump, TWO_POLYMERS, cutoff, n_avg,
                                                t_min=T_MEETING_1000,      t_max=T_END)

    # ── Build figure ───────────────────────────────────────────────────────
    # 6 rows: ts200 / hic200 / ts600 / hic600 / ts1000 / hic1000
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
    outer = GridSpec(
        6, 1, figure=fig,
        height_ratios=[1.0, 0.85, 1.0, 0.85, 1.0, 0.85],
        hspace=0.58,
    )

    # ── Swi6=200 ──────────────────────────────────────────────────────────
    ax_ts = fig.add_subplot(outer[0])
    draw_timeseries(ax_ts, ts200, types200,
                    t_meetings=[T_MEETING_200],
                    panel_label="(a)",
                    t1_display=T_END)
    draw_contact_row(fig, outer[1],
                     mats=[mat_pre_200, mat_post_200],
                     badge_indices=[0, 3],
                     panel_label="(b)")

    # ── Swi6=600 ──────────────────────────────────────────────────────────
    ax_ts = fig.add_subplot(outer[2])
    draw_timeseries(ax_ts, ts600, types600,
                    t_meetings=[T_MEETING_600],
                    panel_label="(c)",
                    t1_display=T_END)
    draw_contact_row(fig, outer[3],
                     mats=[mat_pre_600, mat_post_600],
                     badge_indices=[0, 3],
                     panel_label="(d)")

    # ── Swi6=1000  (4 contact maps) ────────────────────────────────────────
    ax_ts = fig.add_subplot(outer[4])
    draw_timeseries(ax_ts, ts1000, types1000,
                    t_meetings=[T_SWITCH_START_1000, T_SWITCH_END_1000, T_MEETING_1000],
                    panel_label="(e)",
                    t1_display=T_END)
    draw_contact_row(fig, outer[5],
                     mats=[mat_1000_pre, mat_1000_switch,
                            mat_1000_postsw, mat_1000_post],
                     badge_indices=[0, 1, 2, 3],
                     panel_label="(f)")

    if outfile:
        fig.savefig(outfile, dpi=PRX_RC["savefig.dpi"], bbox_inches="tight")
        print(f"Saved → {outfile}")
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",    default=None)
    ap.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF)
    ap.add_argument("--n-avg",  type=int,   default=N_FRAMES_AVG)
    args = ap.parse_args()
    make_figure(cutoff=args.cutoff, n_avg=args.n_avg, outfile=args.out)