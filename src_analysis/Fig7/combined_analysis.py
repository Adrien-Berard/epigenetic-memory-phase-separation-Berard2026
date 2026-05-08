import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from pathlib import Path
import pickle
import hashlib
from tqdm import tqdm
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
#  >>>  EDIT THIS BLOCK TO CONFIGURE THE SCRIPT  <<<
# =============================================================================

# -- Trajectory dump files (up to 4) -----------------------------------------
DUMP_FILES = [
    '/home/adrien/19_03_Sim/WihtoutEpe1/WithoutBE/dump.lammpstrj',
    '/home/adrien/19_03_Sim/WithEpe1/WithoutBE/dump.lammpstrj',
    '/home/adrien/19_03_Sim/WithEpe1/2or3Nuc/2_nucleation_sites/dump.lammpstrj',
    '/home/adrien/19_03_Sim/WithEpe1/2or3Nuc/3_nucleation_sites/dump.lammpstrj',
]

# -- id_and_type.dat files (one per simulation, same order as DUMP_FILES) -----
#    Format expected: one atom per line → "id  type"
#    These define which bead IDs belong to the polymer and their fixed types.
#    Set to None to fall back to --n-beads sequential IDs from the dump itself.
KYMO_CHIP_FILES = [
    '/home/adrien/19_03_Sim/WihtoutEpe1/WithoutBE/id_and_type.dat',
    '/home/adrien/19_03_Sim/WithEpe1/WithoutBE/id_and_type.dat',
    '/home/adrien/19_03_Sim/WithEpe1/2or3Nuc/2_nucleation_sites/id_and_type.dat',
    '/home/adrien/19_03_Sim/WithEpe1/2or3Nuc/3_nucleation_sites/id_and_type.dat',
]

# -- Snapshot image files for panels (a), (b), (e) ----------------------------
#    Provide paths to PNG/JPG images, or set to None to leave panel empty.
SNAPSHOT_A = '/home/adrien/19_03_Sim/Figure6a.pdf'   # e.g. '/home/adrien/snapshots/model1.png'
SNAPSHOT_B = '/home/adrien/19_03_Sim/Figure6b.pdf'   # e.g. '/home/adrien/snapshots/model2.png'
SNAPSHOT_E = '/home/adrien/19_03_Sim/epe1_snapshots.pdf'   # e.g. '/home/adrien/snapshots/overview.png'

# -- Panel column labels -------------------------------------------------------
LABELS = [rf"$epe1 \Delta $", "cenH", "2cenH", "3cenH"]

# -- Physics / sampling --------------------------------------------------------
N_BEADS          = 500      # fallback if KYMO_CHIP_FILES entries are None
CONTACT_DIST     = 3.0      # distance cutoff for HiC contact (simulation units)
MAX_KYMO_FRAMES  = 2001     # kymo uses every frame from 0 up to this index

# -- Output --------------------------------------------------------------------
OUTPUT_FILE = "final_figure.pdf"

# =============================================================================
#  END OF CONFIG  —  no need to edit below this line
# =============================================================================

# -- PLOTTING -----------------------------------------------------------------
# Row 0 (30%):   (a) Model snapshot 1 (50%) | (b) Model snapshot 2 (50%)
# Row 1 (55%):   (c) kymo chip 1|2|3|4      | (e) spans row 1–2 (50%)
# Row 2 (15%):   (d) HiC contact 1|2|3|4   | (e continues)
# ---------------------------------------------------------------------------

A4_WIDTH  = 7.5
A4_HEIGHT = 5

# Bead-type → color
COLORS = {1: "#4e79a7", 2: "#f0c040", 3: "#e15759"}

FRUIT_PUNCH_HIC = mcolors.LinearSegmentedColormap.from_list(
    "hic", ["#ffffff", "#ffe0ec", "#ff4d7d", "#b5001f", "#3d0010"]
)

# ---------------------------------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family"         : "serif",
    "font.serif"          : ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size"           : 6,
    "axes.labelsize"      : 6,
    "axes.titlesize"      : 6,
    "legend.fontsize"     : 6,
    "xtick.labelsize"     : 5,
    "ytick.labelsize"     : 5,
    "xtick.direction"     : "in",
    "ytick.direction"     : "in",
    "xtick.top"           : True,
    "ytick.right"         : True,
    "xtick.minor.visible" : True,
    "ytick.minor.visible" : True,
    "xtick.minor.top"     : True,
    "ytick.minor.right"   : True,
    "xtick.major.size"    : 5,
    "ytick.major.size"    : 5,
    "xtick.minor.size"    : 3,
    "ytick.minor.size"    : 3,
    "lines.linewidth"     : 1.5,
    "figure.dpi"          : 150,
})

# ---------------------------------------------------------------------------
# CACHING
# ---------------------------------------------------------------------------

CACHE_VERSION = "v3"   # bump this whenever load_frames logic changes

def _cache_path(file: str, suffix: str) -> Path:
    h = hashlib.md5(file.encode()).hexdigest()
    return Path(f".cache_{h}_{suffix}_{CACHE_VERSION}.pkl")


def _load_cache(path: Path):
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return None


def _save_cache(path: Path, data):
    with open(path, "wb") as fh:
        pickle.dump(data, fh)

# ---------------------------------------------------------------------------
# ID-AND-TYPE FILE READER
# ---------------------------------------------------------------------------

def load_id_and_type(filepath: str) -> tuple[set, dict]:
    """
    Read an id_and_type.dat file which is a LAMMPS dump containing
    only 'id' and 'type' columns (no coordinates).
    Reads only the first frame — types are fixed throughout the simulation.
    Returns:
        polymer_ids  – set of atom IDs
        fixed_types  – dict {atom_id: bead_type}
    """
    polymer_ids = set()
    fixed_types = {}

    with open(filepath) as fh:
        in_atoms = False
        idx_id   = 0
        idx_type = 1

        for line in fh:
            line = line.strip()
            if not line:
                continue

            if line.startswith("ITEM: ATOMS"):
                cols     = line.split()[2:]
                idx_id   = cols.index("id")   if "id"   in cols else 0
                idx_type = cols.index("type") if "type" in cols else 1
                in_atoms = True
                continue

            if line.startswith("ITEM:"):
                in_atoms = False
                continue

            if not in_atoms:
                continue

            parts = line.split()
            if len(parts) < max(idx_id, idx_type) + 1:
                continue

            aid   = int(parts[idx_id])
            atype = int(parts[idx_type])
            polymer_ids.add(aid)
            fixed_types[aid] = atype

    return polymer_ids, fixed_types

# ---------------------------------------------------------------------------
# TRAJECTORY INDEXING  (single binary handle throughout)
# ---------------------------------------------------------------------------

def index_trajectory(filepath):
    """
    Single fast pass over the file. For each frame records:
        timestep   : int
        offset     : byte position of the 'ITEM: TIMESTEP' line
        n_atoms    : number of atoms in this frame

    Also prints a summary: total frames, timestep interval, time span.

    The number of lines per frame is:
        1  ITEM: TIMESTEP
        1  <timestep value>
        1  ITEM: NUMBER OF ATOMS
        1  <n_atoms value>
        1  ITEM: BOX BOUNDS ...
        3  <box lines>
        1  ITEM: ATOMS ...
        n_atoms  <atom lines>
      = 9 + n_atoms  lines total per frame
    We do NOT need to count these manually — we just seek by byte offset.
    """
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {filepath}")

    index = []       # list of {"timestep": int, "offset": int, "n_atoms": int}
    reading_ts     = False
    reading_natoms = False
    current_ts     = None
    current_offset = None

    print(f"[1/3] Scanning trajectory index: {filepath}")
    file_size = path.stat().st_size

    with open(path, "rb") as fh:   # binary for reliable tell()
        with tqdm(total=file_size, desc="  Indexing",
                  unit="B", unit_scale=True, dynamic_ncols=True) as pbar:
            while True:
                offset = fh.tell()
                raw = fh.readline()
                if not raw:
                    break
                pbar.update(len(raw))
                line = raw.decode("ascii", errors="replace").strip()

                if line == "ITEM: TIMESTEP":
                    current_offset = offset
                    reading_ts = True
                    reading_natoms = False
                    continue

                if reading_ts:
                    current_ts = int(line)
                    reading_ts = False
                    continue

                if line == "ITEM: NUMBER OF ATOMS":
                    reading_natoms = True
                    continue

                if reading_natoms:
                    n_atoms = int(line)
                    reading_natoms = False
                    index.append({
                        "timestep": current_ts,
                        "offset":   current_offset,
                        "n_atoms":  n_atoms,
                    })

    if not index:
        sys.exit("[ERROR] No frames found in trajectory file.")

    # ── Summary
    n_total = len(index)
    ts_list = [f["timestep"] for f in index]
    dt = None
    if n_total > 1:
        diffs = [ts_list[i+1] - ts_list[i] for i in range(min(10, n_total - 1))]
        if len(set(diffs)) == 1:
            dt = diffs[0]
        else:
            dt = int(np.median(diffs))

    print(f"\n  Trajectory summary:")
    print(f"    Total frames   : {n_total}")
    print(f"    Atoms/frame    : {index[0]['n_atoms']}")
    if dt is not None:
        print(f"    Timestep freq  : every {dt} steps")
    print(f"    First timestep : {ts_list[0]}")
    print(f"    Last  timestep : {ts_list[-1]}")
    if dt and dt > 0:
        print(f"    Time span      : {ts_list[-1] - ts_list[0]} steps  "
              f"({(ts_list[-1] - ts_list[0]) // dt} intervals)")

    return index

# ---------------------------------------------------------------------------
# FRAME PARSING  (reuses the same binary handle passed from load_frames)
# ---------------------------------------------------------------------------


def _detect_columns(header_line: bytes):
    """
    Parse "ITEM: ATOMS id type xs ys zs …" and return column indices for
    id, type, the three coordinate fields, and a boolean `scaled`.
    Accepts x/xs/xu (unscaled / scaled / unwrapped).
    """
    tokens = header_line.decode().strip().split()
    cols   = tokens[2:]   # drop "ITEM:" and "ATOMS"

    def find(names):
        for n in names:
            if n in cols:
                return cols.index(n), n
        raise ValueError(f"None of {names} found in ATOMS header: {cols}")

    idx_id,   _    = find(["id"])
    idx_type, _    = find(["type"])
    idx_x,    nx   = find(["x", "xs", "xu"])
    idx_y,    _    = find(["y", "ys", "yu"])
    idx_z,    _    = find(["z", "zs", "zu"])
    scaled         = nx == "xs"          
    return idx_id, idx_type, idx_x, idx_y, idx_z, scaled


_FRAME_DIAG_DONE = False   # print diagnostics only for the very first frame

def parse_frame(fh, frame_info: dict):
    """
    Parse one frame from an already-open **binary** file handle.
    Returns (timestep, box [3×2], coords {id: xyz}, types {id: int}).

    Handles:
    - 2-column box lines  (xlo xhi)
    - 3-column box lines  (xlo xhi xy/xz/yz tilt — triclinic; tilt ignored)
    - unscaled coords     (x  y  z  — already in Å / σ)
    - scaled   coords     (xs ys zs — in [0,1], multiplied by L on read)
    """
    global _FRAME_DIAG_DONE

    fh.seek(frame_info["offset"])

    fh.readline()                        # "ITEM: TIMESTEP"
    ts      = int(fh.readline())
    fh.readline()                        # "ITEM: NUMBER OF ATOMS"
    n_atoms = int(fh.readline())
    box_header = fh.readline().decode().strip()   # "ITEM: BOX BOUNDS pp pp pp"

    # Read 3 box lines; each may have 2 (orthogonal) or 3 (triclinic) values
    raw_box = []
    for _ in range(3):
        vals = list(map(float, fh.readline().split()))
        raw_box.append(vals[:2])          # keep only xlo/xhi, discard tilt
    box = np.array(raw_box)              # (3, 2)
    L   = box[:, 1] - box[:, 0]         # (3,)

    atoms_header = fh.readline()         # "ITEM: ATOMS id type x y z …"
    idx_id, idx_type, idx_x, idx_y, idx_z, scaled = _detect_columns(atoms_header)

    coords = {}
    types  = {}
    for _ in range(n_atoms):
        parts = fh.readline().split()
        aid   = int(parts[idx_id])
        atype = int(parts[idx_type])
        xyz   = np.array([float(parts[idx_x]),
                           float(parts[idx_y]),
                           float(parts[idx_z])])
        if scaled:
            xyz = xyz * L + box[:, 0]    # scaled → real coordinates
        coords[aid] = xyz
        types[aid]  = atype

    if not _FRAME_DIAG_DONE:
        _FRAME_DIAG_DONE = True
        sample_ids = sorted(coords.keys())[:5]
        print(f"\n  [parse_frame diag] ts={ts}")
        print(f"    box_header : {box_header}")
        print(f"    box        : {box.tolist()}")
        print(f"    L          : {L.tolist()}")
        print(f"    scaled     : {scaled}")
        print(f"    first 5 coords (after unscaling if needed):")
        for sid in sample_ids:
            print(f"      id={sid}  xyz={coords[sid].tolist()}")
        # compute a few distances between consecutive polymer beads
        consec = sorted(coords.keys())[:10]
        dists = []
        for a, b in zip(consec, consec[1:]):
            d = coords[b] - coords[a]
            d -= np.round(d / L) * L
            dists.append(float(np.sqrt((d**2).sum())))
        print(f"    consecutive-bead distances (first 9): "
              f"{[f'{v:.3f}' for v in dists]}")

    return ts, box, coords, types

# ---------------------------------------------------------------------------
# CONTACT MATRIX
# ---------------------------------------------------------------------------

def contact_matrix(coords: dict, ids: list, cutoff: float, box: np.ndarray) -> np.ndarray:
    """
    Compute a binary contact matrix with periodic minimum-image convention.
    ids must be pre-sorted and filtered to polymer beads present in coords.
    """
    L    = box[:, 1] - box[:, 0]                              # (3,)
    pos  = np.array([coords[i] for i in ids])                 # (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]      # (N, N, 3)
    diff -= np.round(diff / L) * L                            # minimum image
    dist = np.sqrt((diff ** 2).sum(axis=-1))                  # (N, N)
    mat  = (dist <= cutoff).astype(np.float32)
    np.fill_diagonal(mat, 0)
    return mat

# ---------------------------------------------------------------------------
# FRAME LOADING  (one binary handle, cached)
# ---------------------------------------------------------------------------

def load_frames(traj_file: str,
                index: list[dict],
                kymo_indices: list[int],
                hic_indices: list[int],
                polymer_ids: set,
                cutoff: float,
                fixed_types: dict | None = None):
    """
    Parse frames and return (hic_avg, kymo_matrix, bead_ids).

    kymo_indices  – frames used for the kymograph (high temporal resolution)
    hic_indices   – frames used for HiC averaging (can be a different, sparser set)

    Both sets are read in a single pass over the file.
    """
    cpath  = _cache_path(traj_file, "frames")
    cached = _load_cache(cpath)
    if cached is not None:
        print(f"  ↳ using cache: {cpath}")
        return cached

    all_indices = sorted(set(kymo_indices) | set(hic_indices))
    kymo_set    = set(kymo_indices)
    hic_set     = set(hic_indices)

    contact_mats = []
    kymo_rows    = []
    ids_out      = None

    with open(traj_file, "rb") as fh:
        for fi in tqdm(all_indices, desc=f"  parsing {Path(traj_file).name}"):
            ts, box, coords, dump_types = parse_frame(fh, index[fi])

            ids = sorted(k for k in coords if k in polymer_ids)
            if ids_out is None:
                ids_out = ids

            # HiC: accumulate contact matrix for averaging
            if fi in hic_set:
                contact_mats.append(contact_matrix(coords, ids, cutoff, box))

            # Kymo: record bead types for this frame
            if fi in kymo_set:
                src = dump_types #fixed_types if fixed_types is not None else dump_types
                kymo_rows.append({aid: src[aid] for aid in ids if aid in src})

    hic_avg = np.mean(contact_mats, axis=0)

    # Build (T_kymo × N) integer type matrix
    kymo      = np.zeros((len(kymo_rows), len(ids_out)), dtype=np.int8)
    id_to_col = {aid: col for col, aid in enumerate(ids_out)}
    for t, tdict in enumerate(kymo_rows):
        for aid, atype in tdict.items():
            if aid in id_to_col:
                kymo[t, id_to_col[aid]] = atype

    result = (hic_avg, kymo, ids_out)
    _save_cache(cpath, result)
    return result


def run_pipeline(traj_file: str,
                 polymer_ids: set,
                 cutoff: float,
                 max_kymo_frames: int = 2001,
                 fixed_types: dict | None = None):
    print(f"\nProcessing: {traj_file}")
    index = index_trajectory(traj_file)
    n     = len(index)
    print(f"  {n} frames found")

    # Kymo: every frame from 0 up to max_kymo_frames (no skipping)
    kymo_indices = list(range(min(n, max_kymo_frames)))

    # HiC: all available frames
    hic_indices  = list(range(min(n, max_kymo_frames)))

    print(f"  kymo frames : {len(kymo_indices)}  (frames 0–{kymo_indices[-1]})")
    print(f"  HiC  frames : {len(hic_indices)}  (all)")

    return load_frames(traj_file, index,
                       kymo_indices, hic_indices,
                       polymer_ids, cutoff, fixed_types)

# ---------------------------------------------------------------------------
# FIGURE LAYOUT
# ---------------------------------------------------------------------------
# Left block (50 % width), 3 rows per column:
#   row 0 (~55 %): kymograph
#   row 1 (~10 %): ChIP strip
#   row 2 (~35 %): HiC contact map
# Right block (50 % width), full height: panels a, b (top) + e (bottom)
# ---------------------------------------------------------------------------

def make_figure(n_cols: int = 4):
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), dpi=2000)

    outer = GridSpec(2, 1,
                     height_ratios=[0.28, 0.72],
                     hspace=0.08,
                     figure=fig)

    # Top row: a (45%) | b (45%) | padding (10%)
    gs_top = outer[0].subgridspec(1, 3, width_ratios=[0.45, 0.45, 0.10], wspace=0.08)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    # gs_top[0, 2] left empty for alignment

    # Bottom row: left data block (50%) | right panel e (50%)
    gs_bot = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=-0.2)

    gs_left = gs_bot[0].subgridspec(3, n_cols + 1,
                                    height_ratios=[0.55, 0.10, 0.35],
                                    width_ratios=[1, 1, 1, 1, 0.08],
                                    hspace=0.12, wspace=0.06)
    ax_kymo = [fig.add_subplot(gs_left[0, i]) for i in range(n_cols)]
    ax_chip = [fig.add_subplot(gs_left[1, i]) for i in range(n_cols)]
    ax_hic  = [fig.add_subplot(gs_left[2, i]) for i in range(n_cols)]
    ax_cbar = fig.add_subplot(gs_left[2, n_cols])

    ax_e = fig.add_subplot(gs_bot[1])
    pos = ax_e.get_position()
    ax_e.set_position([
        pos.x0,
        pos.y0 - 0.03,
        pos.width,
        pos.height * 1.05
    ])

    return fig, ax_a, ax_b, ax_kymo, ax_chip, ax_hic, ax_e, ax_cbar

# ---------------------------------------------------------------------------
# PANEL HELPERS
# ---------------------------------------------------------------------------

def panel_label(ax,x ,s: str):
    ax.text(x, 1.06, s,
            transform=ax.transAxes,
            fontsize=7, va="top", ha="right")


def draw_snapshot_panel(ax, img, placeholder_text="snapshot\n(not provided)"):
    if img is not None:
        ax.imshow(img)
    else:
        ax.set_facecolor("#dddddd")
        ax.text(0.5, 0.5, placeholder_text,
                ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "bottom", "left", "right"]].set_visible(False)


def plot_kymo_panel(ax, kymo: np.ndarray, cmap, norm,
                    show_ylabel: bool = True, n_frames: int = None):
    """
    kymo shape: (T, N)  — time on y-axis, bead position on x-axis.
    """
    T = kymo.shape[0] if n_frames is None else n_frames
    ax.imshow(kymo, cmap=cmap, norm=norm,
              aspect="auto", origin="upper", interpolation="none", rasterized=True)
    ax.set_xticks([])
    

    if show_ylabel:
        tick_pos = [0, 1000, 1999]
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(["0",'1e4',"2e4"])
        ax.set_ylabel(r"Time ($\tau_{LJ}$)")
        ax.yaxis.set_label_coords(-0.4,0.5)
        ax.yaxis.tick_left()
        ax.tick_params(axis='y', left=True, right=False)
    else:
        ax.tick_params(axis='y', which='both',left=False, labelleft=False,right=False,labelright=False)


    ax.spines[["top", "right"]].set_visible(False)


def plot_chip_panel(ax, kymo: np.ndarray, n_beads: int,
                    show_ylabel: bool = True, show_xlabel: bool = False):
    """
    ChIP-seq strip: fraction of time each bead spends as type-3 (H3K9me3).
    kymo shape: (T, N).
    """
    binary      = (kymo == 3).astype(float)
    mean_signal = binary.mean(axis=0)          # (N,) — average over time
    x           = np.arange(n_beads)
    color       = COLORS[3]

    ax.plot(x, mean_signal, color=color, lw=1.2)
    ax.fill_between(x, mean_signal, color=color, alpha=0.35)

    ax.set_xlim(0, n_beads - 1)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    xticks       = [0, 249,499]
    xtick_labels = [str(t + 1) for t in xticks]
    ax.set_xticks(xticks)
    ax.xaxis.tick_bottom()


    if show_xlabel:
        ax.set_xticklabels(xtick_labels)
        
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel("H3K9me3")
        ax.yaxis.set_tick_params()
        ax.set_yticks([0,1])
        ax.set_yticklabels(['0%','100%'])
        ax.yaxis.set_label_coords(-0.4,0.5)
        ax.yaxis.tick_left()
        ax.tick_params(axis='y', left=True, right=False)
    else:
        ax.tick_params(axis='y',which='both', left=False, labelleft=False,right=False,labelright=False)

def set_exact_pixel_size(ax, n_pixels, dpi=2000):
    fig = ax.figure

    # size of one pixel in inches
    size_in = n_pixels / dpi

    # get figure size
    fig_w, fig_h = fig.get_size_inches()

    # convert desired axes size into figure fraction
    bbox = ax.get_position()
    ax_width_frac  = size_in / fig_w
    ax_height_frac = size_in / fig_h

    ax.set_position([
        bbox.x0,
        bbox.y0,
        ax_width_frac,
        ax_height_frac
    ])

def plot_hic_panel(ax, mat: np.ndarray, show_ticks: bool = False,show_yticks:bool=False,cax=None):
    """
    Log₁₀ contact map — one pixel per monomer pair, no interpolation.
    vmin/vmax clipped to the 2nd–98th percentile of off-diagonal values.
    """
    log_mat  = np.log10(mat + 1e-3)
    off_diag = log_mat[~np.eye(log_mat.shape[0], dtype=bool)]
    # set_exact_pixel_size(ax, log_mat.shape[0])
    
    vmin, vmax =  -3.1,0.5 # np.percentile(off_diag, [2, 98])
    print(np.percentile(off_diag, [2, 98]))

    im = ax.imshow(log_mat, cmap='Reds',
              vmin=vmin, vmax=vmax,
              origin="lower",
              interpolation="none")   # exact 1 pixel per monomer pair
    # ax.set_aspect('equal')  # critical
    # print(ax.bbox.width, ax.bbox.height)
    ticks  = [0, 249]
    labels = [str(t + 1) for t in ticks]
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    if cax is not None:
        ax.get_figure().colorbar(im, cax=cax, label=r'$\log_{10}(P_{contact})$')

    if show_yticks:
        ax.set_yticks([0,249,499])
        ax.set_yticklabels([1,250,500])
        ax.set_ylabel("Nucleosome\nposition", rotation=90)
        ax.yaxis.set_label_coords(-0.25,0.5)
    else:
        ax.set_yticklabels([])


    N      = mat.shape[0]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Nucleosome\nposition",fontsize=5.75)


# ---------------------------------------------------------------------------
# SNAPSHOT LOADER  (PNG / JPG / PDF)
# ---------------------------------------------------------------------------

def load_snapshot(path: str | None) -> np.ndarray | None:
    """
    Load a snapshot image from a PNG, JPG, or PDF file.
    For PDFs the first page is rasterised to a numpy RGBA array.
    Returns None if path is None or the file does not exist.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f" snapshot not found: {path}")
        return None

    if p.suffix.lower() == ".pdf":
        try:
            import fitz                          # PyMuPDF  (pip install pymupdf)
            doc  = fitz.open(str(p))
            page = doc[0]
            # render at 150 dpi (mat scale=150/72 ≈ 2.08)
            mat  = fitz.Matrix(2000 / 72, 2000 / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            arr  = np.frombuffer(pix.samples, dtype=np.uint8)
            return arr.reshape(pix.height, pix.width, 3)
        except ImportError:
            pass
        try:
            from pdf2image import convert_from_path   # pip install pdf2image
            imgs = convert_from_path(str(p), dpi=2000, first_page=1, last_page=1)
            return np.array(imgs[0])
        except ImportError:
            print(f"  PDF snapshot needs PyMuPDF or pdf2image: pip install pymupdf")
            return None

    return plt.imread(str(p))

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    dump_files = [f for f in DUMP_FILES if f is not None]
    kymo_files = KYMO_CHIP_FILES
    n          = min(len(dump_files), 4)

    hic_mats  = []
    kymo_mats = []

    for i in range(n):
        kf = kymo_files[i] if (kymo_files and i < len(kymo_files)) else None
        if kf is not None:
            polymer_ids, fixed_types = load_id_and_type(kf)
            print(f"  id_and_type: {len(polymer_ids)} beads loaded from {kf}")
        else:
            polymer_ids = set(range(1, N_BEADS + 1))
            fixed_types = None

        hic_avg, kymo, _ = run_pipeline(
            dump_files[i], polymer_ids, CONTACT_DIST,
            max_kymo_frames=MAX_KYMO_FRAMES,
            fixed_types=fixed_types,
        )
        hic_mats.append(hic_avg)
        kymo_mats.append(kymo)

    # ── colormaps ─────────────────────────────────────────────────────────
    type_values = sorted(COLORS)
    cmap_kymo   = ListedColormap([COLORS[k] for k in type_values])
    bounds      = [v - 0.5 for v in type_values] + [type_values[-1] + 0.5]
    norm_kymo   = BoundaryNorm(bounds, cmap_kymo.N)

    n_beads  = hic_mats[0].shape[0]
    n_frames = kymo_mats[0].shape[0]

    # ── build figure ──────────────────────────────────────────────────────
    fig, ax_a, ax_b, ax_kymo, ax_chip, ax_hic, ax_e, ax_cbar = make_figure(n_cols=n)

    draw_snapshot_panel(ax_a, load_snapshot(SNAPSHOT_A))
    draw_snapshot_panel(ax_b, load_snapshot(SNAPSHOT_B))
    draw_snapshot_panel(ax_e, load_snapshot(SNAPSHOT_E))

    for i in range(n):
        lbl = LABELS[i] if i < len(LABELS) else f"sim {i+1}"

        if i > 0:
            ax_kymo[i].sharey(ax_kymo[0])
        plot_kymo_panel(ax_kymo[i], kymo_mats[i], cmap_kymo, norm_kymo,
                        show_ylabel=(i == 0), n_frames=n_frames)
        ax_kymo[i].set_title(lbl, pad=3,style='italic')

        if i > 0:
            ax_chip[i].sharey(ax_chip[0])
        plot_chip_panel(ax_chip[i], kymo_mats[i], n_beads,
                        show_ylabel=(i == 0), show_xlabel=False)

        plot_hic_panel(ax_hic[i], hic_mats[i], show_ticks=(i==0), show_yticks=(i==0),
               cax=ax_cbar if i == 3 else None)

    panel_label(ax_a,0,       "a)")
    panel_label(ax_b, 0,     "b)")
    panel_label(ax_kymo[0],-0.6,  "c)")
    panel_label(ax_hic[0], -0.6,  "d)")
    panel_label(ax_e,  0,     "e)")


    plt.savefig(OUTPUT_FILE, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()