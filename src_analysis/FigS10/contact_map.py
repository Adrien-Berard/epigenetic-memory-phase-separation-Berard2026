"""
LAMMPS Polymer Contact Map Analysis
====================================
Outputs:
  01_contact_evolution.png  — grid of N_FRAMES_EVOLUTION sampled frames
  02_avg_contact_map.png    — time-averaged contact map
  03_std_contact_map.png    — std-deviation map
  04_hic_style_map.png      — Hi-C log-scaled map
  05_contact_scaling.png    — power-law scaling
  06_contact_evolution.html — interactive HTML slider (open in any browser)

Parsing strategy
----------------
Step 1 — index_trajectory()
    Fast single pass: reads ONLY "ITEM: TIMESTEP" and "ITEM: NUMBER OF ATOMS"
    lines. Records (timestep, byte_offset, n_atoms) for every frame, then
    computes the fixed line-skip needed to jump to the next frame.
    Reports total frame count, timestep frequency, and trajectory duration.

Step 2 — pick_frame_indices()
    Chooses N_FRAMES_AVG and N_FRAMES_EVOLUTION indices that are evenly
    distributed across the full trajectory using the index built above.

Step 3 — load_frames()
    Seeks directly to the byte offsets of only the selected frames; parses
    atom coordinates only for those frames (no wasted reading).

Dependencies:
    pip install numpy matplotlib tqdm
"""

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# ─────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────
TRAJ_FILE          = "dump.lammpstrj"
POLYMER_IDS        = set(range(1, 501))   # atom IDs 1–80
CONTACT_DIST       = 2.0                 # contact cutoff in σ
N_FRAMES_EVOLUTION = 50                  # frames for grid + HTML slider
N_FRAMES_AVG       = 100                 # frames for time-average / std
OUTPUT_DIR         = Path(".")
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
# FRUIT PUNCH COLORMAPS
# ══════════════════════════════════════════════
FRUIT_PUNCH = mcolors.LinearSegmentedColormap.from_list(
    "fruit_punch", ["#ffffff", "#ffd6e0", "#ff6b9d", "#c0392b", "#6b0000"])
FRUIT_PUNCH_STD = mcolors.LinearSegmentedColormap.from_list(
    "fruit_punch_std", ["#f8f0fb", "#e8c8f0", "#ff6b9d", "#c0392b", "#6b0000"])
FRUIT_PUNCH_HIC = mcolors.LinearSegmentedColormap.from_list(
    "fruit_punch_hic", ["#ffffff", "#ffe0ec", "#ff4d7d", "#b5001f", "#3d0010"])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — INDEX PASS
#   Reads ONLY "ITEM: TIMESTEP" and "ITEM: NUMBER OF ATOMS" lines.
#   Everything else is counted by line so we know how many lines to skip,
#   but never parsed. Returns a list of FrameInfo dicts.
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PICK EVENLY DISTRIBUTED FRAME INDICES
# ══════════════════════════════════════════════════════════════════════════════

def pick_frame_indices(index, n_evolution, n_avg):
    """
    Returns two sorted arrays of integer positions into `index`:
        evo_idx  — N_FRAMES_EVOLUTION evenly spaced over the full trajectory
        avg_idx  — N_FRAMES_AVG       evenly spaced over the full trajectory
    The union of both is what we actually need to load from disk.
    """
    n_total = len(index)
    n_evo   = min(n_evolution, n_total)
    n_av    = min(n_avg,       n_total)

    evo_idx = np.linspace(0, n_total - 1, n_evo, dtype=int)
    avg_idx = np.linspace(0, n_total - 1, n_av,  dtype=int)

    print(f"\n[2/3] Frame selection:")
    print(f"    Evolution grid : {n_evo} frames  "
          f"(t={index[evo_idx[0]]['timestep']} … {index[evo_idx[-1]]['timestep']})")
    print(f"    Averaging      : {n_av} frames  "
          f"(t={index[avg_idx[0]]['timestep']} … {index[avg_idx[-1]]['timestep']})")

    return evo_idx, avg_idx


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — LOAD ONLY THE SELECTED FRAMES (seek by byte offset)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_frame_at(fh, frame_info):
    """
    fh is already seeked to frame_info['offset'] (start of 'ITEM: TIMESTEP').
    Parses and returns (timestep, box, coords_dict).
    """
    fh.readline()                              # ITEM: TIMESTEP
    timestep = int(fh.readline().strip())
    fh.readline()                              # ITEM: NUMBER OF ATOMS
    n_atoms = int(fh.readline().strip())

    fh.readline()                              # ITEM: BOX BOUNDS ...
    box = []
    for _ in range(3):
        lo, hi = map(float, fh.readline().split())
        box.append([lo, hi])
    box = np.array(box)
    L = box[:, 1] - box[:, 0]

    header = fh.readline()                     # ITEM: ATOMS id type xs ys zs
    cols   = header.split()[2:]
    id_col = cols.index("id")
    try:
        x_col  = cols.index("xs"); scaled = True
    except ValueError:
        x_col  = cols.index("x");  scaled = False
    y_col, z_col = x_col + 1, x_col + 2

    coords = {}
    for _ in range(n_atoms):
        parts = fh.readline().split()
        aid = int(parts[id_col])
        x   = float(parts[x_col])
        y   = float(parts[y_col])
        z   = float(parts[z_col])
        if scaled:
            x = box[0, 0] + x * L[0]
            y = box[1, 0] + y * L[1]
            z = box[2, 0] + z * L[2]
        coords[aid] = np.array([x, y, z])

    return timestep, box, coords


def load_frames(filepath, index, needed_indices, polymer_ids, cutoff):
    """
    Seeks to each needed frame by byte offset; parses only those frames.
    Returns a dict  position → (timestep, contact_matrix)
    and the sorted polymer bead IDs (from the first frame).
    """
    needed_sorted = sorted(set(needed_indices))   # visit offsets in order
    results  = {}
    ids_out  = None

    print(f"\n[3/3] Loading {len(needed_sorted)} frames from disk …")
    with open(filepath, "r") as fh:
        for pos in tqdm(needed_sorted, desc="  Parsing frames",
                        unit="frame", dynamic_ncols=True):
            fh.seek(index[pos]["offset"])
            ts, box, coords = _parse_frame_at(fh, index[pos])
            mat, ids = _coords_to_contact_matrix(coords, polymer_ids, cutoff, box)
            results[pos] = (ts, mat)
            if ids_out is None:
                ids_out = ids

    print(f"  → done. Polymer beads: {len(ids_out)} "
          f"(IDs {ids_out[0]}–{ids_out[-1]})")
    return results, ids_out


# ══════════════════════════════════════════════
# CONTACT MATRIX  (pure numpy, minimum-image PBC)
# ══════════════════════════════════════════════

def _coords_to_contact_matrix(coords, polymer_ids, cutoff, box):
    ids = sorted(pid for pid in polymer_ids if pid in coords)
    if not ids:
        raise ValueError("No polymer atoms found in this frame.")
    L   = box[:, 1] - box[:, 0]
    pos = np.array([coords[i] for i in ids])           # (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    diff -= np.round(diff / L) * L                        # minimum image
    dist  = np.sqrt((diff ** 2).sum(axis=-1))             # (N, N)
    mat   = (dist <= cutoff).astype(np.float32)
    np.fill_diagonal(mat, 0)
    return mat, ids


# ══════════════════════════════════════════════
# AXIS TICK HELPER
# ══════════════════════════════════════════════

def _ticks(ids, n=8):
    N = len(ids)
    t = np.arange(0, N, max(1, N // n))
    return t, [str(ids[i]) for i in t]


# ══════════════════════════════════════════════
# FIGURE 1 — Evolution grid
# ══════════════════════════════════════════════

def plot_evolution(evo_frames, output_dir):
    n_show = len(evo_frames)
    ncols  = min(10, n_show)
    nrows  = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.6, nrows * 1.6 + 0.4),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, (ts, mat) in zip(axes, evo_frames):
        ax.imshow(mat, cmap=FRUIT_PUNCH, vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ax.set_title(f"t={ts}", fontsize=6)
        ax.axis("off")
    for ax in axes[n_show:]:
        ax.set_visible(False)

    fig.suptitle(f"Contact Map Evolution  (cutoff = {CONTACT_DIST} σ)",
                 fontsize=12, fontweight="bold")
    out = output_dir / "01_contact_evolution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════
# FIGURE 2 — Time-averaged map
# ══════════════════════════════════════════════

def plot_average(avg_mat, ids, output_dir):
    ticks, labels = _ticks(ids)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(avg_mat, cmap=FRUIT_PUNCH, vmin=0, vmax=1,
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("Monomer index", fontsize=11)
    ax.set_ylabel("Monomer index", fontsize=11)
    ax.set_title(f"Time-averaged Contact Map  "
                 f"(N={len(ids)}, cutoff={CONTACT_DIST} σ)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Contact probability")
    fig.tight_layout()
    out = output_dir / "02_avg_contact_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════
# FIGURE 3 — Std-deviation map
# ══════════════════════════════════════════════

def plot_std(std_mat, ids, output_dir):
    ticks, labels = _ticks(ids)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(std_mat, cmap=FRUIT_PUNCH_STD, vmin=0,
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("Monomer index", fontsize=11)
    ax.set_ylabel("Monomer index", fontsize=11)
    ax.set_title(f"Contact Map Std Deviation  "
                 f"(N={len(ids)}, cutoff={CONTACT_DIST} σ)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Std deviation")
    fig.tight_layout()
    out = output_dir / "03_std_contact_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════
# FIGURE 4 — Hi-C style
# ══════════════════════════════════════════════

def plot_hic(avg_mat, ids, output_dir):
    N = len(ids)
    ticks, labels = _ticks(ids)
    log_mat = np.log10(avg_mat + 1e-3)

    fig = plt.figure(figsize=(9.5, 5.2))
    gs  = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.38)
    ax_map = fig.add_subplot(gs[0])
    ax_sep = fig.add_subplot(gs[1])

    vmin, vmax = np.percentile(log_mat, [2, 98])
    im = ax_map.imshow(log_mat, cmap=FRUIT_PUNCH_HIC, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest")
    ax_map.set_xlabel("Monomer index", fontsize=11)
    ax_map.set_ylabel("Monomer index", fontsize=11)
    ax_map.set_title("Hi-C Style Map  (log₁₀ contact probability)",
                     fontsize=11, fontweight="bold")
    ax_map.set_xticks(ticks); ax_map.set_xticklabels(labels)
    ax_map.set_yticks(ticks); ax_map.set_yticklabels(labels)
    fig.colorbar(im, ax=ax_map, label="log₁₀(P_contact)")

    sep_vals = [np.diagonal(avg_mat, offset=s).mean() for s in range(1, N)]
    ax_sep.plot(sep_vals, np.arange(1, N), color="#c0392b", lw=1.8)
    ax_sep.set_xscale("log")
    ax_sep.set_xlabel("P(contact)", fontsize=10)
    ax_sep.set_ylabel("|i − j|", fontsize=10)
    ax_sep.set_title("P vs separation", fontsize=10, fontweight="bold")
    ax_sep.grid(True, alpha=0.3)

    fig.suptitle(f"Hi-C Analysis  (N={N}, cutoff={CONTACT_DIST} σ)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "04_hic_style_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════
# FIGURE 5 — Scaling law
# ══════════════════════════════════════════════

def plot_scaling(avg_mat, ids, output_dir):
    N = len(ids)
    sep_vals    = np.array([np.diagonal(avg_mat, offset=s).mean()
                            for s in range(1, N)])
    separations = np.arange(1, N)
    mask        = separations >= 2
    alpha, intercept = np.polyfit(np.log(separations[mask]),
                                  np.log(sep_vals[mask] + 1e-10), 1)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.loglog(separations, sep_vals, "o", ms=4, color="#ff6b9d", label="Data")
    ax.loglog(separations, np.exp(intercept) * separations ** alpha,
              "--", color="#6b0000", label=f"Fit  α = {alpha:.2f}")
    ax.set_xlabel("|i − j|  (sequence separation)", fontsize=11)
    ax.set_ylabel("Mean contact probability", fontsize=11)
    ax.set_title("Contact Scaling Law", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = output_dir / "05_contact_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}  (α = {alpha:.3f})")


# ══════════════════════════════════════════════
# FIGURE 6 — Interactive HTML slider
# ══════════════════════════════════════════════

def _mat_to_b64(mat, cmap, vmin=0, vmax=1, px=400):
    fig, ax = plt.subplots(figsize=(px / 100, px / 100), dpi=100)
    ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
              origin="lower", interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def export_html_slider(evo_frames, ids, output_dir):
    N = len(ids)
    n = len(evo_frames)

    print(f"  Encoding {n} frames as PNG …")
    images_b64 = []
    timesteps  = []
    for ts, mat in tqdm(evo_frames, desc="  Encoding", unit="frame",
                        dynamic_ncols=True):
        images_b64.append(_mat_to_b64(mat, FRUIT_PUNCH))
        timesteps.append(int(ts))

    imgs_json = json.dumps(images_b64)
    ts_json   = json.dumps(timesteps)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Contact Map Evolution</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #1a1a2e; color: #eee;
    font-family: "Segoe UI", system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center;
    padding: 32px 16px 56px; min-height: 100vh;
  }}
  h1 {{ font-size: 1.45rem; font-weight: 700; letter-spacing: .02em;
        margin-bottom: 6px; text-align: center; }}
  .subtitle {{ font-size: .83rem; color: #999; margin-bottom: 24px; text-align: center; }}
  #map-wrap {{
    position: relative; width: 440px; height: 440px;
    border: 2px solid #444; border-radius: 10px; overflow: hidden;
    background: #000; box-shadow: 0 8px 36px rgba(0,0,0,.65);
  }}
  #map-wrap img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .badge {{
    position: absolute; background: rgba(0,0,0,.62); color: #fff;
    font-size: .76rem; padding: 3px 9px; border-radius: 4px;
    pointer-events: none; letter-spacing: .03em;
  }}
  #badge-ts    {{ top: 9px; right: 9px; }}
  #badge-frame {{ top: 9px; left: 9px; color: #ccc; }}
  .controls {{
    display: flex; flex-direction: column; align-items: center;
    gap: 14px; margin-top: 20px; width: 440px;
  }}
  #slider {{
    -webkit-appearance: none; width: 100%; height: 6px; border-radius: 3px;
    background: linear-gradient(to right,
      #e74c3c 0%, #e74c3c var(--pct,0%), #444 var(--pct,0%), #444 100%);
    outline: none; cursor: pointer;
  }}
  #slider::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 18px; height: 18px;
    border-radius: 50%; background: #fff;
    box-shadow: 0 0 5px rgba(0,0,0,.5); cursor: pointer;
  }}
  #slider::-moz-range-thumb {{
    width: 18px; height: 18px; border-radius: 50%;
    background: #fff; border: none; cursor: pointer;
  }}
  .btn-row {{ display: flex; gap: 10px; align-items: center; }}
  button {{
    background: #2d2d4e; color: #eee; border: 1px solid #555;
    border-radius: 6px; padding: 7px 18px; font-size: .87rem;
    cursor: pointer; transition: background .15s;
  }}
  button:hover {{ background: #3a3a6e; }}
  button.on {{ background: #e74c3c; border-color: #e74c3c; color: #fff; }}
  select {{
    background: #2d2d4e; color: #eee; border: 1px solid #555;
    border-radius: 6px; padding: 6px 10px; font-size: .84rem; cursor: pointer;
  }}
  .info {{ display: flex; gap: 28px; font-size: .78rem; color: #888; }}
  .info b {{ color: #ccc; }}
</style>
</head>
<body>
<h1>Contact Map — Time Evolution</h1>
<p class="subtitle">
  N = {N} monomers &nbsp;|&nbsp; cutoff = {CONTACT_DIST} σ &nbsp;|&nbsp; {n} frames
</p>
<div id="map-wrap">
  <img id="img" src="" alt="contact map">
  <div class="badge" id="badge-ts">t = —</div>
  <div class="badge" id="badge-frame">1 / {n}</div>
</div>
<div class="controls">
  <input type="range" id="slider" min="0" max="{n-1}" value="0" step="1">
  <div class="btn-row">
    <button id="btn-prev">&#9664; Prev</button>
    <button id="btn-play" class="on">&#9654; Play</button>
    <button id="btn-next">Next &#9654;</button>
    <span style="font-size:.8rem;color:#aaa">Speed:</span>
    <select id="speed">
      <option value="800">0.5×</option>
      <option value="400" selected>1×</option>
      <option value="200">2×</option>
      <option value="100">4×</option>
      <option value="50">8×</option>
    </select>
  </div>
  <div class="info">
    <span>Frames: <b>{n}</b></span>
    <span>Monomers: <b>{N}</b></span>
    <span>Cutoff: <b>{CONTACT_DIST} σ</b></span>
  </div>
</div>
<script>
const IMGS = {imgs_json};
const TS   = {ts_json};
const NF   = IMGS.length;
let cur = 0, playing = false, timer = null;
const el = id => document.getElementById(id);
const img = el("img"), slider = el("slider");
const badgeTs = el("badge-ts"), badgeFr = el("badge-frame");
const btnPlay = el("btn-play"), btnPrev = el("btn-prev"), btnNext = el("btn-next");
const speedSel = el("speed");

function show(i) {{
  cur = ((i % NF) + NF) % NF;
  img.src = "data:image/png;base64," + IMGS[cur];
  badgeTs.textContent = "t = " + TS[cur].toLocaleString();
  badgeFr.textContent = (cur + 1) + " / " + NF;
  slider.value = cur;
  slider.style.setProperty("--pct", (cur / (NF - 1) * 100) + "%");
}}
function play() {{
  timer = setInterval(() => show(cur + 1), +speedSel.value);
  playing = true; btnPlay.textContent = "⏸ Pause"; btnPlay.classList.add("on");
}}
function pause() {{
  clearInterval(timer);
  playing = false; btnPlay.textContent = "▶ Play"; btnPlay.classList.remove("on");
}}
btnPlay.onclick   = () => playing ? pause() : play();
btnPrev.onclick   = () => {{ pause(); show(cur - 1); }};
btnNext.onclick   = () => {{ pause(); show(cur + 1); }};
slider.oninput    = () => {{ pause(); show(+slider.value); }};
speedSel.onchange = () => {{ if (playing) {{ pause(); play(); }} }};
document.addEventListener("keydown", e => {{
  if (e.key === "ArrowRight") {{ pause(); show(cur + 1); }}
  if (e.key === "ArrowLeft")  {{ pause(); show(cur - 1); }}
  if (e.key === " ")          {{ btnPlay.click(); e.preventDefault(); }}
}});
show(0); play();
</script>
</body>
</html>
"""
    out = output_dir / "06_contact_evolution.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fast index pass — only reads TIMESTEP + NUMBER OF ATOMS lines
    index = index_trajectory(TRAJ_FILE)

    # 2. Pick evenly distributed indices for both tasks
    evo_idx, avg_idx = pick_frame_indices(
        index, N_FRAMES_EVOLUTION, N_FRAMES_AVG
    )

    # 3. Load only the needed frames (seek by byte offset)
    needed = np.union1d(evo_idx, avg_idx)
    frame_data, ids = load_frames(
        TRAJ_FILE, index, needed, POLYMER_IDS, CONTACT_DIST
    )

    # Reassemble ordered lists for each task
    evo_frames = [frame_data[i] for i in evo_idx]   # list of (ts, mat)
    avg_frames = [frame_data[i] for i in avg_idx]

    # 4. Compute average and std from avg_frames
    stack   = np.stack([mat for _, mat in tqdm(avg_frames,
                        desc="  Stacking avg frames", unit="frame",
                        dynamic_ncols=True)], axis=0)
    avg_mat = stack.mean(axis=0)
    std_mat = stack.std(axis=0)

    # 5. Generate all outputs
    print("\n" + "─" * 50)
    print("Generating figures …")
    plot_evolution(evo_frames, output_dir)
    plot_average(avg_mat, ids, output_dir)
    plot_std(std_mat, ids, output_dir)
    plot_hic(avg_mat, ids, output_dir)
    plot_scaling(avg_mat, ids, output_dir)

    print("\nGenerating HTML animation …")
    export_html_slider(evo_frames, ids, output_dir)

    print("─" * 50)
    print(f"All outputs → {output_dir.resolve()}")
    print("Done ✓")


if __name__ == "__main__":
    main()