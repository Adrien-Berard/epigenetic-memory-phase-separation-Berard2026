"""
src_figS13_replication.py
----------
Supplementary Figure S13: replication-phase contact maps from trajectory.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# ─────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────
TRAJ_FILE          = "dump.lammpstrj"
REPLICATION_FILE   = "replication_timeline.dat"   # columns: cycle, event, step
POLYMER_IDS        = set(range(1, 501))            # atom IDs 1–500
CONTACT_DIST       = 3.0                           # contact cutoff in σ
N_FRAMES_SAMPLE    = 1000                           # frames sampled for analysis
OUTPUT_DIR         = Path("Hi-CperCycleState")
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
# COLORMAPS
# ══════════════════════════════════════════════
FRUIT_PUNCH_HIC = mcolors.LinearSegmentedColormap.from_list(
    "fruit_punch_hic", ["#ffffff", "#ffe0ec", "#ff4d7d", "#b5001f", "#3d0010"])


# ══════════════════════════════════════════════
# STEP 1 — INDEX TRAJECTORY
# ══════════════════════════════════════════════

def index_trajectory(filepath):
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {filepath}")

    index = []
    reading_ts = reading_natoms = False
    current_ts = current_offset = None

    print(f"[1/4] Scanning trajectory index: {filepath}")
    file_size = path.stat().st_size

    with open(path, "rb") as fh:
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
                    index.append({"timestep": current_ts,
                                  "offset":   current_offset,
                                  "n_atoms":  n_atoms})

    if not index:
        sys.exit("[ERROR] No frames found in trajectory file.")

    ts_list = [f["timestep"] for f in index]
    print(f"\n  Trajectory summary:")
    print(f"    Total frames   : {len(index)}")
    print(f"    First timestep : {ts_list[0]}")
    print(f"    Last  timestep : {ts_list[-1]}")
    return index


# ══════════════════════════════════════════════
# STEP 2 — LOAD REPLICATION TIMELINE
# ══════════════════════════════════════════════

def load_replication_timeline(path):
    """
    CSV must have columns: cycle, event, step
    Expected events per cycle:
        G2_start, G2_end, Mitosis_start, Mitosis_end
    G1 is inferred as everything between Mitosis_end and the next G2_start.
    """
    df = pd.read_csv(path)
    phases = {}
    for cycle in sorted(df["cycle"].unique()):
        sub = df[df["cycle"] == cycle]
        def get(event):
            rows = sub[sub["event"] == event]["step"].values
            if len(rows) == 0:
                raise ValueError(f"Event '{event}' not found for cycle {cycle}")
            return int(rows[0])
        phases[cycle] = {
            "G2": (get("G2_start"),      get("G2_end")),
            "M":  (get("Mitosis_start"), get("Mitosis_end")),
        }
    return phases


def assign_phase(ts, phases):
    """Return (cycle, phase_str) or (None, None) if timestep falls in no phase."""
    sorted_cycles = sorted(phases.keys())
    for idx, c in enumerate(sorted_cycles):
        g2s, g2e = phases[c]["G2"]
        ms,  me  = phases[c]["M"]

        if g2s <= ts < g2e:
            return c, "G2"
        if ms <= ts < me:
            return c, "M"

        # G1: between Mitosis_end of this cycle and G2_start of next cycle
        if ts >= me:
            if idx + 1 < len(sorted_cycles):
                next_g2s = phases[sorted_cycles[idx + 1]]["G2"][0]
                if ts < next_g2s:
                    return c, "G1"
            else:
                # Last cycle — G1 extends to end of trajectory
                return c, "G1"

    return None, None


# ══════════════════════════════════════════════
# STEP 3 — PARSE FRAMES
# ══════════════════════════════════════════════

def _parse_frame_at(fh, frame_info):
    fh.readline()                              # ITEM: TIMESTEP
    timestep = int(fh.readline().strip())
    fh.readline()                              # ITEM: NUMBER OF ATOMS
    n_atoms = int(fh.readline().strip())

    fh.readline()                              # ITEM: BOX BOUNDS
    box = []
    for _ in range(3):
        lo, hi = map(float, fh.readline().split())
        box.append([lo, hi])
    box = np.array(box)
    L = box[:, 1] - box[:, 0]

    header = fh.readline()                     # ITEM: ATOMS ...
    cols   = header.split()[2:]
    id_col = cols.index("id")
    try:
        x_col = cols.index("xs"); scaled = True
    except ValueError:
        x_col = cols.index("x");  scaled = False
    y_col, z_col = x_col + 1, x_col + 2

    coords = {}
    for _ in range(n_atoms):
        parts = fh.readline().split()
        aid = int(parts[id_col])
        x, y, z = float(parts[x_col]), float(parts[y_col]), float(parts[z_col])
        if scaled:
            x = box[0, 0] + x * L[0]
            y = box[1, 0] + y * L[1]
            z = box[2, 0] + z * L[2]
        coords[aid] = np.array([x, y, z])

    return timestep, box, coords


def _coords_to_contact_matrix(coords, polymer_ids, cutoff, box):
    ids = sorted(pid for pid in polymer_ids if pid in coords)
    if not ids:
        raise ValueError("No polymer atoms found in this frame.")
    L   = box[:, 1] - box[:, 0]
    pos = np.array([coords[i] for i in ids])
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    diff -= np.round(diff / L) * L
    dist  = np.sqrt((diff ** 2).sum(axis=-1))
    mat   = (dist <= cutoff).astype(np.float32)
    np.fill_diagonal(mat, 0)
    return mat, ids


def load_frames(filepath, index, needed_indices, polymer_ids, cutoff):
    needed_sorted = sorted(set(needed_indices))
    results = {}
    ids_out = None

    print(f"\n[3/4] Loading {len(needed_sorted)} frames …")
    with open(filepath, "r") as fh:
        for pos in tqdm(needed_sorted, desc="  Parsing", unit="frame",
                        dynamic_ncols=True):
            fh.seek(index[pos]["offset"])
            ts, box, coords = _parse_frame_at(fh, index[pos])
            mat, ids = _coords_to_contact_matrix(coords, polymer_ids, cutoff, box)
            results[pos] = (ts, mat)
            if ids_out is None:
                ids_out = ids

    print(f"  → Polymer beads: {len(ids_out)} (IDs {ids_out[0]}–{ids_out[-1]})")
    return results, ids_out


# ══════════════════════════════════════════════
# FIGURE — Hi-C per phase with shared colorbar
# ══════════════════════════════════════════════

def _ticks(ids, n=8):
    N = len(ids)
    t = np.arange(0, N, max(1, N // n))
    return t, [str(ids[i]) for i in t]


def plot_hic_phases(phase_avg, ids, output_dir):
    """
    phase_avg: dict  phase_label -> averaged contact matrix
    Produces one PNG per phase AND one combined comparison figure,
    all using the same vmin/vmax on log scale.
    """
    PHASE_ORDER = ["G1", "G2", "M"]
    PHASE_LABEL = {"G1": "G1 (post-mitotic growth)",
                   "G2": "G2 (pre-mitotic)",
                   "M":  "Mitosis"}

    available = [p for p in PHASE_ORDER if p in phase_avg]
    if not available:
        print("  [!] No phase data found — skipping Hi-C phase figures.")
        return

    ticks, labels = _ticks(ids)
    N = len(ids)

    # Compute log matrices and global colour scale
    log_mats = {}
    for p in available:
        log_mats[p] = np.log10(phase_avg[p] + 1e-3)

    all_vals = np.concatenate([m.ravel() for m in log_mats.values()])
    vmin, vmax = np.percentile(all_vals, [2, 98])

    def _sep_curve(avg_mat):
        return [np.diagonal(avg_mat, offset=s).mean() for s in range(1, N)]

    # ── Individual Hi-C + P(s) figures ──────────────────────────────────────
    for p in available:
        fig = plt.figure(figsize=(9.5, 5.2))
        gs  = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.38)
        ax_map = fig.add_subplot(gs[0])
        ax_sep = fig.add_subplot(gs[1])

        im = ax_map.imshow(log_mats[p], cmap=FRUIT_PUNCH_HIC,
                           vmin=vmin, vmax=vmax,
                           origin="lower", interpolation="nearest")
        ax_map.set_xlabel("Monomer index", fontsize=11)
        ax_map.set_ylabel("Monomer index", fontsize=11)
        ax_map.set_title(f"Hi-C — {PHASE_LABEL.get(p, p)}\n"
                         f"(log₁₀ contact probability)",
                         fontsize=11, fontweight="bold")
        ax_map.set_xticks(ticks); ax_map.set_xticklabels(labels)
        ax_map.set_yticks(ticks); ax_map.set_yticklabels(labels)
        fig.colorbar(im, ax=ax_map, label="log₁₀(P_contact)")

        sep_vals = _sep_curve(phase_avg[p])
        ax_sep.plot(sep_vals, np.arange(1, N), color="#c0392b", lw=1.8)
        ax_sep.set_xscale("log")
        ax_sep.set_xlabel("P(contact)", fontsize=10)
        ax_sep.set_ylabel("|i − j|", fontsize=10)
        ax_sep.set_title("P vs separation", fontsize=10, fontweight="bold")
        ax_sep.grid(True, alpha=0.3)

        n_frames = len(phase_avg.get(f"_frames_{p}", [p]))   # fallback
        fig.suptitle(f"Phase: {p}  |  N = {N} monomers  |  cutoff = {CONTACT_DIST} σ",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        out = output_dir / f"hic_{p}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ── Combined comparison figure ───────────────────────────────────────────
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 5.4),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, available):
        im = ax.imshow(log_mats[p], cmap=FRUIT_PUNCH_HIC,
                       vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest")
        ax.set_title(PHASE_LABEL.get(p, p), fontsize=12, fontweight="bold")
        ax.set_xlabel("Monomer index", fontsize=10)
        ax.set_ylabel("Monomer index", fontsize=10)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)

    # Single shared colourbar on the right
    fig.colorbar(im, ax=axes, label="log₁₀(P_contact)", shrink=0.85)
    fig.suptitle(f"Hi-C Contact Maps by Cell-Cycle Phase  "
                 f"(N = {N}, cutoff = {CONTACT_DIST} σ)",
                 fontsize=13, fontweight="bold")

    out = output_dir / "hic_phases_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Index
    index = index_trajectory(TRAJ_FILE)

    # 2. Load replication timeline
    print(f"\n[2/4] Loading replication timeline: {REPLICATION_FILE}")
    phases = load_replication_timeline(REPLICATION_FILE)
    print(f"  Cycles found: {sorted(phases.keys())}")

    # 3. Sample evenly across trajectory and load frames
    n_total  = len(index)
    n_sample = min(N_FRAMES_SAMPLE, n_total)
    sample_idx = np.linspace(0, n_total - 1, n_sample, dtype=int)

    frame_data, ids = load_frames(
        TRAJ_FILE, index, sample_idx, POLYMER_IDS, CONTACT_DIST
    )

    # 4. Assign each frame to a phase and accumulate
    print("\n[4/4] Assigning frames to cell-cycle phases …")
    phase_stacks = defaultdict(list)   # phase -> list of matrices
    phase_counts = defaultdict(int)

    for i in sample_idx:
        ts, mat = frame_data[i]
        _, phase = assign_phase(ts, phases)
        if phase is not None:
            phase_stacks[phase].append(mat)
            phase_counts[phase] += 1

    print("\n  Frame counts per phase:")
    for p in ["G1", "G2", "M"]:
        print(f"    {p}: {phase_counts[p]} frames")

    if not phase_stacks:
        sys.exit("[ERROR] No frames could be assigned to any phase. "
                 "Check that REPLICATION_FILE timesteps overlap with trajectory.")

    # Average per phase
    phase_avg = {p: np.stack(mats).mean(axis=0)
                 for p, mats in phase_stacks.items()}

    # 5. Plot
    print("\nGenerating Hi-C figures …")
    plot_hic_phases(phase_avg, ids, output_dir)

    print("─" * 50)
    print(f"All outputs → {output_dir.resolve()}")
    print("Done")


if __name__ == "__main__":
    main()