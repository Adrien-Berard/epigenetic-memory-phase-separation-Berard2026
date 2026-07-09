"""
src_figS9.py
--------------
Supplemetary figure S9: Swi6* local concentration and CoM distance.
"""

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

plt.rcParams.update({
    "font.family":          "serif",
    "font.size":            11,
    "axes.titlesize":       10,
    "axes.labelsize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "axes.linewidth":       0.8,
    "lines.linewidth":      1.3,
    "grid.linewidth":       0.4,
    "grid.alpha":           0.35,
    "axes.grid":            True,
    "figure.dpi":           150,
    "pdf.fonttype":         42,  
})

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION DEFINITIONS
# Row index drives plot row (0 = Fig6, 1 = FigS7); two sims per row → 2 cols
# ──────────────────────────────────────────────────────────────────────────────
BASE = Path(".")

SIMULATIONS = [
    # ── Fig6 ──────────────────────────────────────────────────────────────────
    {
        "folder":      BASE / "Fig6" / "sim_p2_0.00025_noise_500_swi6_400_nuc_160",
        "label":       "Fig6 – free heterochromatin",
        "polymers":    {"ΔK": (1, 80), "Fixed HC": (81, 160)},
        "target_type": 5,
        "cutoff":      3.0,
        "row":         0,
    },
    {
        "folder":      BASE / "Fig6" / "sim_p2_0.00025_noise_500_swi6_400_nuc_160Fixed",
        "label":       "Fig6 – fixed heterochromatin",
        "polymers":    {"ΔK": (1, 80), "Fixed HC": (81, 160)},
        "target_type": 5,
        "cutoff":      3.0,
        "row":         0,
    },
    # ── FigS7 ─────────────────────────────────────────────────────────────────
    {
        "folder":      BASE / "FigS7" / "2polymersFullAAFullMM_Triplicate1_simBis_1e7timesteps_FullA_FullA_p2_0.00025_noise_500_swi6_400",
        "label":       "FigS7 – FullA / FullA",
        "polymers":    {"Polymer I": (1, 80), "Polymer II": (81, 160)},
        "target_type": 5,
        "cutoff":      3.0,
        "row":         1,
    },
    {
        "folder":      BASE / "FigS7" / "2polymersFullAAFullMM_Triplicate1_simBis_1e7timesteps_FullM_FullM_p2_0.00025_noise_500_swi6_400",
        "label":       "FigS7 – FullM / FullM",
        "polymers":    {"Polymer I": (1, 80), "Polymer II": (81, 160)},
        "target_type": 5,
        "cutoff":      3.0,
        "row":         1,
    },
]

# Visual palette
POLY_COLORS = ["#285A13", "#CAFC6E"]   # polymer 1 / polymer 2 concentration
COM_COLOR   = "#9016A3"                # CoM distance line

# ──────────────────────────────────────────────────────────────────────────────
# PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_dump(filepath: Path):
    """Generator: yields one dict per snapshot."""
    with open(filepath, "r") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if "ITEM: TIMESTEP" not in line:
                continue

            timestep = int(fh.readline().strip())
            fh.readline()                                   # ITEM: NUMBER OF ATOMS
            n_atoms = int(fh.readline().strip())
            fh.readline()                                   # ITEM: BOX BOUNDS …
            box = np.array([
                list(map(float, fh.readline().split()))
                for _ in range(3)
            ])
            fh.readline()                                   # ITEM: ATOMS …

            rows = []
            for _ in range(n_atoms):
                p = fh.readline().split()
                rows.append((int(p[0]), int(p[1]),
                             float(p[2]), float(p[3]), float(p[4])))

            df = pd.DataFrame(rows, columns=["id", "type", "xs", "ys", "zs"])
            yield {"timestep": timestep, "box": box, "atoms": df}


def scaled_to_real(df: pd.DataFrame, box: np.ndarray) -> pd.DataFrame:
    L = box[:, 1] - box[:, 0]                              # [Lx, Ly, Lz]
    df = df.copy()
    df["x"] = df["xs"] * L[0] + box[0, 0]
    df["y"] = df["ys"] * L[1] + box[1, 0]
    df["z"] = df["zs"] * L[2] + box[2, 0]
    return df

# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY  (PBC-correct)
# ──────────────────────────────────────────────────────────────────────────────

def pbc_distance_sq_batch(r1: np.ndarray,
                           r2: np.ndarray,
                           L:  np.ndarray) -> np.ndarray:
    """
    Squared minimum-image distances from ONE point r1 (3,)
    to MANY points r2 (N,3).
    """
    delta  = r2 - r1                            # (N,3)
    delta -= np.round(delta / L) * L            # minimum image
    return (delta * delta).sum(axis=1)          # (N,)


def unwrap_polymer(coords: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Unwrap polymer bead coordinates so consecutive beads are connected
    through real space rather than jumping across periodic images.

    Algorithm: starting from bead 0, shift each successive bead by the
    minimum-image vector from the previous bead.  This gives a continuous
    chain in unwrapped space.

    Parameters
    ----------
    coords : (N, 3)  wrapped real-space coordinates (sorted by bead order)
    L      : (3,)    box lengths

    Returns
    -------
    unwrapped : (N, 3)  coordinates with no periodic jumps
    """
    unwrapped = coords.copy()
    for i in range(1, len(unwrapped)):
        delta  = unwrapped[i] - unwrapped[i - 1]
        delta -= np.round(delta / L) * L        # minimum-image shift
        unwrapped[i] = unwrapped[i - 1] + delta
    return unwrapped


def pbc_com_distance(coords1: np.ndarray,
                     coords2: np.ndarray,
                     L: np.ndarray) -> float:
    """
    PBC-correct distance between the centres of mass of two polymers.

    Steps
    -----
    1. Unwrap each polymer independently so intra-chain coordinates are
       continuous (eliminates spurious COM shifts when part of a chain
       crosses a box face).
    2. Compute each COM in unwrapped space.
    3. Apply the minimum-image convention to the vector between the two
       COMs (the CoMs themselves may still differ by an image).

    Note: coords must be sorted in bead-index order (as they come out of
    the dump file when filtered by atom-ID range).
    """
    uw1  = unwrap_polymer(coords1, L)
    uw2  = unwrap_polymer(coords2, L)
    com1 = uw1.mean(axis=0)
    com2 = uw2.mean(axis=0)
    dv   = com2 - com1
    dv  -= np.round(dv / L) * L                # minimum image of CoM vector
    return float(np.sqrt((dv * dv).sum()))


def concentration_around_polymer(poly_coords: np.ndarray,
                                  tgt_coords:  np.ndarray,
                                  L: np.ndarray,
                                  cutoff: float) -> float:
    """
    Number of unique target atoms within `cutoff` of ANY polymer bead,
    normalised by the volume of a single sphere (number density).
    """
    cutoff_sq = cutoff ** 2
    within    = set()
    for bead in poly_coords:
        dsq = pbc_distance_sq_batch(bead, tgt_coords, L)
        within.update(np.where(dsq <= cutoff_sq)[0].tolist())
    sphere_vol = (4.0 / 3.0) * np.pi * cutoff ** 3
    return len(within) / sphere_vol

# ──────────────────────────────────────────────────────────────────────────────
# PER-SNAPSHOT WORKER  (top-level so multiprocessing can pickle it)
# ──────────────────────────────────────────────────────────────────────────────

def _process_snapshot(snap, polymer_defs, target_type, cutoff):
    ts   = snap["timestep"]
    box  = snap["box"]
    df   = scaled_to_real(snap["atoms"], box)
    L    = box[:, 1] - box[:, 0]

    tgt_coords = df.loc[df["type"] == target_type, ["x", "y", "z"]].values

    row              = {"timestep": ts}
    poly_coord_list  = []

    for name, (id_lo, id_hi) in polymer_defs.items():
        # Keep bead order (sort by atom id) so unwrap_polymer works correctly
        mask   = (df["id"] >= id_lo) & (df["id"] <= id_hi)
        sub    = df.loc[mask].sort_values("id")
        coords = sub[["x", "y", "z"]].values
        poly_coord_list.append(coords)

        if coords.size == 0:
            row[f"conc_{name}"] = np.nan
        else:
            row[f"conc_{name}"] = concentration_around_polymer(
                coords, tgt_coords, L, cutoff
            )

    # PBC-correct CoM distance
    if len(poly_coord_list) == 2 and all(c.size > 0 for c in poly_coord_list):
        row["com_distance"] = pbc_com_distance(
            poly_coord_list[0], poly_coord_list[1], L
        )
    else:
        row["com_distance"] = np.nan

    return row

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS DRIVER
# ──────────────────────────────────────────────────────────────────────────────

def analyse_simulation(sim: dict, n_workers: int) -> pd.DataFrame:
    folder      = Path(sim["folder"])
    dump_file   = folder / "dump.lammpstrj"
    csv_cache   = folder / "concentration_timeseries.csv"

    if csv_cache.exists():
        print(f"  [cache]    {csv_cache}")
        return pd.read_csv(csv_cache)

    if not dump_file.exists():
        print(f"  [MISSING]  {dump_file}")
        return pd.DataFrame()

    print(f"  [parsing]  {dump_file}  ({n_workers} workers)")
    snapshots = list(parse_dump(dump_file))
    print(f"             → {len(snapshots)} snapshots")

    worker = partial(_process_snapshot,
                     polymer_defs=sim["polymers"],
                     target_type=sim["target_type"],
                     cutoff=sim["cutoff"])

    with Pool(n_workers) as pool:
        records = pool.map(worker, snapshots)

    result = (pd.DataFrame(records)
                .sort_values("timestep")
                .reset_index(drop=True))
    result.to_csv(csv_cache, index=False)
    print(f"  [saved]    {csv_cache}")
    return result

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING  — 2 rows × 2 columns
#   col 0 = concentration (left y-axis)
#   col 1 = CoM distance
#   row 0 = Fig6 (both sims overlaid)
#   row 1 = FigS7 (both sims overlaid)
# ──────────────────────────────────────────────────────────────────────────────
    
def plot_all(sim_results: list, output_pdf: str = "FigS9.pdf"):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(8, 6),
        gridspec_kw={
            "hspace": 0.35,
            "wspace": 0.45,
            "left": 0.07,
            "right": 0.92,
            "top": 0.90,
            "bottom": 0.08,
        },
    )

    row_titles = ["Fig. 6", "Fig. S7"]

    # Group sims by row
    from collections import defaultdict
    rows = defaultdict(list)

    for sim, df in zip(SIMULATIONS, sim_results):
        rows[sim["row"]].append((sim, df))

    for row_idx in range(2):
        row_sims = rows[row_idx]

        for col_idx, (sim, df) in enumerate(row_sims):
            
            ax = axes[row_idx, col_idx]
            ax2 = ax.twinx()
            ax.grid(True)
            ax2.grid(False)
            ax2.set_zorder(1)
            ax.set_zorder(2)
            ax.patch.set_visible(False)

            if row_idx == 0 and col_idx == 0:
               ax.text(-0.23, 1.03, "(a)", transform=ax.transAxes)
            if row_idx == 0 and col_idx == 1:
                ax.text(-0.23, 1.03, "(b)", transform=ax.transAxes)
            if row_idx == 1 and col_idx == 0:
                ax.text(-0.23, 1.03, "(c)", transform=ax.transAxes)
            if row_idx == 1 and col_idx == 1:
                ax.text(-0.23, 1.03, "(d)", transform=ax.transAxes)

            if df.empty:
                ax.set_title(f"{row_titles[row_idx]} — missing data")
                continue

            ts = df["timestep"] / 1e3

            ls = "-"
            alpha = [0.9, 0.75][col_idx]

            label_suffix = sim["label"].split("–")[-1].strip()

            # ─────────────────────────────
            # LEFT Y: concentration
            # ─────────────────────────────
            for pname, color in zip(sim["polymers"].keys(), POLY_COLORS):
                col = f"conc_{pname}"
                if col in df.columns:
                    ax.plot(
                        ts, df[col],
                        color=color,
                        linestyle=ls,
                        alpha=alpha,
                        label=f"{pname} conc",
                    )

            ax.set_ylabel("Swi6* local\n" +r"concentration ($\sigma^{-3}$)")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            # ─────────────────────────────
            # RIGHT Y: CoM distance
            # ─────────────────────────────
            if "com_distance" in df.columns:
                ax2.plot(
                    ts,
                    df["com_distance"],
                    color=COM_COLOR,
                    linestyle=ls,
                    alpha=alpha,
                    label="CoM distance",
                )

            ax2.set_ylabel(r"CoM distance ($\sigma$)", color=COM_COLOR)
            ax2.set_yticks([0, 20, 40]) 
            ax2.tick_params(axis='y', colors=COM_COLOR)
            ax2.spines["right"].set_color(COM_COLOR)

            # ─────────────────────────────
            # TITLES / LABELS
            # ─────────────────────────────
            # ax.set_title(f"{row_titles[row_idx]} — {label_suffix}")
            if row_idx == 1:
                ax.set_xticks([0, 5e4, 10e4])  
                ax.set_xticklabels(['0', '5', r'$10 \times 10^4$'])
                ax.set_yticks([0, 1.0, 2.0])  
                ax.set_yticklabels(['0', '1.0', '2.0'])

            if row_idx == 0 and col_idx == 0:
                ax.set_xticks([0, 5e4, 10e4, 15e4])  
                ax.set_xticklabels(['0', '5', '10', r'$15 \times 10^4$'])  
                ax.set_yticks([0, 1.0, 2.0])  
                ax.set_yticklabels(['0', '1.0', '2.0'])

            if row_idx == 0 and col_idx == 1:
                ax.set_xticks([0, 1e4, 2e4])  
                ax.set_xticklabels(['0', '1', r'$2 \times 10^4$'])  
                ax.set_yticks([0, 1.0, 2.0])  
                ax.set_yticklabels(['0', '1.0', '2.0'])
           
            ax.set_xlabel(r"Time ($\tau_{LJ}$)")

            # ─────────────────────────────
            # LEGENDS (merge both axes)
            # ─────────────────────────────
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()

            if row_idx == 0 and col_idx == 0:
                ax.legend(lines_1 + lines_2,
                      labels_1 + labels_2,
                      loc="lower right",
                      framealpha=1.0,
                      facecolor="white",
                      edgecolor="black")
                
            if row_idx == 1 and col_idx == 0:
                ax.legend(lines_1 + lines_2,
                      labels_1 + labels_2,
                      loc="upper right",
                      framealpha=1.0,
                      facecolor="white",
                      edgecolor="black")
            
            if row_idx == 0 and col_idx == 1:
                ax.legend(lines_1 + lines_2,
                      labels_1 + labels_2,
                      loc="lower left",
                      framealpha=1.0,
                      facecolor="white",
                      edgecolor="black")
            
            if row_idx == 1 and col_idx == 1:
                ax.legend(lines_1 + lines_2,
                      labels_1 + labels_2,
                      loc="upper left",
                      framealpha=1.0,
                      facecolor="white",
                      edgecolor="black")


    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    print(f"\nSaved: {output_pdf}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse LAMMPS dumps.")
    parser.add_argument("--workers", type=int,
                        default=max(1, cpu_count() - 1),
                        help="Parallel workers (default: nCPU-1)")
    parser.add_argument("--output", default="FigS9.pdf",
                        help="Output PDF filename")
    parser.add_argument("--force", action="store_true",
                        help="Delete cached CSVs and recompute everything")
    args = parser.parse_args()

    if args.force:
        for sim in SIMULATIONS:
            c = Path(sim["folder"]) / "concentration_timeseries.csv"
            if c.exists():
                c.unlink()
                print(f"Removed cache: {c}")

    print(f"Workers: {args.workers}\n")
    t0 = time.time()

    sim_results = []
    for sim in SIMULATIONS:
        print(f"── {sim['label']}")
        df = analyse_simulation(sim, n_workers=args.workers)
        sim_results.append(df)

    print(f"\nAll done in {time.time() - t0:.1f}s")
    plot_all(sim_results, output_pdf=args.output)


if __name__ == "__main__":
    main()