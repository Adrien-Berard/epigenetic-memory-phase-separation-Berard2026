"""
src_figS12.py
----------
Supplementary Figure S12: replicated polymer time-series panels from LAMMPS dumps.
"""

import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11
})
# -- CONFIG -------------------------------------------------------------------
DUMP_FILE     = "id_and_type.dat"
R2_FILE       = "r2.dat"
TIMELINE_FILE = "replication_timeline.dat"
TYPES_FILE    = "types1.dat"

# Define polymers (list of (first_id, last_id) tuples).
POLYMERS = [
    (1,  500)
]

TS_STRIDE    = 1    # time-series downsampling stride
KYMO_STRIDE  = 1    # kymograph downsampling stride
TYPES_STEP   = 1000  # types1.dat is written every this many timesteps
DUMP_STEP    = 10000 # dump (id_and_type.dat) is written every this many timesteps
# Ratio: how many types1.dat rows correspond to one dump frame
TYPES_PER_DUMP = DUMP_STEP // TYPES_STEP   # = 10

# Swi6M is reset every 200 steps inside each noise chunk, so individual rows
# can land anywhere in the reset cycle.  We apply a rolling max over
# TYPES_PER_DUMP rows (= one dump interval = 10 000 steps), then keep only
# every TYPES_PER_DUMP-th row.  This gives exactly one Swi6M point per dump
# frame — same temporal resolution as every other panel — while always
# capturing the true peak count within that window.

TYPE_COLORS = {
    1: "#2166AC",   # A  — blue
    2: "#F4C300",   # U  — yellow
    3: "#D6001C",   # M  — red
}
TYPE_LABELS = {1: "A (type 1)", 2: "U (type 2)", 3: "M (type 3)"}
TYPE_CMAP   = mcolors.ListedColormap([TYPE_COLORS[k] for k in sorted(TYPE_COLORS)])

SWI6M_COLOR = "#1A9641"   # green
RG_COLOR    = "#777777"   # grey
EPE1_COLOR  = "#D68222"   # orange

# Cell cycle phase colours
PHASE_STYLES = {
    "G1":      {"color": "#74C476", "alpha": 0.35, "label": "G1"},
    "S":       {"color": "#9E9AC8", "alpha": 0.35, "label": "S-phase"},
    "G2":      {"color": "#4292C6", "alpha": 0.35, "label": "G2"},
    "Mitosis": {"color": "#EF6548", "alpha": 0.55, "label": "Mitosis"},
    
}

# Analysis window: start of simulation up to Mitosis_start of cycle 10.
# TS_MIN is left as None so the full simulation start is used.
# TS_MAX is derived at runtime from the timeline file.
WINDOW_START_EVENT = None                        # None = use first dump frame
WINDOW_END_EVENT   = ("Mitosis_start", 10)       # (event_name, cycle)

# -- PARSERS ------------------------------------------------------------------

def parse_dump(filepath):
    """
    Parse a LAMMPS custom dump with columns: id  type
    Returns:
        timesteps : list of int
        frames    : list of dict {atom_id: atom_type}
    """
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
    """
    Parse r2.dat  (whitespace-separated, '#' comment lines).
    Columns: timestep  value
    Returns (ts_array, vals_array).
    """
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
    """
    Parse types1.dat  (CSV, '#' comment lines).
    Columns: A, U, M, Swi6, Swi6M

    types1.dat is saved every TYPES_STEP (=1000) steps, which is TYPES_PER_DUMP
    (=10) times denser than the dump (saved every DUMP_STEP=10000 steps).

    Swi6M sawtooth problem
    ----------------------
    Inside each LAMMPS run chunk the command
        run N every 200 "set group allSwi6_dynamic type 4"
    resets ALL Swi6M back to type Swi6 every 200 steps.  The save at each 1000-step
    mark can land at any phase of this reset cycle, producing a sawtooth.

    Fix: rolling max over TYPES_PER_DUMP rows then subsample every TYPES_PER_DUMP
    rows, giving exactly one Swi6M point per dump frame.  The timestep for each
    output row is taken directly from dump_timesteps — no reconstruction needed,
    no drift possible.

    Steps
    -----
    1. Drop row 0 (initialisation spike).
    2. Rolling max (Swi6M) / rolling mean (A,U,M,Swi6) over TYPES_PER_DUMP rows.
    3. Keep every TYPES_PER_DUMP-th row (last row of each complete window).
    4. Trim / align to dump_timesteps length.
    5. Use dump_timesteps directly as the x-axis.

    Returns (ts_array, df_downsampled) aligned to the dump timestep array.
    """
    df = pd.read_csv(filepath, comment="#",
                     names=["A", "U", "M", "Swi6", "Swi6M","Epe1"])

    # 1. Drop row 0 (init spike)
    df = df.iloc[1:].reset_index(drop=True)

    w = TYPES_PER_DUMP  # = 10

    # 2. Rolling aggregation
    swi6m_max  = df["Swi6M"].rolling(w, min_periods=w).max()
    other_mean = df[["A", "U", "M", "Swi6M","Swi6","Epe1"]].rolling(w, min_periods=w).mean()

    df_roll = other_mean.copy()
    df_roll["Swi6M_max"] = swi6m_max
    print(f'df_roll lenght = {len(df_roll["Swi6M"])}')
    print(f'swi6m max lenght = {len(df_roll["Swi6M_max"])}')
    # 3. Subsample: keep every w-th row starting at w-1 (last row of first window)
    keep   = np.arange(w - 1, len(df), w)
    df_out = df_roll.iloc[keep].reset_index(drop=True)
    print(f'df_out lenght = {len(df_out["Swi6M"])}')
    
    # 4. Align to dump: trim whichever is longer
    n_dump = len(dump_timesteps)
    n_out  = len(df_out)
    print(f'n out = {n_out} and n dump = {n_dump}')
    n_min      = min(n_dump, n_out)
    n_max      = max(n_dump, n_out)
    round_ratio = round(n_max/n_min)
    if n_out != n_dump:
        print(f"  [info] types1.dat gives {n_out} windows, dump has {n_dump} frames "
              f"— using every {round_ratio} points")
    df_out  = df_out.iloc[::round_ratio].reset_index(drop=True)
    ts_out  = np.array(dump_timesteps, dtype=np.int64)

    # Final safety trim: ensure both arrays have identical length
    n_final = min(len(ts_out), len(df_out))
    ts_out  = ts_out[:n_final]
    df_out  = df_out.iloc[:n_final].reset_index(drop=True)

    return ts_out, df_out[["A", "U", "M", "Swi6", "Swi6M","Epe1"]]


def parse_timeline(filepath):
    """
    Parse replication_timeline.dat  (CSV, header: step,event,cycle).
    Returns a list of dicts sorted by step.
    """
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


def find_event_step(events, event_name, cycle):
    """Return the timestep of the first matching (event_name, cycle) pair."""
    for e in events:
        if e["event"] == event_name and e["cycle"] == cycle:
            return e["step"]
    raise ValueError(f"Event '{event_name}' cycle {cycle} not found in timeline.")


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


def build_phase_bands(events, x_max):
    """
    Convert the event list into phase bands:
        [{'phase': str, 'start': int, 'end': int, 'cycle': int}, ...]
    Unlabelled gaps → S-phase.
    """
    bands = []
    for i, ev in enumerate(events):
        name, start, cycle = ev["event"], ev["step"], ev["cycle"]
        if name == "G2_start":
            end = _find_next(events, i, "G2_end", cycle)
            if end is not None:
                bands.append({"phase": "G2", "start": start, "end": end, "cycle": cycle})
        elif name == "Mitosis_start":
            end = _find_next(events, i, "Mitosis_end", cycle)
            if end is not None:
                bands.append({"phase": "Mitosis", "start": start, "end": end, "cycle": cycle})
        elif name == "G1_start":
            next_g2 = _find_next_event_step(events, i, "G2_start")
            end = next_g2 if next_g2 is not None else x_max
            bands.append({"phase": "G1", "start": start, "end": end, "cycle": cycle})

    bands.sort(key=lambda b: b["start"])

    filled = []
    prev_end = events[0]["step"] if events else 0
    for b in bands:
        if b["start"] > prev_end:
            filled.append({"phase": "S", "start": prev_end, "end": b["start"], "cycle": -1})
        filled.append(b)
        prev_end = max(prev_end, b["end"])
    if prev_end < x_max:
        filled.append({"phase": "S", "start": prev_end, "end": x_max, "cycle": -1})

    return filled


def draw_phase_bands(ax, bands):
    """Draw semi-transparent vertical phase bands on *ax*."""
    for b in bands:
        style = PHASE_STYLES.get(b["phase"], {"color": "grey", "alpha": 0.2})
        ax.axvspan(b["start"], b["end"],
                   color=style["color"], alpha=style["alpha"], linewidth=0)


def phase_legend_patches():
    return [
        Patch(color=v["color"], alpha=max(v["alpha"], 0.6), label=v["label"])
        for v in PHASE_STYLES.values()
    ]


# -- PROCESSING ---------------------------------------------------------------

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
             timeline_events=None,
             outfile=None):

    ts      = np.array(timesteps)
    ts_ts   = ts[::TS_STRIDE]
    ts_kymo = ts[::KYMO_STRIDE]

    arrays_ts   = [arr[::TS_STRIDE]   for arr in arrays]
    arrays_kymo = [arr[::KYMO_STRIDE] for arr in arrays]
    counts_list = [compute_counts(arr) for arr in arrays_ts]

    bounds = [0.5, 1.5, 2.5, 3.5]
    norm   = mcolors.BoundaryNorm(bounds, TYPE_CMAP.N)
    n_polymers = len(arrays)

    # Global x range — anchor on dump timesteps (ground truth).
    # rg_ts and timeline events are also included so nothing is clipped.
    # types_ts is intentionally excluded: after windowed aggregation its
    # midpoint timestamps do NOT span the full simulation range, so including
    # them would shrink or shift the shared x-axis.
    all_ts = list(ts)
    if rg_ts is not None:  all_ts += list(rg_ts)
    if timeline_events:    all_ts += [e["step"] for e in timeline_events]
    x_min, x_max = min(all_ts), max(all_ts)

    # Phase bands (built once, reused everywhere)
    bands = build_phase_bands(timeline_events, x_max) if timeline_events else None

    # ---- Build ordered row list ----
    # Each entry: (tag, height_ratio)
    rows = []
    if timeline_events:
        rows.append(("timeline", 0.40))
    for i in range(n_polymers):
        rows.append((f"polymer_{i}", 1.0))
    if types_df is not None:
        rows.append(("swi6m", 1.0))
    if rg_ts is not None:
        rows.append(("rg", 1.0))
    rows.append(("kymo", 1.8))

    height_ratios = [r[1] for r in rows]
    fig_height    = sum(h * 3.0 for h in height_ratios)/1.4

    fig = plt.figure(figsize=(8, fig_height), layout='constrained')
    gs  = fig.add_gridspec(len(rows), 1,
                           height_ratios=height_ratios,
                           hspace=0.08)

    # Create axes with sharex so limits/ticks are truly synchronised
    axs     = {}
    ref_ax  = None
    for idx, (tag, _) in enumerate(rows):
        ax = fig.add_subplot(gs[idx], sharex=ref_ax)
        if ref_ax is None:
            ref_ax = ax
        axs[tag] = ax

    # Hide x-tick labels on every panel except the very last
    last_tag = rows[-1][0]
    for tag, _ in rows:
        ax = axs[tag]
        if tag != last_tag:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(axis="x", which="both", bottom=False)
        else:
            x_ticks = [0.101e8, 0.3e8,0.5e8,0.7e8,0.9e8,1.1e8]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(['0','2','4','6','8',rf'$10\times10^5$'])
            ax.set_xlabel(r"Time ($\tau_{LJ}$)")

    # ------------------------------------------------------------------ #
    # 0. Cell cycle timeline
    # ------------------------------------------------------------------ #
    if timeline_events and bands:
        ax = axs["timeline"]
        seen_phases = set()
        for b in bands:
            style = PHASE_STYLES.get(b["phase"],
                                     {"color": "grey", "alpha": 0.4, "label": b["phase"]})
            label = style["label"] if b["phase"] not in seen_phases else None
            seen_phases.add(b["phase"])
            ax.barh(y=0,
                    width=b["end"] - b["start"],
                    left=b["start"],
                    height=0.8,
                    color=style["color"],
                    alpha=min(style["alpha"] * 2, 0.9),
                    label=label)

        span = x_max - x_min
        for b in bands:
            mid   = (b["start"] + b["end"]) / 2
            width = b["end"] - b["start"]
            if b["phase"] == "Mitosis" and b["cycle"] >= 0:
                ax.text(mid, 0, rf"M{b['cycle']+1}",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
            elif b["phase"] in ("G1", "G2", "S") and width / span > 0.03:
                ax.text(mid, 0, b["phase"],
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if b["phase"] == "S" else "#222222")
            if b['phase'] == 'G2':
                ax.axvline(b["start"], color=PHASE_STYLES["S"]['color'], linewidth=2)

        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        # ax.set_title("Cell Cycle Timeline", fontsize=10, pad=3)
        ax.legend(handles=phase_legend_patches(),
                  fontsize=7, loc="upper right", ncol=len(PHASE_STYLES))
        ax.spines[["top", "right", "bottom"]].set_visible(False)

    # ------------------------------------------------------------------ #
    # 1. Polymer type-count panels  (A / U / M)
    # ------------------------------------------------------------------ #
    for i in range(n_polymers):
        tag    = f"polymer_{i}"
        ax     = axs[tag]
        counts = counts_list[i]
        ids    = ids_list[i]

        if bands:
            draw_phase_bands(ax, bands)

        for t in [1, 2, 3]:
            ax.plot(ts_ts, counts[t],
                    color=TYPE_COLORS[t],
                    label=TYPE_LABELS[t],
                    linewidth=1.4)

        # ax.set_title(f"Polymer {i+1}  (IDs {ids[0]}–{ids[-1]})",
        #              fontsize=10, pad=3)
        ax.set_ylabel("Nucleosome type\nCount")
        ax.grid(alpha=0.25)
        y_ticks = [0,250,500]
        ax.set_yticks(y_ticks)
        # ax.legend(fontsize=8, loc="upper right")

    # ------------------------------------------------------------------ #
    # 2. Swi6M panel and Epe1
    # ------------------------------------------------------------------ #
    if types_df is not None and types_ts is not None:
        ax = axs["swi6m"]
        if bands:
            draw_phase_bands(ax, bands)

        # types_ts and types_df are pre-filtered and pre-downsampled in parse_types
        ax.plot(types_ts, types_df["Swi6M"].values,
                color=SWI6M_COLOR, linewidth=1.4, label="Swi6M")

        ax.set_ylabel("Swi6*\nCount")
        y_ticks = [0,13,26]
        ax.set_yticks(y_ticks)
        ax.grid(alpha=0.25)
        # ax.legend(fontsize=8, loc="upper right")

    # ------------------------------------------------------------------ #
    # 3. Rg panel
    # ------------------------------------------------------------------ #
    if rg_ts is not None:
        ax = axs["rg"]
        if bands:
            draw_phase_bands(ax, bands)

        ax.plot(rg_ts, rg_vals,
                color=RG_COLOR, linewidth=1.4, label=r"$R_g$")
        # ax.set_title(r"Radius of gyration $R_g$", fontsize=10, pad=3)
        ax.set_ylabel(r"$R_g^2 (\sigma^2)$")
        y_ticks = [7,16.5,26]
        ax.set_yticks(y_ticks)
        ax.grid(alpha=0.25)
        # ax.legend(fontsize=8, loc="upper right")

    # ------------------------------------------------------------------ #
    # 4. Kymograph
    # ------------------------------------------------------------------ #
    ax_kymo  = axs["kymo"]
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
                    y_offset + 0.5]
        )
        if i < n_polymers - 1:
            ax_kymo.axhline(y=y_offset + n_beads + 0.5,
                            color="black", linewidth=2)
        y_offset += n_beads

    ax_kymo.set_ylim(y_offset + 0.5, 0.5)
    y_ticks = [0,100,200,300,400,500]
    ax_kymo.set_yticks(y_ticks)
    ax_kymo.set_ylabel("Nucleosome position")
    # ax_kymo.set_title("Kymograph", fontsize=10, pad=3)
    # legend_patches = [Patch(color=TYPE_COLORS[t], label=TYPE_LABELS[t])
    #                   for t in [1, 2, 3]]
    # ax_kymo.legend(handles=legend_patches, fontsize=8, loc="upper right")

    # ---- Enforce shared x limits on every axis explicitly ----
    # sharex alone is not enough: each ax.plot() call auto-scales that axis
    # to its own data range (types_ts after windowing is shorter than the
    # full simulation).  Setting xlim on every axis after all plotting is done
    # guarantees a consistent range everywhere.
    for ax in axs.values():
        ax.set_xlim(x_min, x_max)

    # layout='constrained' on the figure handles spacing — no tight_layout needed

    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {outfile}")
    else:
        plt.show()


# -- MAIN ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot LAMMPS polymer dump.")
    ap.add_argument("dump",        nargs="?", default=DUMP_FILE,
                    help=f"Path to dump file  (default: {DUMP_FILE})")
    ap.add_argument("--rg",        default=R2_FILE,
                    help=f"Path to Rg data file  (default: {R2_FILE})")
    ap.add_argument("--timeline",  default=TIMELINE_FILE,
                    help=f"Path to replication timeline CSV  (default: {TIMELINE_FILE})")
    ap.add_argument("--types",     default=TYPES_FILE,
                    help=f"Path to types1.dat  (default: {TYPES_FILE})")
    ap.add_argument("--out",       default=None,
                    help="Save figure to this path instead of displaying it")
    args = ap.parse_args()

    # --- Resolve analysis window from timeline ----------------------------
    # Parse timeline first (needed to look up the end timestep).
    timeline_events = None
    try:
        timeline_events = parse_timeline(args.timeline)
        print(f"Reading {args.timeline} -> {len(timeline_events)} events")
    except FileNotFoundError:
        print(f"  [warn] {args.timeline} not found — skipping timeline panel")

    # Determine TS_MAX from the cell-cycle event defined in WINDOW_END_EVENT.
    TS_MAX = None
    if timeline_events and WINDOW_END_EVENT is not None:
        end_event, end_cycle = WINDOW_END_EVENT
        try:
            TS_MAX = find_event_step(timeline_events, end_event, end_cycle)
            print(f"  -> Analysis window end: {end_event} cycle {end_cycle} "
                  f"at timestep {TS_MAX}")
        except ValueError as exc:
            print(f"  [warn] {exc} — using full simulation range")

    # --- Load and filter dump --------------------------------------------
    print(f"Reading {args.dump} ...")
    timesteps_full, frames_full = parse_dump(args.dump)
    print(f"  -> {len(timesteps_full)} frames, steps {timesteps_full[0]}–{timesteps_full[-1]}")

    if TS_MAX is not None:
        filtered  = [(ts, fr) for ts, fr in zip(timesteps_full, frames_full) if ts <= TS_MAX]
        timesteps, frames = zip(*filtered) if filtered else ([], [])
        timesteps = list(timesteps)
        frames    = list(frames)
        print(f"  -> After filtering (≤ {TS_MAX}): {len(timesteps)} frames")
    else:
        timesteps, frames = timesteps_full, frames_full

    ids_list, arrays = build_arrays(frames, POLYMERS)
    for i, ids in enumerate(ids_list):
        print(f"  -> Polymer {i+1}: {len(ids)} beads")

    # --- Rg --------------------------------------------------------------
    rg_ts, rg_vals = None, None
    try:
        rg_ts, rg_vals = parse_r2(args.rg)
        if TS_MAX is not None:
            mask    = rg_ts <= TS_MAX
            rg_ts   = rg_ts[mask]
            rg_vals = rg_vals[mask]
        print(f"Reading {args.rg} -> {len(rg_ts)} data points")
    except FileNotFoundError:
        print(f"  [warn] {args.rg} not found — skipping Rg panel")

    # --- types1.dat ------------------------------------------------------
    # Align against the FULL dump timestep list so the ratio calculation is
    # correct, then trim to TS_MAX afterwards (same pattern as Rg).
    types_df, types_ts = None, None
    try:
        types_ts, types_df = parse_types(args.types, timesteps_full)
        if TS_MAX is not None:
            mask     = types_ts <= TS_MAX
            types_ts = types_ts[mask]
            types_df = types_df.iloc[:int(mask.sum())].reset_index(drop=True)
        print(f"Reading {args.types} -> {len(types_df)} windows, "
              f"steps {types_ts[0]}–{types_ts[-1]}")
    except FileNotFoundError:
        print(f"  [warn] {args.types} not found — skipping Swi6M panel")

    # --- Filter timeline events to window --------------------------------
    if timeline_events and TS_MAX is not None:
        timeline_events = [e for e in timeline_events if e["step"] <= TS_MAX]
        print(f"  -> Timeline events in window: {len(timeline_events)}")

    plot_all(timesteps, ids_list, arrays,
             rg_ts=rg_ts,       rg_vals=rg_vals,
             types_df=types_df, types_ts=types_ts,
             timeline_events=timeline_events,
             outfile=args.out)


if __name__ == "__main__":
    main()