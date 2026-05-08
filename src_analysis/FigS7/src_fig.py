# src_fig.py
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import string

folders = [
    "2polymersFullAAFullMM_Triplicate1_simBis_1e7timesteps_FullA_FullA_p2_0.00025_noise_500_swi6_400",
    "2polymersFullAAFullMM_Triplicate1_simBis_1e7timesteps_FullM_FullM_p2_0.00025_noise_500_swi6_400",
]

PRX_RC = {
    "font.family":        "serif",
    "font.size":          1,   # 8 × 1.4
    "axes.labelsize":     10,   # 9 × 1.4
    "axes.titlesize":     9,   # 8 × 1.4
    "xtick.labelsize":    9,   # 8 × 1.4
    "ytick.labelsize":    9,   # 8 × 1.4
    "legend.fontsize":    9,    # 7 × 1.4
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

mpl.rcParams.update(PRX_RC)

# Mapping of type numbers to labels/colors
type_map_i = {
    1: ("A", "#277CD1"),
    2: ("U", "#F8CB17"),
    3: ("M", "#FD1D3B"),
}

type_map_ii = {
    1: ("A", "#194D81"),
    2: ("U", "#B38F02"),
    3: ("M", "#9E0318"),
}

TYPE_COLORS = {
    1: "#2166AC",   # A  — blue
    2: "#F4C300",   # U  — yellow
    3: "#D6001C",   # M  — red
    4: "#194D81",
    5: "#B38F02",
    6: "#9E0318",
}

TYPE_LABELS = {1: "A-I", 2: "U-I", 3: "M-I",4:'A-II',5:'U-II',6:'M-II'}

TYPE_CMAP   = mcolors.ListedColormap([TYPE_COLORS[k] for k in sorted(TYPE_COLORS)])

def _label_panel(ax, idx, x=-0.18, y=1.05): #x=-0.12 before
    """PRX-style panel label slightly outside top-left of axes."""
    label = f"({string.ascii_lowercase[idx]})"
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        clip_on=False,
        zorder=10,
    )

def read_timeseries(filepath):

    timesteps = []
    data = defaultdict(list)

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):

        if lines[i].startswith("ITEM: TIMESTEP"):

            timestep = int(lines[i + 1].strip())
            timesteps.append(timestep)

            while not lines[i].startswith("ITEM: ATOMS"):
                i += 1

            i += 1

            while i < len(lines) and not lines[i].startswith("ITEM:"):
                atom_id, atom_type = map(int, lines[i].split())

                data[atom_id].append(atom_type)

                i += 1

            continue

        i += 1

    return np.array(timesteps), data


fig, axes = plt.subplots(
    2, 2,
    figsize=(7.5, 4),
    sharex=True,
    sharey=True
)

for col, folder in enumerate(folders):
    

    filepath = os.path.join(folder, "id_and_type.dat")
    timesteps, data = read_timeseries(filepath)

    groups = [
        range(1, 81),
        range(81, 161)
    ]

    for row, ids_group in enumerate(groups):

        ax = axes[row, col]
        if row==1:
            type_map=type_map_ii
        else:
            type_map=type_map_i
        for tval, (label, color) in type_map.items():

            counts = []

            for frame in range(len(timesteps)):

                c = 0

                for atom_id in ids_group:
                    if data[atom_id][frame] == tval:
                        c += 1

                counts.append(c)

            ax.plot(
                timesteps[::50],
                counts[::50],
                color=color,
                label=label
            )
            ax.set_xticks([0,0.5e8,1e8])
            ax.set_yticks([0,40,80])
            ax.grid(alpha=0.2)
        if row == 1:
            ax.set_xlabel(r"Time ($\tau_{\mathrm{LJ}}$)")
            ax.set_xticklabels([0,0.5,rf'$1 \times 10^5$'])
        if col==0:
            if row==0:
                ax.set_ylabel("Polymer I\nCount\nnucleosomal type")
            else:
                ax.set_ylabel("Polymer II\nCount\nnucleosomal type")
            ax.set_yticklabels([0,40,80])
        if row==1 and col==0:
            type_handles = [
                Patch(color=TYPE_COLORS[k],  label=TYPE_LABELS[k])
                for k in sorted(TYPE_COLORS)
            ]
            ax.legend(handles=type_handles,ncol=2,loc='center left')
        if row==0:
            if col==0:
                _label_panel(ax,col)
            else:
                _label_panel(ax,col, x=-0.08)
plt.tight_layout()
plt.savefig('supp2switching_polymers.pdf',dpi = 500)