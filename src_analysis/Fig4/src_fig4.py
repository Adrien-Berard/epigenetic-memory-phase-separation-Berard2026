from pathlib import Path
import os
"""
src_fig4.py
----------
Build Figure 4 phase-scan composite from precomputed scan CSV and types panels.
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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Divider, Size

try:
    from matplotlib.figure import Figure
    import matplotlib.svg as msvg
    SVG_AVAILABLE = True
except ImportError:
    SVG_AVAILABLE = False
from pdf2image import convert_from_path
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Matplotlib rcParams
# ---------------------------------------------------------------------------

mpl.rcParams["font.family"] = "serif"

PRX_RC = {
    "font.family":        "serif",
    "font.size":          7.5,  
    "axes.labelsize":     7, 
    "axes.titlesize":     7,   
    "xtick.labelsize":    6.5,  
    "ytick.labelsize":    6.5,  
    "legend.fontsize":    9,    
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


A4_WIDTH  = 7.5    
A4_HEIGHT = 7.5*4/9 

dt = 1e-3

swi6_200_types = DATA_ROOT / "SPombe_MatRegion_Model" / "ParameterScanDifferentSwi6" / "sim_p2_0.00025_noise_500_swi6_200" / "types1.dat" # 1 line header and then A,U,M,Swi6,Swi6M, just display A,U,M 
swi6_400_types = DATA_ROOT / "SPombe_MatRegion_Model" / "ParameterScanDifferentSwi6" / "sim_p2_0.00025_noise_500_swi6_400" / "types1.dat"
swi6_600_types = DATA_ROOT / "SPombe_MatRegion_Model" / "ParameterScanDifferentSwi6" / "sim_p2_0.00025_noise_500_swi6_600" / "types1.dat"
# diagram_path = DATA_ROOT / "SPombe_MatRegion_Model" / "GMM_PhaseDiagram" / "FinerScan" / "phase_diagram_finer.pdf"
scan_csv = DATA_ROOT / "SPombe_MatRegion_Model" / "GMM_PhaseDiagram" / "FinerScan" / "all_results.csv"

TYPE_COLORS = {
    1: "#2166AC",   # A  — blue
    2: "#F4C300",   # U  — yellow
    3: "#D6001C",   # M  — red
}
# --- load time series (adjust if your format differs) ---
df200 = pd.read_csv(swi6_200_types, comment="#",
                    names=["A", "U", "M", "Swi6", "Swi6M"])
df400 = pd.read_csv(swi6_400_types, comment="#",
                    names=["A", "U", "M", "Swi6", "Swi6M"])
df600 = pd.read_csv(swi6_600_types, comment="#",
                    names=["A", "U", "M", "Swi6", "Swi6M"])

dfscan = pd.read_csv(scan_csv)

def build_rgb_grid(fhA_grid, fhM_grid):
    H, W  = fhA_grid.shape
    rgb   = np.zeros((H, W, 3))
    green = np.array([00, 1.00, 00])
    blue  = np.array([00, 0, 1])
    red   = np.array([1, 0, 0])

    for i in range(H):
        for j in range(W):
            a = float(fhA_grid[i, j])
            m = float(fhM_grid[i, j]) 



            both_high = a * m
            a_only    = a * (1 - m)
            m_only    = (1 - a) * m

            rgb[i, j] = (both_high * green +
                         a_only    * blue  +
                         m_only    * red )

    return np.clip(rgb, 0, 1)


def _rgb_phase_page(ax, fhA_grid, fhM_grid, x_vals, y_vals, noise,
                    x_label=r"$k_2$ ($\tau_{\mathrm{LJ}}^{-1}$)", y_label='Count\ntotal Swi6', title=None):
    rgb = build_rgb_grid(fhA_grid, fhM_grid)

    # fig = plt.figure(figsize=(6, 5))

    green = np.array([0, 1, 0])
    blue  = np.array([00, 0, 1])
    red  = np.array([1, 0, 0])
    # blue  = np.array([0.10, 0.35, 0.85])
    # red   = np.array([0.85, 0.15, 0.15])

    def make_cmap(color, reverse=False):
        colors = [(0,0,0,0), (*color, 1)]
        if reverse:
            colors = colors[::-1]
            
        return mpl.colors.LinearSegmentedColormap.from_list("", colors)

    cmap_green = make_cmap(green)
    cmap_blue  = make_cmap(blue)
    cmap_red   = make_cmap(red)
    norm_green = mpl.colors.Normalize(vmin=0, vmax=1)
    norm_blue  = mpl.colors.Normalize(vmin=0, vmax=1)
    norm_red   = mpl.colors.Normalize(vmin=0, vmax=1)
    # ScalarMappables
    sm_g = mpl.cm.ScalarMappable(norm=norm_green, cmap=cmap_green)
    sm_b = mpl.cm.ScalarMappable(norm=norm_blue,  cmap=cmap_blue)
    sm_r = mpl.cm.ScalarMappable(norm=norm_red,   cmap=cmap_red)


    for sm in (sm_g, sm_b, sm_r):
        sm.set_array([])

    # --- Layout ---
    horiz = [
        Size.Scaled(0.7),   # image
        Size.Fixed(0.14),   # gap
        Size.Fixed(0.14),   # blue
        Size.Fixed(0.12),   # gap
        Size.Fixed(0.14),   # red
        Size.Fixed(0.12),   # gap
        Size.Fixed(0.14),   # green
    ]

    vert = [
        Size.Scaled(0.125),  # bottom margin (12.5%)
        Size.Scaled(0.65),   # colorbar height (75%)
        Size.Scaled(0.125),  # top margin (12.5%)
    ]

    # rect = (0.1, 0.1, 0.8, 0.8)
    # div = Divider(fig, rect, horiz, vert, aspect=False)
    
    fig = ax.figure
    pos = ax.get_position()
    rect = (pos.x0, pos.y0, pos.width, pos.height) 
    div = Divider(fig, rect, horiz, vert, aspect=False)
    ax.remove()
    # --- Axes ---
    # ax_img = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=0))

    # cax1 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=0))
    # cax2 = fig.add_axes(rect, axes_locator=div.new_locator(nx=4, ny=0))
    # cax3 = fig.add_axes(rect, axes_locator=div.new_locator(nx=6, ny=0))
    # Image
    ax_img = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=0, ny1=3))


    cax_blue  = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=1))
    cax_red   = fig.add_axes(rect, axes_locator=div.new_locator(nx=4, ny=1))
    cax_green = fig.add_axes(rect, axes_locator=div.new_locator(nx=6, ny=1))
   

    x = np.array(x_vals) / dt
    # x_log = np.log10(x)
    # # compute bin edges from centers
    # x_edges = np.zeros(len(x_log) + 1)
    # x_edges[1:-1] = 0.5 * (x_log[1:] + x_log[:-1])
    # x_edges[0] = x_log[0] - (x_log[1] - x_log[0]) / 2
    # x_edges[-1] = x_log[-1] + (x_log[-1] - x_log[-2]) / 2
    


    
    im = ax_img.imshow(
        rgb,
        aspect="equal",
        origin="lower"
    )
    n = len(x_vals)
    x_centers = np.arange(n)   # these ARE pixel centers for imshow
    pos = np.interp(np.log10(0.25e-3),
                    np.log10(x_vals),
                    np.arange(n))
    tick_idx = [0, pos, n//2, n-1]
    tick_labels = ['0.2','0.25','0.32','0.5']
    
    H = len(y_vals)
    y_200, y_400 = np.interp(200,y_vals,np.arange(H)),np.interp(400,y_vals,np.arange(H))
    ax_img.set_yticks([0,y_200, H//2, y_400,H-1])
    ax_img.set_yticklabels([f"{int(v)}" for v in [50,200,325,400,600]], fontsize = PRX_RC['xtick.labelsize'])


    ax_img.set_xticks(tick_idx)
    ax_img.set_xticklabels(tick_labels, fontsize = PRX_RC['xtick.labelsize'])
    # --- choose tick positions in DATA coordinates ---
                # vertical line (from bottom to point)
                
    ax_img.vlines(pos, -0.5, H-1, color='black', alpha=0.8, linewidth=1, zorder=1,linestyle='--')
    print(x_vals)
    
    # y_ticks = [75,225,350,425,625]

    # # --- log-spaced ticks (but axis stays linear) ---
    # x_ticks_log = [0.2, 0.275,x[len(x)//2], 0.5]
    # x_ticks_labels = [0.2, 0.25,x[len(x)//2], 0.5]



    # ax_img.set_xticks(x_ticks_log)
    # ax_img.set_xticklabels(x_ticks_labels, fontsize = PRX_RC['font.size'])

    # ax_img.set_yticks(y_ticks)
    # ax_img.set_yticklabels([f"{int(v)}" for v in [50,200,325,400,600]], fontsize = PRX_RC['font.size'])

    ax_img.set_xlabel(x_label, fontsize = PRX_RC['font.size'])
    ax_img.set_ylabel(y_label, fontsize = PRX_RC['font.size'])

    # --- vertical line now matches axis scale ---
    # k1 = 1e-1
    # ax_img.vlines(
    #     np.log10(k1),
    #     ymin=50,
    #     ymax=1050,
    #     colors='black',
    #     linestyles='--'
    # )
    star_coords = [(pos, y_200), (pos, y_400), (pos, H-1)]
    markers = [ 'o','*','s']

    def draw_stars(ax, coords, mark, color='yellow', size=100):
        xs, ys = zip(*coords)
        ax.scatter(xs, ys,
                marker=mark,
                s=size,
                c=color,
                edgecolors='black',
                linewidths=1,
                zorder=5)
        
    def draw_star_guides(ax, coords, color='black', alpha=0.8, lw=1):
        for x, y in coords:
            # horizontal line (from left to point)
            ax.hlines(y, -0.5, x, color=color, alpha=alpha, linewidth=lw, zorder=1,linestyle='--')
            
    # stars
    if star_coords is not None:
        for star_coord, marker in zip(star_coords, markers):
            if marker == '*':
                draw_stars(ax_img, [star_coord], marker,size=240)
            else:
                draw_stars(ax_img, [star_coord], marker)
            draw_star_guides(ax_img, [star_coord])
            
    # --- Colorbars ---
    cb_blue = fig.colorbar(sm_b, cax=cax_blue)
    cb_red  = fig.colorbar(sm_r, cax=cax_red)
    cb_green = fig.colorbar(sm_g, cax=cax_green)

    cb_green.set_ticks([0, 1])
    cb_green.ax.tick_params(labelsize=8)
    cb_blue.set_ticks([])
    cb_red.set_ticks([])
    
    cb_blue.ax.set_title(rf"$f^A_A \cdot f^A_M$", fontsize=9, pad=6,rotation=45)
    cb_red.ax.set_title(rf"$f^M_M \cdot f^M_A$", fontsize=9, pad=6,rotation=45)
    cb_green.ax.set_title(rf"$f^A_A \cdot f^M_M$", fontsize=9, pad=6,rotation=45)
    
    # cb_blue.ax.set_title("A", fontsize=8)
    # cb_red.ax.set_title("M", fontsize=8)
    # cb_green.ax.set_title("balance", fontsize=8)
    # for cb in (cb1, cb2, cb3):
    #     cb.ax.tick_params(labelsize=9, length=3)

    _label_panel(ax_img,'(a)')
    
    return fig

def _label_panel(ax, label, x=-0.19, y=1.01): #x=-0.12 before
    """ panel label slightly outside top-left of axes."""
    label = label
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        clip_on=False,
        zorder=10,fontsize=PRX_RC['font.size']
    )
    
# --- figure layout ---
fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
fig.subplots_adjust(bottom=0.15)
gs = gridspec.GridSpec(3, 2, width_ratios=[1.6, 1.4], hspace=0.2, wspace=0.45)

# =========================
# A: phase diagram (left)
# =========================

axA = fig.add_subplot(gs[:, 0])

fhA_grid = dfscan.pivot(index="swi6", columns="p2", values="all_tauA_startA").to_numpy()
fhM_grid = dfscan.pivot(index="swi6", columns="p2", values="all_tauM_startM").to_numpy()
x_vals = sorted(dfscan['p2'].unique())
y_vals = sorted(dfscan['swi6'].unique())

_rgb_phase_page(axA, fhA_grid, fhM_grid, x_vals, y_vals, noise=500,x_label=r"$k_2$ ($\tau_{\mathrm{LJ}}^{-1}$)", y_label='Count\ntotal Swi6', title=None)

# =========================
# B: stacked time series
# =========================
max_len = min(len(df600['A']),len(df400['A']),len(df200['A']))
max_timestep = 176950000 # from dump 600 swi6
first_timestep = 1000000 # from dump 600 swi6
duration = max_timestep - first_timestep
conversion_frames_to_step = int(duration / max_len) # 500


cut_steps = 15e7
cut_frames = int(cut_steps / conversion_frames_to_step)

xticks_frames = [
    0,
    int(7.5e4 * 1e3 / 500),
    int(15e4 * 1e3 / 500)
]
xlabels = [0,'7.5',rf'$15 \times 10^4$'] # in time



markers = [ 'o','*','s'] # 600,400, 200

def draw_star(ax, coords, mark, color='yellow', size=120):
    xs, ys = coords[0],coords[1]
    ax.scatter(xs, ys,
            marker=mark,
            s=size,
            c=color,
            edgecolors='black',
            linewidths=1,
            zorder=5)

# stars



axB1 = fig.add_subplot(gs[0, 1])
_label_panel(axB1,'(b)')
axB2 = fig.add_subplot(gs[1, 1], sharex=axB1)
axB3 = fig.add_subplot(gs[2, 1], sharex=axB1)

# --- SWI6 600 ---
axB1.plot( df600['A'].iloc[:cut_frames:1000], label ='A', color = TYPE_COLORS[1])
axB1.plot( df600['U'].iloc[:cut_frames:1000], label ='U', color = TYPE_COLORS[2])
axB1.plot( df600['M'].iloc[:cut_frames:1000], label ='M', color = TYPE_COLORS[3])
axB1.set_ylabel("Count\nnucleosomal\ntype", fontsize = PRX_RC['font.size'], rotation=90)
axB1.grid(alpha=0.20, linewidth=0.4)
axB1.set_ylim(0,80)
axB1.set_yticks([0,40,80])
axB1.set_yticklabels([0,40,80], fontsize = PRX_RC['xtick.labelsize'])
axB1.tick_params(labelbottom=False)
draw_star(axB1, [10,70], markers[2], color = 'yellow', size = 120)

# --- SWI6 400 ---
axB2.plot( df400['A'].iloc[:cut_frames:1000], label ='A', color = TYPE_COLORS[1])
axB2.plot( df400['U'].iloc[:cut_frames:1000], label ='U', color = TYPE_COLORS[2])
axB2.plot( df400['M'].iloc[:cut_frames:1000], label ='M', color = TYPE_COLORS[3])
axB2.set_ylabel("Count\nnucleosomal\ntype", fontsize = PRX_RC['font.size'], rotation=90)
axB2.grid(alpha=0.20, linewidth=0.4)
axB2.set_yticks([0,40,80])
axB2.set_yticklabels([0,40,80], fontsize = PRX_RC['xtick.labelsize'])
axB2.set_ylim(0,80)
axB2.tick_params(labelbottom=False)
draw_star(axB2, [10,70], markers[1], color = 'yellow', size = 300)

# --- SWI6 200 ---
axB3.plot( df200['A'].iloc[:cut_frames:1000], label ='A', color = TYPE_COLORS[1])
axB3.plot( df200['U'].iloc[:cut_frames:1000], label ='U', color = TYPE_COLORS[2])
axB3.plot( df200['M'].iloc[:cut_frames:1000], label ='M', color = TYPE_COLORS[3])
axB3.set_ylabel("Count\nnucleosomal\ntype", fontsize = PRX_RC['font.size'], rotation=90)
axB3.grid(alpha=0.20, linewidth=0.4)
axB3.set_yticks([0,40,80])
axB3.set_yticklabels([0,40,80], fontsize = PRX_RC['xtick.labelsize'])
axB3.set_ylim(0,80)
axB3.set_xticks(xticks_frames)
axB3.set_xticklabels(xlabels, fontsize = PRX_RC['xtick.labelsize'])
axB3.set_xlabel(r"Time ($\tau_{\mathrm{LJ}}$)", fontsize = PRX_RC['font.size'],labelpad = 2)
draw_star(axB3, [10,70], markers[0], color = 'yellow', size = 120)

# plt.tight_layout()
plt.savefig('FigureScan_0605.pdf',dpi=500)