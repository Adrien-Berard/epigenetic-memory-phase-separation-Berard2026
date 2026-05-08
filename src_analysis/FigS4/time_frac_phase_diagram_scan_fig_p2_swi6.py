import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Divider, Size

import argparse
import csv
from dataclasses import dataclass

try:
    from sklearn.mixture import GaussianMixture
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: scikit-learn. Install with `pip install scikit-learn`."
    ) from e

mpl.rcParams["font.family"] = "serif"
# ============================================================
# CONFIG
# ============================================================
CONDENSED_DIR = "/home/adrien/SPombe_MatRegion_Model/ParameterScanDifferentSwi6Differentp2-13-03-26-Revision"
DILUTE_DIR    = "/home/adrien/SPombe_MatRegion_Model/ParameterScanDifferentSwi6Differentp2-13-03-26-Revision"
OUTPUT_DIR    = "/home/adrien/SPombe_MatRegion_Model/GMM_PhaseDiagram/Fullp2vsswi6/13-03"

NOISE_VALUES   = [250,500,1000]
FRAC_HI_THRESH = 0.35
STRIDE         = 100

# # X-axis and Y-axis parameter names (used for directory parsing and labels)
# # Change these to match your directory naming convention, e.g. "p2"/"swi6"
# X_PARAM = "p1"    # name of the parameter that varies along x-axis
# Y_PARAM = "p2"  # name of the parameter that varies along y-axis

# CONDENSED_DIR = "/home/adrien/SPombe_MatRegion_Model/ParameterScanDifferentSwi6Differentp2-13-03-26-Revision"
# DILUTE_DIR    = "/home/adrien/SPombe_MatRegion_Model/ParameterScanDifferentSwi6Differentp2-13-03-26-Revision"

# CONDENSED_DIR = '/home/adrien/SPombe_MatRegion_Model/P1_P2_FINE_scan_swi6_400/Noise500_finefiner'
# DILUTE_DIR    = '/home/adrien/SPombe_MatRegion_Model/P1_P2_FINE_scan_swi6_400/Noise500_finefiner'
# OUTPUT_DIR    = "/home/adrien/SPombe_MatRegion_Model/GMM_PhaseDiagram/p1vsp2/finerScan/OtherRegion"

# NOISE_VALUES   = [500]
# FRAC_HI_THRESH = 0.35
# STRIDE         = 100

# X-axis and Y-axis parameter names (used for directory parsing and labels)
# Change these to match your directory naming convention, e.g. "p2"/"swi6"
X_PARAM = "p2"    # name of the parameter that varies along x-axis
Y_PARAM = "swi6"  # name of the parameter that varies along y-axis

dt    = 1e-3
NOISE_RATES    = 1/(np.array(NOISE_VALUES)*dt)

# Sub-folder names inside  Noise{N}/  for the two initial conditions.
FULLA_SUBDIR = "FullA"
# FULLM_SUBDIR = "FullM_Swi6MStart"
FULLM_SUBDIR = "FullM"


# ============================================================
# CORE FUNCTIONS
# ============================================================
def compute_phi(A, U, M):
    N = A + U + M
    N[N == 0] = np.nan
    return (A - M) / N


@dataclass(frozen=True)
class GmmDiagnostics:
    threshold: float
    mu_lo:     float
    mu_hi:     float


def _fit_gmm_1d(phi_all, random_state=0):
    x = phi_all.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, covariance_type="full",
                           random_state=random_state)
    gmm2 = GaussianMixture(n_components=2, covariance_type="full",
                           random_state=random_state)
    gmm1.fit(x)
    gmm2.fit(x)
    bic_delta = float(gmm1.bic(x) - gmm2.bic(x))
    return gmm2, bic_delta


def _gmm_order_params_1d(gmm2):
    means   = gmm2.means_.reshape(-1)
    vars_   = gmm2.covariances_.reshape(-1)
    weights = gmm2.weights_.reshape(-1)
    order   = np.argsort(means)
    lo, hi  = int(order[0]), int(order[1])
    return (
        float(means[lo]),   float(means[hi]),
        float(vars_[lo]),   float(vars_[hi]),
        float(weights[lo]), float(weights[hi]),
        lo, hi,
    )


def _gmm_intersection_threshold(mu1, mu2, var1, var2, w1, w2):
    mu_lo, mu_hi = (mu1, mu2) if mu1 < mu2 else (mu2, mu1)
    if var1 <= 0 or var2 <= 0:
        return 0.5 * (mu_lo + mu_hi)
    a = (1.0 / var2) - (1.0 / var1)
    b = (-2.0 * mu2 / var2) + (2.0 * mu1 / var1)
    c = (mu2*mu2/var2) - (mu1*mu1/var1) \
        - 2.0 * np.log((w2 * np.sqrt(var1)) / (w1 * np.sqrt(var2)))
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return 0.5 * (mu_lo + mu_hi)
        x = -c / b
        return float(x if mu_lo < x < mu_hi else 0.5 * (mu_lo + mu_hi))
    disc = b*b - 4*a*c
    if disc < 0:
        return 0.5 * (mu_lo + mu_hi)
    sqrt_disc = np.sqrt(disc)
    for r in ((-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)):
        if mu_lo < r < mu_hi:
            return float(r)
    return 0.5 * (mu_lo + mu_hi)


def compute_gmm_diagnostics(phi_all, gmm2, bic_delta):
    mu_lo, mu_hi, var_lo, var_hi, w_lo, w_hi, lo, hi = _gmm_order_params_1d(gmm2)
    threshold = _gmm_intersection_threshold(mu_lo, mu_hi, var_lo, var_hi, w_lo, w_hi)
    return GmmDiagnostics(threshold=threshold, mu_lo=mu_lo, mu_hi=mu_hi), lo, hi


def frac_hi(labels):
    n = labels.size
    if n == 0:
        return 0.0
    return float(np.sum(labels == 1)) / n


# ============================================================
# DIRECTORY PARSING
# ============================================================
def _parse_sim_dir(name, x_param, y_param):
    """
    Parse a directory name of the form
        sim_<key>_<val>_<key>_<val>[_<key>_<val>...]
    and return (x_val, y_val) as floats for the requested params.
    Returns None if the name does not match or either param is absent.
    """
    if not name.startswith("sim_"):
        return None
    parts = name.split("_")
    # parts[0] == "sim"; then key/value pairs
    params = {}
    i = 1
    while i < len(parts) - 1:
        key = parts[i]
        val = parts[i + 1]
        params[key] = val
        i += 2
    if x_param not in params or y_param not in params:
        return None
    try:
        x_val = float(params[x_param])
        # y_val = float(params[y_param])
        y_val = int(params[y_param])
    except ValueError:
        return None
    return x_val, y_val


# ============================================================
# LOAD SIMULATION
# ============================================================
def load_sim(path):
    if not os.path.exists(path):
        return None
    arr = np.loadtxt(path, comments="#", delimiter=',',
                     usecols=(0, 1, 2), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    A = arr[:, 0].astype(np.float64)
    U = arr[:, 1].astype(np.float64)
    M = arr[:, 2].astype(np.float64)
    if A.size < 20:
        print(f"Warning: short simulation {path}")
    return {"A": A, "U": U, "M": M}


# ============================================================
# PROCESS POINT  (top-level for ProcessPoolExecutor pickling)
# ============================================================
def process_point(args):
    (noise, x_val, y_val, sim_dir_name,
     dilute_dir, condensed_dir,
     stride, stable_thresh,
     x_param, y_param,
     fulla_subdir, fullm_subdir) = args

    path_A = os.path.join(dilute_dir,    f"Noise{noise}", fulla_subdir, sim_dir_name, "types1.dat")
    path_M = os.path.join(condensed_dir, f"Noise{noise}", fullm_subdir, sim_dir_name, "types1.dat")

    arrays_A = load_sim(path_A)
    arrays_M = load_sim(path_M)

    if arrays_A is None or arrays_M is None:
        missing = [p for p, a in [(path_A, arrays_A), (path_M, arrays_M)] if a is None]
        print(f"  Missing files: {missing}")
        return (noise, x_val, y_val, 0.0, 0.0, None, None, None)

    phi_A   = compute_phi(**arrays_A)[::stride]
    phi_M   = compute_phi(**arrays_M)[::stride]
    phi_all = np.concatenate([phi_A, phi_M])
    phi_all = phi_all[np.isfinite(phi_all)]

    if phi_all.size < 80:
        print(f"  Warning: simulation too short {x_param}={x_val} {y_param}={y_val} noise={noise}")
        return (noise, x_val, y_val, 0.0, 0.0, phi_A, phi_M, None)


    phi_combined = np.concatenate([phi_A, phi_M])

    tau_A_both = len(phi_combined[phi_combined > 0])/len(phi_combined)
    tau_M_both = len(phi_combined[phi_combined < 0])/len(phi_combined)
    
    tau_A_startA = len(phi_A[phi_A > 0])/len(phi_A)
    tau_M_startM = len(phi_M[phi_M < 0])/len(phi_M)

    return (noise, x_val, y_val, phi_A, phi_M, tau_A_both, tau_M_both, tau_A_startA , tau_M_startM )


# ============================================================
# RGB OVERLAY GRID
# ============================================================
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


# ============================================================
# PLOTTING
# ============================================================
def _fmt(v):
    """Format axis tick: scientific notation for floats, plain string otherwise."""
    if isinstance(v, float):
        return f"{v/dt:.2f}"
    return str(v)

def sci_label(val):
    exp = np.log10(val)
    if val == 1.0:
        return rf"$10^0$"
    # Check if exponent is (almost) integer
    if np.isclose(exp, round(exp), atol=1e-8):
        return rf"$10^{{{int(round(exp))}}}$"
    

    # Otherwise: split into prefactor × 10^n
    n = int(np.floor(exp))
    a = val / (10**n)
    
    return rf"${a:.1f}.10^{{{n}}}$"


def _rgb_phase_page(pdf, fhA_grid, fhM_grid, x_vals, y_vals, noise,
                    x_label="x", y_label="y", title=None):
    
    # _rgb_phase_plot(ax, fhA_grid, fhM_grid,
    #                     x_vals, y_vals, noise,
    #                 x_label="x", y_label="y", title=None)
    rgb = build_rgb_grid(fhA_grid, fhM_grid)

    fig = plt.figure(figsize=(6, 5))

    # Your colors
    green = np.array([0.10, 1.00, 0.10])
    blue  = np.array([0.10, 0.35, 0.85])
    red   = np.array([0.85, 0.15, 0.15])

    def make_cmap(color, reverse=False):
        colors = [(0,0,0,0), (*color, 1)]
        if reverse:
            colors = colors[::-1]
            
        return mpl.colors.LinearSegmentedColormap.from_list("", colors)

    def make_cmap_symmetric(color):
        return mpl.colors.LinearSegmentedColormap.from_list(
            "",
            [
                (0.0, (*color, 1.0)),   # -1 → strong
                (0.5, (0, 0, 0, 0)),    #  0 → transparent
                (1.0, (*color, 1.0)),   # +1 → strong
            ]
        )
        
    cmap_green = make_cmap_symmetric(green)
    cmap_blue  = make_cmap(blue)
    cmap_red   = make_cmap(red, reverse=True)
    norm_green = mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    norm_blue  = mpl.colors.Normalize(vmin=0, vmax=1)
    norm_red   = mpl.colors.Normalize(vmin=-1, vmax=0)
    # ScalarMappables
    sm_g = mpl.cm.ScalarMappable(norm=norm_green, cmap=cmap_green)
    sm_b = mpl.cm.ScalarMappable(norm=norm_blue,  cmap=cmap_blue)
    sm_r = mpl.cm.ScalarMappable(norm=norm_red,   cmap=cmap_red)

    for sm in (sm_g, sm_b, sm_r):
        sm.set_array([])

    # --- Layout ---
    horiz = [
        Size.Scaled(1.0),   # image
        Size.Fixed(0.08),   # gap
        Size.Fixed(0.14),   # blue/red column
        Size.Fixed(0.06),   # gap
        Size.Fixed(0.14),   # green
    ]

    vert = [
        Size.Scaled(1.0),   # bottom (red)
        Size.Fixed(0.05),   # small gap
        Size.Scaled(1.0),   # top (blue)
    ]

    rect = (0.1, 0.1, 0.8, 0.8)
    div = Divider(fig, rect, horiz, vert, aspect=False)

    # --- Axes ---
    # ax_img = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=0))

    # cax1 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=0))
    # cax2 = fig.add_axes(rect, axes_locator=div.new_locator(nx=4, ny=0))
    # cax3 = fig.add_axes(rect, axes_locator=div.new_locator(nx=6, ny=0))
    # Image
    ax_img = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=0, ny1=3))

    # Blue (top half)
    cax_blue = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=2))

    # Red (bottom half)
    cax_red = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=0))

    # Green (full height)
    cax_green = fig.add_axes(rect, axes_locator=div.new_locator(nx=4, ny=0, ny1=3))

    x = np.array(x_vals) / dt
    x_log = np.log10(x)
    # compute bin edges from centers
    x_edges = np.zeros(len(x_log) + 1)
    x_edges[1:-1] = 0.5 * (x_log[1:] + x_log[:-1])
    x_edges[0] = x_log[0] - (x_log[1] - x_log[0]) / 2
    x_edges[-1] = x_log[-1] + (x_log[-1] - x_log[-2]) / 2
    

    
    im = ax_img.imshow(
        rgb,
        aspect="auto",
        origin="lower",
        extent=[
            x_edges[0], x_edges[-1],
            50, 1050
        ]
    )

    # --- choose tick positions in DATA coordinates ---

    y_ticks = [100,550,1000]

    # --- log-spaced ticks (but axis stays linear) ---
    x_ticks_log = np.linspace(x_log[0], x_log[-1], 5)
    x_ticks = 10**x_ticks_log

    ax_img.set_xticks(x_ticks_log)
    ax_img.set_xticklabels([sci_label(v) for v in x_ticks])

    ax_img.set_yticks(y_ticks)
    ax_img.set_yticklabels([f"{int(v)}" for v in [100,550,1000]])

    ax_img.set_xlabel(x_label, fontsize=13)
    ax_img.set_ylabel(y_label, fontsize=13)

    # --- vertical line now matches axis scale ---
    k1 = 1e-1
    ax_img.vlines(
        np.log10(k1),
        ymin=50,
        ymax=1050,
        colors='black',
        linestyles='--'
    )

    # --- Colorbars ---
    cb_blue = fig.colorbar(sm_b, cax=cax_blue)
    cb_red  = fig.colorbar(sm_r, cax=cax_red)
    cb_green = fig.colorbar(sm_g, cax=cax_green)

    cb_green.set_ticks([-1, 0, 1])
    cb_green.ax.tick_params(labelsize=8)
    cb_blue.set_ticks([])
    cb_red.set_ticks([])
    
    # cb_blue.ax.set_title("A", fontsize=8)
    # cb_red.ax.set_title("M", fontsize=8)
    # cb_green.ax.set_title("balance", fontsize=8)
    # for cb in (cb1, cb2, cb3):
    #     cb.ax.tick_params(labelsize=9, length=3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# =========================
# helpers
# =========================

def draw_square(ax, x_min, x_max, y_min, y_max,
                color='white', lw=2):
    import matplotlib.patches as patches
    rect = patches.Rectangle((x_min, y_min),
                             x_max - x_min,
                             y_max - y_min,
                             fill=False,
                             edgecolor=color,
                             linewidth=lw)
    ax.add_patch(rect)


def draw_stars(ax, coords, mark, color='yellow', size=100):
    xs, ys = zip(*coords)
    ax.scatter(xs, ys,
            marker=mark,
            s=size,
            c=color,
            edgecolors='black',
            linewidths=1,
            zorder=5)


# =========================
# core plot (NO colorbars here)
# =========================

def _rgb_phase_plot(ax_img, fhA_grid, fhM_grid,
                    x_vals, y_vals, noise,
                    x_label="x", y_label="Count total Swi6"):

    rgb = build_rgb_grid(fhA_grid, fhM_grid)

    x = np.array(x_vals) / dt
    x_log = np.log10(x)

    # bin edges
    x_edges = np.zeros(len(x_log) + 1)
    x_edges[1:-1] = 0.5 * (x_log[1:] + x_log[:-1])
    x_edges[0] = x_log[0] - (x_log[1] - x_log[0]) / 2
    x_edges[-1] = x_log[-1] + (x_log[-1] - x_log[-2]) / 2

    ax_img.imshow(
        rgb,
        aspect="auto",
        origin="lower",
        extent=[x_edges[0], x_edges[-1], 50, 1050]
    )

    # ticks
    y_ticks = [100, 550, 1000]
    x_ticks_log = np.linspace(x_log[0], x_log[-1], 5)
    x_ticks = 10**x_ticks_log

    ax_img.set_xticks(x_ticks_log)
    ax_img.set_xticklabels([sci_label(v) for v in x_ticks])

    ax_img.set_yticks(y_ticks)
    ax_img.set_yticklabels([f"{int(v)}" for v in y_ticks])

    ax_img.set_ylabel(y_label, fontsize=12)

    # vertical reference line
    k1 = 1e-1
    ax_img.vlines(np.log10(k1),
                  ymin=50, ymax=1050,
                  colors='black',
                  linestyles='--')
    def noise_to_rate(noise):
        return 1/(noise*dt)
    ax_img.set_title(
    rf"$\Gamma = {noise_to_rate(noise)}\,(\tau_{{\mathrm{{LJ}}}}^{{-1}})$"
)


# =========================
# main function
# =========================

def plot_phase_diagrams(all_fhA, all_fhM, noise_values,
                        x_param, y_param,
                        output_path,
                        star_coords=None,
                        square_index=None,
                        square_bounds=None):

    # sort noise (HIGH on top)
    noise_values = sorted(noise_values, reverse=True)

    fig, axes = plt.subplots(len(noise_values), 1,
                             figsize=(6, 4 * len(noise_values)),
                             sharex=True)

    if len(noise_values) == 1:
        axes = [axes]

    # =========================
    # SAME colormap setup
    # =========================

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

    sm_g = mpl.cm.ScalarMappable(norm=norm_green, cmap=cmap_green)
    sm_b = mpl.cm.ScalarMappable(norm=norm_blue,  cmap=cmap_blue)
    sm_r = mpl.cm.ScalarMappable(norm=norm_red,   cmap=cmap_red)

    for sm in (sm_g, sm_b, sm_r):
        sm.set_array([])

    # =========================
    # loop over subplots
    # =========================
    for i, (ax, noise) in enumerate(zip(axes, noise_values)):

        noise_keys = [k for k in all_fhA if k[0] == noise]
        if not noise_keys:
            continue

        x_vals = sorted(set(k[1] for k in noise_keys))
        y_vals = sorted(set(k[2] for k in noise_keys))

        x_to_j = {v: j for j, v in enumerate(x_vals)}
        y_to_i = {v: i for i, v in enumerate(y_vals)}

        rows, cols = len(y_vals), len(x_vals)
        fhA_grid = np.full((rows, cols), np.nan)
        fhM_grid = np.full((rows, cols), np.nan)

        for key in noise_keys:
            _, x_val, y_val = key
            i_idx, j_idx = y_to_i[y_val], x_to_j[x_val]
            fhA_grid[i_idx, j_idx] = all_fhA[key]
            fhM_grid[i_idx, j_idx] = all_fhM[key]

        _rgb_phase_plot(ax, fhA_grid, fhM_grid,
                        x_vals, y_vals, noise)
        markers = [ 'o','*','s']
        # stars 
        if star_coords is not None:
            if noise == '500' or noise==500:
                for star_coord, marker in zip(star_coords, markers):
                    if marker == '*':
                        draw_stars(ax, [star_coord], marker,size=240)
                    else:
                        draw_stars(ax, [star_coord], marker)

        # square on chosen subplot
        if square_index is not None and i == square_index:
            draw_square(ax, *square_bounds)

    # =========================
    # labels
    # =========================
    if x_param == 'p2':
        x_lab = r"$k_2$ ($\tau_{\mathrm{LJ}}^{-1}$)"
    elif x_param == 'p1':
        x_lab = r"$k_1$ ($\tau_{\mathrm{LJ}}^{-1}$)"
    else:
        x_lab = 'Count total Swi6'

    axes[-1].set_xlabel(x_lab,fontsize=11)

    # =========================
    # colorbars (shared)
    # =========================
    # leave space for right panel
    plt.subplots_adjust(right=0.80)

    # create a dedicated invisible "panel axis"
    panel = fig.add_axes([0.82, 0.15, 0.16, 0.7])
    panel.axis("off")

    # ---- internal layout inside panel ----
    # everything is in panel-relative coordinates

    # sizes
    bar_w = 0.18
    gap   = 0.01

    # vertical centering block
    total_h = 0.75
    bottom  = 0.5 - total_h / 2

    # positions (centered stack)
    x_blue  = 0.05
    x_red   = 0.05
    x_green = 0.55

    h_blue  = total_h * 0.34
    h_red   = total_h * 0.34
    h_green = total_h * 0.70   # green spans larger, like your original

    # align red/blue so they look continuous


    y_green = 0.5 - h_green / 2   # centered
    y_red   = y_green
    y_blue  = y_green + h_red + gap

    # =========================
    # create axes inside panel
    # =========================
    # cax_blue  = panel.inset_axes([x_blue, y_blue, bar_w, h_blue])
    # cax_red   = panel.inset_axes([x_red,  y_red,  bar_w, h_red])
    # cax_green = panel.inset_axes([x_green, y_green, bar_w, h_green])

    # =========================
    # colorbars
    # =========================
    # cb_blue  = fig.colorbar(sm_b, cax=cax_blue)
    # cb_red   = fig.colorbar(sm_r, cax=cax_red)
    # cb_green = fig.colorbar(sm_g, cax=cax_green)

    # # styling (important for “continuous red/blue feel”)
    # cb_blue.set_ticks([])
    # cb_red.set_ticks([])

    # cb_green.set_ticks([-1, 0, 1])
    # cb_green.ax.tick_params(labelsize=10)

    # =========================
    # save
    # =========================
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    
# ============================================================
# MAIN LOOP
# ============================================================
def run(dilute_dir, condensed_dir, output_dir, noise_values,
        stride, stable_thresh, max_workers,
        x_param=X_PARAM, y_param=Y_PARAM,
        fulla_subdir=FULLA_SUBDIR, fullm_subdir=FULLM_SUBDIR):

    os.makedirs(output_dir, exist_ok=True)

    all_tauA_both = {}
    all_tauM_both = {}
    all_tauA_startA = {}
    all_tauM_startM = {}

    for noise in noise_values:
        print(f"\n=== Noise {noise} ===")
        base = os.path.join(dilute_dir, f"Noise{noise}", fulla_subdir)
        if not os.path.isdir(base):
            print(f"  Directory not found: {base}  — skipping noise={noise}")
            continue

        pairs = []
        for sim_dir in os.listdir(base):
            parsed = _parse_sim_dir(sim_dir, x_param, y_param)
            if parsed is None:
                continue
            x_val, y_val = parsed
            pairs.append((
                noise, x_val, y_val, sim_dir,
                dilute_dir, condensed_dir,
                stride, stable_thresh,
                x_param, y_param,
                fulla_subdir, fullm_subdir,
            ))

        if not pairs:
            print(f"  No simulation directories found under {base}")
            continue

        print(f"  Found {len(pairs)} simulation directories")

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(process_point, p): p for p in pairs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    orig = futures[future]
                    print(f"  process_point failed for {orig[:3]}: {exc}")
                    continue

                n_res, x_res, y_res, phi_A, phi_M, tau_A_both, tau_M_both, tau_A_startA , tau_M_startM  = result
                key = (n_res, x_res, y_res)
                all_tauA_both[key] = tau_A_both
                all_tauM_both[key] = tau_M_both
                all_tauA_startA[key] = tau_A_startA 
                all_tauM_startM[key] = tau_M_startM

        print(f"  Processed {len([k for k in all_tauA_both if k[0] == noise])} points")

    if not all_tauA_both:
        print("\nNo results collected — check your paths and parameter names.")
        return

    # ============================================================
    # CSV
    # ============================================================
    csv_path = os.path.join(output_dir, "all_results.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noise", x_param, y_param, "frac_hi_A", "frac_hi_M"])
        for key in sorted(all_tauA_both.keys()):
            noise, x_val, y_val = key
            writer.writerow([noise, x_val, y_val,
                             all_tauA_both.get(key), all_tauA_both.get(key)])
    print(f"\nSaved CSV: {csv_path}")

    # ============================================================
    # PDF — one RGB phase diagram per noise value
    # ============================================================
    # pdf_path = os.path.join(output_dir, "phase_diagrams.pdf")
    # with PdfPages(pdf_path) as pdf:
    #     for noise in noise_values:
    #         noise_keys = [k for k in all_fhA if k[0] == noise]
    #         if not noise_keys:
    #             continue

    #         x_vals = sorted(set(k[1] for k in noise_keys))
    #         y_vals = sorted(set(k[2] for k in noise_keys))
    #         x_to_j = {v: j for j, v in enumerate(sorted(x_vals))}
    #         # y_to_i = {v: i for i, v in enumerate(sorted(y_vals, reverse = True))}
    #         y_to_i = {v: i for i, v in enumerate(sorted(y_vals))}

    #         rows, cols = len(y_vals), len(x_vals)
    #         fhA_grid   = np.full((rows, cols), np.nan, dtype=np.float32)
    #         fhM_grid   = np.full((rows, cols), np.nan, dtype=np.float32)

    #         for key in noise_keys:
    #             _, x_val, y_val = key
    #             i, j = y_to_i[y_val], x_to_j[x_val]
    #             fhA_grid[i, j] = all_fhA[key]
    #             fhM_grid[i, j] = all_fhM[key]
            
    #         if x_param == 'p2':
    #             x_lab = r"$k_2$ ($\tau_{\mathrm{LJ}}^{-1}$)"
    #         elif x_param == 'p1':
    #             x_lab = r"$k_1$ ($\tau_{\mathrm{LJ}}^{-1}$)"
    #         else:
    #             x_lab = 'Count\ntotal Swi6'
                
    #         if y_param == 'p2':
    #             y_lab = r"$k_2$ ($\tau_{\mathrm{LJ}}^{-1}$)"
    #         elif y_param == 'p1':
    #             y_lab = r"$k_1$ ($\tau_{\mathrm{LJ}}^{-1}$)"
    #         else:
    #             y_lab = 'Count\ntotal Swi6'
                
    #         _rgb_phase_page(pdf, fhA_grid, fhM_grid, x_vals, y_vals, noise,
    #                         x_label=x_lab, y_label=y_lab)

    # print(f"Saved PDF: {pdf_path}")
    plot_phase_diagrams(
    all_tauA_startA,
    all_tauM_startM,
    noise_values,
    x_param,
    y_param,
    "time_frac_phase_diagrams_full_NOcbar.svg",

    star_coords=[(np.log10(0.37), 200), (np.log10(0.37), 400), (np.log10(0.37), 600)],

    square_index=1,  # middle subplot
    square_bounds=(np.log10(0.25), np.log10(0.5), 50, 650)
)

# def plot_phase_subplots(all_fhA, all_fhM, noise_values,
#                        x_param, y_param,
#                        output_path):

#     import numpy as np
#     import matplotlib.pyplot as plt

#     # sort noise values (highest on top)
#     noise_values = sorted(noise_values, reverse=True)

#     fig, axes = plt.subplots(len(noise_values), 1,
#                              figsize=(6, 4 * len(noise_values)),
#                              sharex=True)

#     if len(noise_values) == 1:
#         axes = [axes]

#     for ax, noise in zip(axes, noise_values):

#         noise_keys = [k for k in all_fhA if k[0] == noise]
#         if not noise_keys:
#             continue

#         x_vals = sorted(set(k[1] for k in noise_keys))
#         y_vals = sorted(set(k[2] for k in noise_keys))

#         x_to_j = {v: j for j, v in enumerate(x_vals)}
#         y_to_i = {v: i for i, v in enumerate(y_vals)}

#         rows, cols = len(y_vals), len(x_vals)
#         fhA_grid = np.full((rows, cols), np.nan, dtype=np.float32)
#         fhM_grid = np.full((rows, cols), np.nan, dtype=np.float32)

#         for key in noise_keys:
#             _, x_val, y_val = key
#             i, j = y_to_i[y_val], x_to_j[x_val]
#             fhA_grid[i, j] = all_fhA[key]
#             fhM_grid[i, j] = all_fhM[key]

#         # your existing plotting function, but now pass ax
#         _rgb_phase_plot(ax, fhA_grid, fhM_grid,
#                         x_vals, y_vals, noise)

#         ax.set_ylabel(f"{noise=}")

#     # shared x-label
#     if x_param == 'p2':
#         x_lab = r"$k_2$ ($\tau_{\mathrm{LJ}}^{-1}$)"
#     elif x_param == 'p1':
#         x_lab = r"$k_1$ ($\tau_{\mathrm{LJ}}^{-1}$)"
#     else:
#         x_lab = 'Count total Swi6'

#     axes[-1].set_xlabel(x_lab)

#     plt.tight_layout()
#     plt.savefig(output_path)
#     print(f"Saved: {output_path}")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMM-based RGB phase diagram for S. pombe mating-region model."
    )
    parser.add_argument("--dilute-dir",      default=DILUTE_DIR)
    parser.add_argument("--condensed-dir",   default=CONDENSED_DIR)
    parser.add_argument("--output-dir",      default=OUTPUT_DIR)
    parser.add_argument("--noise",           nargs="*", type=int, default=NOISE_VALUES)
    parser.add_argument("--stride",          type=int,   default=STRIDE)
    parser.add_argument("--frac-hi-thresh",  type=float, default=FRAC_HI_THRESH,
                        help="Min fraction of time in high-phi state to count as stable.")
    parser.add_argument("--x-param",         default=X_PARAM,
                        help="Name of the x-axis parameter in directory names (e.g. 'p2').")
    parser.add_argument("--y-param",         default=Y_PARAM,
                        help="Name of the y-axis parameter in directory names (e.g. 'swi6').")
    parser.add_argument("--fulla-subdir",    default=FULLA_SUBDIR,
                        help="Sub-folder inside Noise{N}/ for the dilute (A) IC. "
                             "Default: 'FullA'.")
    parser.add_argument("--fullm-subdir",    default=FULLM_SUBDIR,
                        help="Sub-folder inside Noise{N}/ for the condensed (M) IC. "
                             "Default: 'FullM'.")
    parser.add_argument("--max-workers",     type=int,
                        default=max(1, (os.cpu_count() or 6) - 1))
    args = parser.parse_args()

    run(
        dilute_dir    = args.dilute_dir,
        condensed_dir = args.condensed_dir,
        output_dir    = args.output_dir,
        noise_values  = args.noise,
        stride        = args.stride,
        stable_thresh = args.frac_hi_thresh,
        max_workers   = args.max_workers,
        x_param       = args.x_param,
        y_param       = args.y_param,
        fulla_subdir  = args.fulla_subdir,
        fullm_subdir  = args.fullm_subdir,
    )