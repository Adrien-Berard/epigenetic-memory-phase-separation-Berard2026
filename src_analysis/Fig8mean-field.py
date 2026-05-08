"""
Nucleosome Gillespie Dynamics — PRX-style figure generator
Runs a Gillespie simulation and produces three publication-ready panels:
  1. m(t) and s(t)
  2. Vol(t)
  3. Phase diagram with vector field and trajectory
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# PRX rcParams
# ---------------------------------------------------------------------------
PRX_RC = {
    "font.family":        "serif",
    "font.size":          8.5,
    "axes.labelsize":     8,
    "axes.titlesize":     8,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
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
plt.rcParams.update(PRX_RC)

# ---------------------------------------------------------------------------
# Model parameters (match HTML defaults)
# ---------------------------------------------------------------------------
L     = 50
C     = 6.0
rho   = 20.0
beta  = 4.0
gamma = 1.0
k1    = 1.0
k2    = 1.0
Gamma = 0.065      # spontaneous conversion (Γ)
delta = 1 / L
m0, s0 = 0.20, 0.20
t_max = 150_000
div_interval = 10_000   # set to 0 to disable cell division
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _label_panel(ax, label, x=-0.12, y=1.01): #x=-0.12 before
    """PRX-style panel label slightly outside top-left of axes."""
    label = label
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        clip_on=False,
        zorder=10,fontsize=PRX_RC['font.size']
    )
    
def volume(m, s):
    raw = L * (C - 3 * m) / max(1 + s, 1e-9)
    return max(raw, 1e-9)


def model_rates(m, s):
    """Return all Gillespie process rates."""
    m = np.clip(m, 0.0, 1.0)
    s = np.clip(s, 0.0, 1.0)
    Vol  = volume(m, s)
    x    = m * L / Vol
    nden = L / Vol

    mp_recruit = max(0.0, k2**2 * s * m * (1 - m))
    mm_recruit = max(0.0, k1 * k2 * m * (1 - m)**2)
    mp_noise   = max(0.0, Gamma * (1 - m))
    mm_noise   = max(0.0, Gamma * m)

    s_capture_raw = (rho / Vol**(1/3)) * x * (1 - nden - s)
    s_escape      = max(0.0, s * np.exp(-beta * (s + x)))
    s_plus        = max(0.0, gamma * s_capture_raw)
    s_minus       = max(0.0, gamma * s_escape) + max(0.0, -gamma * s_capture_raw)

    # [name, rate, variable, step]
    return [
        ("m+ recruit", mp_recruit, "m", +delta),
        ("m- recruit", mm_recruit, "m", -delta),
        ("m+ Γ",       mp_noise,   "m", +delta),
        ("m- Γ",       mm_noise,   "m", -delta),
        ("s+ bind",    s_plus,     "s", +delta),
        ("s- loss",    s_minus,    "s", -delta),
    ]


def drift_at(m, s):
    procs = model_rates(m, s)
    dm = (procs[0][1] + procs[2][1]) - (procs[1][1] + procs[3][1])
    ds = procs[4][1] - procs[5][1]
    return dm, ds


def gaussian_draw_m(old_m):
    """Gaussian inheritance model at cell division."""
    mu    = np.clip(0.25 + old_m / 2, 0, 1)
    sigma = np.sqrt(max(0, mu * (1 - mu) / L))
    raw   = rng.normal(mu, sigma)
    clipped = np.clip(raw, 0, 1)
    lattice = np.clip(round(clipped * L) / L, 0, 1)
    return lattice


# ---------------------------------------------------------------------------
# Gillespie simulation
# ---------------------------------------------------------------------------

def gillespie_sim():
    t   = 0.0
    m   = np.clip(m0, 0.0, 1.0)
    s   = np.clip(s0, 0.0, 1.0)
    ts, ms, ss, vols, divs = [t], [m], [s], [volume(m, s)], [False]

    next_div = div_interval if div_interval > 0 else np.inf

    while t < t_max:
        # --- cell division ---
        if div_interval > 0 and next_div <= t_max:
            procs = model_rates(m, s)
            total = sum(p[1] for p in procs)
            dt_stoch = (-np.log(max(rng.random(), 1e-12)) / total
                        if total > 0 else np.inf)
            next_event_t = t + dt_stoch

            if next_div <= next_event_t:
                t  = next_div
                m  = gaussian_draw_m(m)
                s  = 0.0
                ts.append(t); ms.append(m); ss.append(s)
                vols.append(volume(m, s)); divs.append(True)
                next_div += div_interval
                continue

        procs = model_rates(m, s)
        total = sum(p[1] for p in procs)

        if total <= 0 or not np.isfinite(total):
            break

        dt  = -np.log(max(rng.random(), 1e-12)) / total
        t  += dt

        u   = rng.random() * total
        cum = 0.0
        chosen = procs[-1]
        for p in procs:
            cum += p[1]
            if u <= cum:
                chosen = p
                break

        if chosen[2] == "m":
            m = np.clip(m + chosen[3], 0, 1)
        else:
            s = np.clip(s + chosen[3], 0, 1)

        ts.append(t); ms.append(m); ss.append(s)
        vols.append(volume(m, s)); divs.append(False)

    return (np.array(ts), np.array(ms), np.array(ss),
            np.array(vols), np.array(divs, dtype=bool))


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
print("Running Gillespie simulation…")
T, M, S, V, DIV = gillespie_sim()
print(f"  Done. {len(T):,} stored events, final t = {T[-1]:.0f}")

div_times = T[DIV]

# ---------------------------------------------------------------------------
# Colour palette (PRX-friendly: no vivid primaries — use muted hues)
# ---------------------------------------------------------------------------
C_M   = "#D6001C"   #  red for m
C_S   = "#1A9641"    #  green for s
C_VOL = "#ec9005"   # orange for Vol
C_DIV = "#888888"   # grey dashes for divisions


N_FIELD = 60   # resolution of sign-map background
N_ARROW = 14   # number of arrows per axis



# ---- sign-map quadrant background ----
mg = np.linspace(0, 1, N_FIELD)
sg = np.linspace(0, 1, N_FIELD)
MG, SG = np.meshgrid(mg, sg)
DM = np.zeros_like(MG)
DS = np.zeros_like(SG)
for i in range(N_FIELD):
    for j in range(N_FIELD):
        dm, ds = drift_at(MG[i, j], SG[i, j])
        DM[i, j] = dm
        DS[i, j] = ds

# Colour: Q1 dm>0 ds>0 (red), Q2 dm>0 ds<0 (orange), Q3 dm<0 ds>0 (cyan), Q4 dm<0 ds<0 (blue)
quad = np.zeros((N_FIELD, N_FIELD))
quad[(DM >= 0) & (DS >= 0)] = 0
quad[(DM >= 0) & (DS <  0)] = 1
quad[(DM <  0) & (DS >= 0)] = 2
quad[(DM <  0) & (DS <  0)] = 3

cmap_quad = ListedColormap([
    "#f0b0b0",   # Q1: red-tinted
    "#f5d8a0",   # Q2: amber-tinted
    "#a8e0d8",   # Q3: cyan-tinted
    "#aabde0",   # Q4: blue-tinted
])


# ---- vector field arrows ----
ma = np.linspace(0.04, 0.96, N_ARROW)
sa = np.linspace(0.04, 0.96, N_ARROW)
MA, SA = np.meshgrid(ma, sa)
DMA = np.zeros_like(MA)
DSA = np.zeros_like(SA)
for i in range(N_ARROW):
    for j in range(N_ARROW):
        dm, ds = drift_at(MA[i, j], SA[i, j])
        DMA[i, j] = dm
        DSA[i, j] = ds

mag = np.hypot(DMA, DSA)
mag[mag == 0] = 1
scale = 0.065 / mag
UU = DMA * scale * (0.34 + 0.92 * mag / mag.max())
VV = DSA * scale * (0.34 + 0.92 * mag / mag.max())

# ---- trajectory ----
# Thin the stored trace for the phase plot
stride = max(1, len(T) // 3000)

# Division events
if div_times.size > 0:
    div_idx = np.where(DIV)[0]

# Fixed-point marker (simple numerical search)
best_score = np.inf
fp_m, fp_s = 0.5, 0.5
for fm in np.linspace(0.05, 0.95, 28):
    for fs in np.linspace(0.05, 0.95, 28):
        dm, ds = drift_at(fm, fs)
        score  = dm**2 + ds**2
        if score < best_score:
            best_score = score
            fp_m, fp_s = fm, fs

# Parameter annotation
param_text = (
    rf"$L={L},\ C={C},\ \rho={rho:.0f}$" "\n"
    rf"$\beta={beta:.0f},\ \gamma={gamma:.0f},\ \Gamma={Gamma}$" "\n"
    rf"$k_1={k1:.1f},\ k_2={k2:.1f},\ \delta={delta:.3f}$"
)


# ---------------------------------------------------------------------------
# Combined 3-panel figure (optional — useful for papers)
# ---------------------------------------------------------------------------
fig4 = plt.figure(figsize=(3.5, 6))
gs   = fig4.add_gridspec(4, 1, hspace=0.35,
                          left=0.08, right=0.97, top=0.96, bottom=0.08)
ax_ms   = fig4.add_subplot(gs[0, 0])
ax_vol  = fig4.add_subplot(gs[1, 0])
ax_ph   = fig4.add_subplot(gs[2:, 0])

# — m, s —
ax_ms.plot(T, M, color=C_M, lw=1.2, label=r"$m$")
ax_ms.plot(T, S, color=C_S, lw=1.2, label=r"$s$", alpha=0.88)
for td in div_times: ax_ms.axvline(td, color=C_DIV, lw=0.4, ls="--", alpha=0.35)
ax_ms.set_xlim(0, T[-1]); ax_ms.set_ylim(0, 1)
ax_ms.set_xlabel("Time"); ax_ms.set_ylabel(r"$m,\,s$" +'\nfractions')
ax_ms.set_yticks([0,0.5,1])
ax_ms.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax_ms.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax_ms.set_xticks([0,75000,150000])
ax_ms.set_xticklabels([0,7.5,rf'$15 \times 10^4$'])
ax_ms.legend(handlelength=1.4, loc="upper left")
_label_panel(ax_ms,'(a)')

# — Vol —
ax_vol.plot(T, V, color=C_VOL, lw=1.2, label=r"$\mathrm{Vol}$")
for td in div_times: ax_vol.axvline(td, color=C_DIV, lw=0.4, ls="--", alpha=0.35)
ax_vol.set_xlim(0, T[-1]); ax_vol.set_ylim(90,310)
ax_vol.set_xlabel("Time"); ax_vol.set_ylabel(r"$\mathrm{Vol}$")
ax_vol.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax_vol.set_xticks([0,75000,150000])
ax_vol.set_xticklabels([0,7.5,rf'$15 \times 10^4$'])
ax_vol.set_yticks([100,200,300])
ax_vol.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax_vol.legend(handlelength=1.4, loc="upper right")
_label_panel(ax_vol,'(b)')
# — Phase —
ax_ph.pcolormesh(mg, sg, quad, cmap=cmap_quad, vmin=-0.5, vmax=3.5,
                 shading="auto", rasterized=True, alpha=0.7)
ax_ph.quiver(MA, SA, UU, VV,
             angles="xy", scale_units="xy", scale=1,
             width=0.003, headwidth=4, headlength=5, headaxislength=3.5,
             color="0.30", alpha=0.7, rasterized=True)
ax_ph.plot(M[::stride], S[::stride], color="0.15", lw=0.55, alpha=0.2, zorder=3)
if div_times.size > 0:
    ax_ph.scatter(M[div_idx], S[div_idx], s=5, color=C_DIV, zorder=4)
# ax_ph.scatter([M[-1]], [S[-1]], s=20, color=C_M, zorder=5)
ax_ph.plot(fp_m, fp_s, "x", color="#7c3aed", ms=4, mew=1.0, zorder=5)
ax_ph.set_xlim(0, 1); ax_ph.set_ylim(0, 1)
ax_ph.set_xlabel(r"$m$"); ax_ph.set_ylabel(r"$s$")
ax_ph.set_yticks([0,0.5,1])
ax_ph.set_xticks([0,0.5,1])
ax_ph.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax_ph.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax_ph.text(0.4, 0.8, param_text, transform=ax_ph.transAxes,
           ha="right", va="bottom", fontsize=5.8,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9))
_label_panel(ax_ph,'(c)')

fig4.savefig("outputs/prx_meanfield.pdf")
fig4.savefig("outputs/prx_meanfield.png")
print("  Saved prx_meanfield.pdf / .png")


print("All figures saved.")