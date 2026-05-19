"""
src_figS7.py
----------
Supplementary Figure S7: timescale and reaction-rate analysis.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- styling () ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    
    "mathtext.fontset": "stix",  # best match to Times for equations
    
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ---------- data ----------
dt      = 0.001
N       = 80
k2 = 2.5e-4 / dt
k1 = 1e-4 / dt

monomers = np.arange(1, N+1)

flory_A = 0.59
flory_M = 1/3

reaction_rates_zone = np.array([1/k1, 1/k2])

rouse_time = monomers**2 / (3*np.pi**2)
rouse_A_flory = monomers**(1 + 2*flory_A) / (3*np.pi**2)
rouse_M_flory = monomers**(1 + 2*flory_M) / (3*np.pi**2)

# ---------- colorblind-safe palette (Okabe–Ito) ----------
colors = {
    "ideal": "#6A3D9A",   # purple
    "floryA": "#0072B2",  # blue
    "floryM": "#D55E00",  # vermillion
    "zone":  "#E69F00",   # orange
    "h1":    "#009E73",   # green
    "h2":    "#CC79A7"    # pink (adjusted)
}

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(6.5, 4.5))

# main curves
ax.plot(monomers, rouse_time,
        label="Rouse (ideal)",
        color=colors["ideal"], lw=2)

ax.plot(monomers, rouse_A_flory,
        label=r"Self-avoiding ($\nu=0.59$)",
        color=colors["floryA"], lw=2)

ax.plot(monomers, rouse_M_flory,
        label=r"Collapsed ($\nu=1/3$)",
        color=colors["floryM"], lw=2)

# reaction zone
ax.fill_between(monomers,
                reaction_rates_zone[0],
                reaction_rates_zone[1],
                color=colors["zone"],
                alpha=0.25,
                label=r"Reaction rates window ($1/k_1$ to $1/k_2$)")

# reference times
ax.axhline(0.5, color=colors["h1"], ls="--", lw=1.5, alpha=0.7,
           label=r"Noisy conversion time ($1/\Gamma$)")

ax.axhline(0.2, color=colors["h2"], ls="--", lw=1.5, alpha=0.7,
           label=r"Swi6$^*$ lifetime ($\delta(t-0.2))$")

# scales
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(1, N)

# labels
ax.set_xlabel("Number of monomers $N$")
ax.set_ylabel(r"Time ($\tau_{\mathrm{LJ}}$)")

# legend (clean PRX style)
ax.legend(frameon=False, loc="upper left")

# layout
fig.tight_layout()

# save as vector PDF (PRX standard)
plt.savefig("timescale_analysis.pdf", format="pdf", bbox_inches="tight")

plt.show()