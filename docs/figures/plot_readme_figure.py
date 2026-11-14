#!/usr/bin/env python3
# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Figure of the README: the two quasi-bound levels of the Schwarzschild benchmark.

The data are those of results/quasibound_search.json, written by quasibound_search.py; the panels
are those of the manuscript's figure drawn by make_fig07.py. From this folder,
    python3 plot_readme_figure.py
writes kerr-dirac-quasibound.png. The style is the stamped copy manuscript_style.py beside it.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "results" / "quasibound_search.json"
sys.path.insert(0, str(HERE))
import manuscript_style as hs  # noqa: E402

data = json.loads(DATA.read_text())
mass = data["parameters"]["mass"]
E1, g1 = data["root"]["E_n"], data["root"]["gamma_n"]
E2, g2 = data["neighbour_root"]["E"], data["neighbour_root"]["gamma"]
bw = data["breit_wigner"]

hs.configure(family="latin-modern")
plt.rcParams["savefig.dpi"] = 265
plt.rcParams["figure.facecolor"] = hs.PAPER
plt.rcParams["savefig.facecolor"] = hs.PAPER

fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(hs.WIDTH, 2.45), constrained_layout=True)
dotted = (0, (1.6, 1.6))

# (a) the two poles near the threshold, with the hydrogenic estimates on the real axis
axa.axhline(0.0, color=hs.MIST, lw=0.6)
axa.axvline(mass, color=hs.SLATE, lw=0.8, linestyle=hs.DASHES[1])
for E, g in ((E1, g1), (E2, g2)):
    axa.plot(E, g * 1e3, "x", ms=6, mew=1.5, color=hs.NAVY)
axa.annotate(r"$\omega_1$", (E1, g1 * 1e3), textcoords="offset points", xytext=(6, -3),
             fontsize=7.5, color=hs.NAVY, ha="left", va="center")
axa.annotate(r"$\omega_2$", (E2, g2 * 1e3), textcoords="offset points", xytext=(-6, 6),
             fontsize=7.5, color=hs.NAVY, ha="right", va="bottom")
for n in (1, 2):
    axa.plot(mass * (1 - mass ** 2 / (2 * n * n)), 0.0, "o", ms=4.0, markerfacecolor="white",
             markeredgecolor=hs.SLATE)
axa.set_xlim(0.1915, 0.2007)
axa.set_ylim(-0.12, 0.95)
axa.set_xticks([0.192, 0.196, 0.200])
axa.set_xlabel(r"$\mathrm{Re}\,\omega$")
axa.set_ylabel(r"$10^{3}\,\mathrm{Im}\,\omega$")
hs.panel(axa, "(a) the two poles")

# (b) the relative Evans residual along the real axis
om = np.array(data["scan"]["omega"])
rr = np.array(data["scan"]["relative_residual"])
for E in (E1, E2):
    axb.axvline(E, color=hs.SLATE, lw=0.8, linestyle=dotted)
axb.semilogy(om, rr, color=hs.NAVY, lw=1.1)
axb.set_xlim(0.187, 0.2)
axb.set_xlabel(r"$\omega$")
axb.set_ylabel(r"$|D^{\mathrm{dec}}|_{\mathrm{rel}}$")
hs.panel(axb, "(b) residual on the real axis")

# (c) the measured phase derivative against the Lorentzian predicted by the pole omega_1
oc, dc = np.array(bw["omega_core"]), np.array(bw["dphi_domega_core"])
ow, dw = np.array(bw["omega_wing"]), np.array(bw["dphi_domega_wing"])
a1, a0 = bw["background_affine"]
xs = np.linspace(ow.min(), ow.max(), 400)
pred = -2.0 * g1 / ((xs - E1) ** 2 + g1 * g1) + a1 * (xs - E1) + a0
axc.plot((xs - E1) / abs(g1), pred * 1e-3, color=hs.WINE, lw=1.1)
axc.plot((oc - E1) / abs(g1), dc * 1e-3, "o", ms=3.4, markerfacecolor="white",
         markeredgecolor=hs.NAVY, markeredgewidth=0.9, linestyle="none")
axc.plot((ow - E1) / abs(g1), dw * 1e-3, "s", ms=3.0, markerfacecolor="white",
         markeredgecolor=hs.TEAL, markeredgewidth=0.9, linestyle="none")
axc.set_xlabel(r"$(\omega - E_1)/|\gamma_1|$")
axc.set_ylabel(r"$10^{-3}\,d\varphi/d\omega$")
hs.panel(axc, "(c) Breit--Wigner shape")

handles = [
    Line2D([], [], color=hs.NAVY, marker="x", ms=5, mew=1.4, linestyle="none"),
    Line2D([], [], color=hs.SLATE, marker="o", ms=4, markerfacecolor="white", linestyle="none"),
    Line2D([], [], color=hs.SLATE, lw=0.8, linestyle=hs.DASHES[1]),
    Line2D([], [], color=hs.SLATE, lw=0.8, linestyle=dotted),
    Line2D([], [], color=hs.WINE, lw=1.1),
    Line2D([], [], color=hs.NAVY, marker="o", ms=3.4, markerfacecolor="white", linestyle="none"),
    Line2D([], [], color=hs.TEAL, marker="s", ms=3.0, markerfacecolor="white", linestyle="none"),
]
labels = ["poles", "hydrogenic levels", r"threshold $m_e$", r"$\mathrm{Re}\,\omega_1$, $\mathrm{Re}\,\omega_2$",
          "Lorentzian of the pole", "phase derivative, core", "phase derivative, wings"]
hs.legend_outside(fig, where="lower", ncol=4, handles=handles, labels=labels)
hs.save(fig, HERE / "kerr-dirac-quasibound.png")
print("written", HERE / "kerr-dirac-quasibound.png")
