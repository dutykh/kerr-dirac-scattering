#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Figure 7 of the manuscript: the certified quasi-bound fingerprint.

Panel (a): the two certified quasi-bound poles in the complex omega plane
near the threshold m_e, against the hydrogenic estimates.
Panel (b): the real-axis scan of the relative Evans residual |D_dec|,
whose dips locate the levels.
Panel (c): the parameter-free Breit-Wigner overlay: measured phase
derivative of S_HH versus the signed Lorentzian predicted by the pole,
plus the affine background estimated on the wings.

Refuses to produce the figure unless results/quasibound_search.json
carries gate.certified = true (the honest-gating policy of the paper).
Requires `python3 quasibound_search.py` to have been run first.
"""

import json
import sys
from pathlib import Path

import numpy as np

import ku_style as ks
from ku_style import plt

ks.mpl.rcParams.update({
    "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})

RES = Path("results")


def main() -> None:
    data = json.loads((RES / "quasibound_search.json").read_text())
    if not data["gate"]["certified"]:
        sys.exit("gate not certified: no figure is produced (policy).")

    mass = data["parameters"]["mass"]
    E1, g1 = data["root"]["E_n"], data["root"]["gamma_n"]
    E2, g2 = data["neighbour_root"]["E"], data["neighbour_root"]["gamma"]
    bw = data["breit_wigner"]

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(6.6, 2.35))

    # ---- panel (a): poles near the threshold ----------------------------
    axa.axhline(0.0, color=ks.GREY, lw=0.7)
    axa.axvline(mass, color="#9C5310", lw=1.0, ls="--")
    axa.text(mass - 2.0e-4, 0.82, r"$m_e$", color="#9C5310", fontsize=8,
             ha="right")
    for n, (E, g) in enumerate(((E1, g1), (E2, g2)), start=1):
        axa.plot(E, g * 1e3, "x", ms=7, mew=1.8, color=ks.NAVY)
        axa.annotate(rf"$\omega_{n}$", (E, g * 1e3),
                     textcoords="offset points", xytext=(-2, 6),
                     fontsize=8, color=ks.NAVY, ha="right")
    for n in (1, 2):
        axa.plot(mass * (1 - (mass ** 2) / (2 * n * n)), 0.0, "o", ms=5,
                 mfc="none", mec=ks.GREY)
    axa.set_xlim(0.1915, 0.2007)
    axa.set_ylim(-0.12, 0.95)
    axa.set_xticks([0.192, 0.196, 0.200])
    axa.set_xlabel(r"$\mathrm{Re}\,\omega$")
    axa.set_ylabel(r"$\mathrm{Im}\,\omega\ \times10^{3}$")
    axa.set_title("(a) certified poles", fontsize=10)
    ks.grid(axa)

    # ---- panel (b): real-axis Evans scan --------------------------------
    om = np.array(data["scan"]["omega"])
    dd = np.array(data["scan"]["relative_residual"])
    axb.semilogy(om, dd, color=ks.NAVY, lw=1.2)
    for E in (E1, E2):
        axb.axvline(E, color=ks.CORAL, lw=0.8, ls=":")
    axb.set_xlabel(r"$\omega$")
    axb.set_ylabel(r"$|D^{\rm dec}|_{\rm rel}$")
    axb.set_title("(b) Evans residual scan", fontsize=10)
    ks.grid(axb)

    # ---- panel (c): parameter-free Breit-Wigner overlay -----------------
    oc = np.array(bw["omega_core"])
    dc = np.array(bw["dphi_domega_core"])
    ow = np.array(bw["omega_wing"])
    dw = np.array(bw["dphi_domega_wing"])
    a1, a0 = bw["background_affine"]
    xs = np.linspace(ow.min(), ow.max(), 400)
    pred = -2.0 * g1 / ((xs - E1) ** 2 + g1 * g1) + a1 * (xs - E1) + a0
    x0 = (oc - E1) / abs(g1)
    xw = (ow - E1) / abs(g1)
    xp = (xs - E1) / abs(g1)
    axc.plot(xp, pred * 1e-3, color=ks.CORAL, lw=1.4,
             label="pole prediction")
    axc.plot(x0, dc * 1e-3, "o", ms=3.4, color=ks.NAVY, label="measured core")
    axc.plot(xw, dw * 1e-3, "s", ms=3.0, mfc="none", mec=ks.NAVY,
             label="measured wings")
    axc.set_xlabel(r"$(\omega-E_1)/|\gamma_1|$")
    axc.set_ylabel(r"$d\varphi/d\omega\ \times10^{-3}$")
    axc.set_title("(c) Breit–Wigner overlay", fontsize=10)
    ks.grid(axc)
    ks.legend(axc, loc="lower right", fontsize=6.4)

    fig.tight_layout()
    ks.savefig(fig, "fig07-quasibound")


if __name__ == "__main__":
    main()
