#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Figure 6(b,c) of the manuscript: finite-matching convergence diagnostics.

Panel (b): unitarity defect ||S* S - I||_2 versus the endpoint u_+ for the
Schwarzschild and Kerr benchmarks, with the u_+^{-2} reference slope, and
the (gauge-sensitive) transmission-phase drift showing the u_+^{-1} log u_+
class — the two-tier convergence of Section XI.

Panel (c): structure-preserving diagnostics at the tolerance floor — the
determinant-Wronskian relative variation and the matching-point invariance
of S — plotted on their own scale so they are not compressed against the
much larger finite-endpoint error.

Requires results/convergence_study.json and the benchmark JSONs
(run `make benchmarks` first). Writes ../latex/figures/fig06b-convergence.pdf.
"""

import json
from pathlib import Path

import numpy as np

import ku_style as ks
from ku_style import plt

# Print-size figure: designed at the manuscript line width (~6.3 in), so the
# fonts below are the final printed sizes.
ks.mpl.rcParams.update({
    "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})

RES = Path("results")


def main() -> None:
    conv = json.loads((RES / "convergence_study.json").read_text())
    schw = json.loads((RES / "schwarzschild_dirac_matching_output.json").read_text())
    kerr = json.loads((RES / "kerr_dirac_matching_output.json").read_text())

    fig, (axb, axc) = plt.subplots(1, 2, figsize=(6.4, 2.85))

    # ---- panel (b): two-tier convergence --------------------------------
    for geom, color, marker, label in (
            ("schwarzschild", ks.NAVY, "o", "Schwarzschild"),
            ("kerr", ks.CORAL, "s", "Kerr $a=0.4$")):
        rows = conv[geom]["first_order"]["rows"]
        u = np.array([r["u_right"] for r in rows])
        d = np.array([r["unitary_defect"] for r in rows])
        p = np.array([r["phase_drift_S12"] for r in rows])
        axb.loglog(u, d, marker=marker, ms=3.6, color=color, lw=1.2,
                   label=rf"defect, {label}")
        axb.loglog(u, p, marker=marker, ms=3.6, color=color, lw=1.0,
                   ls="--", alpha=0.65,
                   label=rf"phase drift, {label}")
    uref = np.array([55.0, 420.0])
    axb.loglog(uref, 0.9 * (uref / 60.0) ** (-2.0) * 3.5e-4, color=ks.GREY,
               lw=1.0, ls=":", label=r"$\propto u_+^{-2}$")
    axb.loglog(uref, (np.log(uref) / uref) / (np.log(60.0) / 60.0) * 8e-3,
               color=ks.GREY, lw=1.0, ls="-.",
               label=r"$\propto u_+^{-1}\log u_+$")
    axb.set_xticks([60, 120, 240, 360])
    axb.get_xaxis().set_major_formatter(ks.mpl.ticker.ScalarFormatter())
    axb.get_xaxis().set_minor_formatter(ks.mpl.ticker.NullFormatter())
    axb.set_xlabel(r"$u_+$")
    axb.set_ylabel(r"diagnostic")
    axb.set_title(r"(b) invariant vs gauge-sensitive convergence",
                  fontsize=10)
    ks.grid(axb)
    ks.legend(axb, loc="lower left", ncol=1, fontsize=6.4)

    # ---- panel (c): tolerance-floor diagnostics -------------------------
    for data, color, marker, label in (
            (schw, ks.NAVY, "o", "Schwarzschild"),
            (kerr, ks.CORAL, "s", "Kerr $a=0.4$")):
        u = np.array([r["u_right"] for r in data["runs"]])
        w = np.array([r["wronskian_out_relative_variation"]
                      for r in data["runs"]])
        m = np.array([r["matching_point_variation_frobenius"]
                      for r in data["runs"]])
        axc.semilogy(u, w, marker=marker, ms=4, color=color, lw=1.2,
                     label=rf"$\delta W$, {label}")
        axc.semilogy(u, m, marker=marker, ms=4, color=color, lw=1.0, ls="--",
                     alpha=0.65, label=rf"$\delta_{{u_*}}S$, {label}")
    axc.axhspan(1e-11, 1e-9, color=ks.SKY, alpha=0.18, lw=0,
                label="ODE tolerance band")
    axc.set_xlabel(r"$u_+$")
    axc.set_ylabel(r"structure defects")
    axc.set_ylim(3e-12, 3e-7)
    axc.set_title(r"(c) Wronskian and matching invariance", fontsize=10)
    ks.grid(axc)
    ks.legend(axc, loc="upper right", fontsize=6.4)

    fig.tight_layout()
    ks.savefig(fig, "fig06b-convergence")


if __name__ == "__main__":
    main()
