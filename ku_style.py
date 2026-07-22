#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Shared Matplotlib style and helpers for the figures of the Kerr-Dirac
two-channel scattering manuscript.

Importing this module configures a consistent publication style (serif text,
Computer Modern math fonts, a fixed colour palette) and exposes ``savefig`` to
write a vector PDF into ``latex/figures/``.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for convenience)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
NAVY = "#142D6E"
SKY = "#7896D2"
CORAL = "#E74C3C"
TEAL = "#16A085"
GOLD = "#9A7A2E"
GREY = "#555555"

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
_RC = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,   # embed TrueType, no Type 3 (journal requirement)
    "ps.fonttype": 42,
}
mpl.rcParams.update(_RC)

# Directory latex/figures/ relative to this file (codes/ -> latex/figures/).
FIG_DIR = Path(__file__).resolve().parent.parent / "latex" / "figures"


def savefig(fig, name):
    """Save *fig* as a vector PDF ``<name>.pdf`` into ``latex/figures/``."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / (name + ".pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")
    return out


def grid(ax):
    """Apply the house background grid to an axis."""
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)


def legend(ax, **kw):
    """House legend style."""
    opts = dict(frameon=True, fancybox=False, edgecolor="gray", framealpha=0.9)
    opts.update(kw)
    return ax.legend(**opts)
