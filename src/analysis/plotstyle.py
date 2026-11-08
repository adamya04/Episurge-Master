#!/usr/bin/env python3
"""
EpiSurge - shared figure style.

Before this module every figure set its own fonts, sizes, DPI and spine
treatment inline, so the seven existing figures do not match each other.
A reader of the report sees that as carelessness before they read a single
axis label.

Every plotting script imports from here and calls nothing of matplotlib's
global state itself.

    from plotstyle import apply, finish, save, PALETTE, MODEL_COLOURS

    apply()
    fig, ax = plt.subplots(figsize=(8, 5))
    ...
    finish(ax)
    save(fig, "fig_whatever")

Colour policy
-------------
PALETTE is the Okabe-Ito colourblind-safe sequence. It is used for any
series whose identity is arbitrary (arms, folds, relations).

MODEL_COLOURS pins B0/B1/B2 to the exact hexes the E2 figures already use.
Those figures are in RESULTS.md and are referred to in writing; silently
recolouring them would make older text wrong about what the reader sees.
New series take PALETTE.

DPI
---
300, not the 200 the earlier figures used. Journals ask for 300 minimum on
raster figures. Regenerating the existing figures at 300 is the intended
side effect of adopting this module, not an accident.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

FIGURES = Path("figures")

DPI = 300

# Okabe-Ito, reordered so the first three are distinguishable in greyscale.
PALETTE = [
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#CC79A7",   # reddish purple
    "#E69F00",   # orange
    "#56B4E9",   # sky blue
    "#F0E442",   # yellow
    "#000000",   # black
]

# Pinned: these three appear in figures already cited in RESULTS.md.
MODEL_COLOURS = {"B0": "#888888", "B1": "#e8a33d", "B2": "#1f4e79"}

GREY = "#888888"
RULE = "#666666"


def apply():
    """Set global rcParams. Call once, before creating any figure."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,

        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelpad": 4,
        "axes.prop_cycle": plt.cycler(color=PALETTE),

        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,

        "legend.frameon": False,
        "figure.autolayout": False,
    })


def finish(ax, ygrid=True, xgrid=False):
    """
    The spine and grid treatment every EpiSurge figure uses: no top or
    right spine, a faint grid behind the data.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    if xgrid:
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def zero_line(ax, label=None, x=None):
    """A dashed reference line at y=0, for skill-difference panels."""
    ax.axhline(0, color=GREY, linestyle="--", linewidth=1.3, zorder=1)
    if label:
        ax.annotate(label, xy=(x if x is not None else 0, 0),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=RULE)
    return ax


def save(fig, name, close=True):
    """Write figures/<name>.png and return the path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(out)
    if close:
        plt.close(fig)
    return out