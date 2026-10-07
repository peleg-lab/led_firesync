"""Shared plotting conventions for the B* analyses.

Applies the manuscript's figure style: no top/right spines, no axis titles, no
suptitles, light tick padding.  Use `clean(ax)` after constructing each Axes
and avoid calling `set_title` / `suptitle` in the scripts.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_rc() -> None:
    mpl.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 0,
        "axes.titleweight": "normal",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.titlesize": 0,
        "savefig.bbox": "tight",
    })


def clean(ax: plt.Axes) -> plt.Axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("")
    return ax


def clean_all(axes) -> None:
    try:
        iter(axes)
    except TypeError:
        clean(axes); return
    for a in axes.flat if hasattr(axes, "flat") else axes:
        clean(a)
