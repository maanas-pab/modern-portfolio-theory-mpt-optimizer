"""
Shared matplotlib style: rcParams, palette, and figure helpers so
every chart in the gallery looks like one family.
"""
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PALETTE = {
    "frontier": "#e63946",      # red - the frontier line
    "max_sharpe": "#d62828",    # star marker
    "min_vol": "#f4a261",       # diamond marker
    "benchmark": "#457b9d",     # SPY / market
    "positive": "#2a9d8f",
    "negative": "#e76f51",
    "random": "#8d99ae",
    "ink": "#1d3557",
}

ASSET_COLORS = [
    "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
    "#6a4c93", "#1982c4", "#ff595e", "#8ac926", "#ffca3a",
    "#8338ec", "#3a86ff", "#fb5607", "#06d6a0",
]


def setup_style() -> None:
    """Apply the project's matplotlib defaults (grid, fonts, DPI)."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


