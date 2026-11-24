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

