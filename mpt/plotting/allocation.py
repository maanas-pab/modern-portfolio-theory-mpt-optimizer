"""
Weight charts: grouped bars across strategies, pie breakdowns,
risk-contribution bars, and weight drift over a backtest.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics import risk_contributions
from .style import ASSET_COLORS, PALETTE, asset_color, finish, percent_axis, setup_style

def plot_weights_comparison(tables: dict,
                            out_path: str = "figures/weights_comparison.png") -> str:
    """
    Grouped bars: one cluster per ticker, one bar per named strategy.
    tables maps strategy name -> weights_table DataFrame (ticker/weight).
    """
    setup_style()
    tickers = sorted({t for df in tables.values() for t in df["ticker"]})
    x = np.arange(len(tickers))
    width = 0.8 / len(tables)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [PALETTE["max_sharpe"], PALETTE["min_vol"], PALETTE["benchmark"], "#6a4c93"]
    for k, (name, df) in enumerate(tables.items()):
        lookup = dict(zip(df["ticker"], df["weight"]))
        vals = [lookup.get(t, 0.0) for t in tickers]
        ax.bar(x + k * width, vals, width, label=name,
               color=colors[k % len(colors)], alpha=0.9)
    ax.set_xticks(x + width * (len(tables) - 1) / 2)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Weight")
    percent_axis(ax, "y")
    ax.legend()
    return finish(fig, ax, "Portfolio Allocations Compared", out_path)


