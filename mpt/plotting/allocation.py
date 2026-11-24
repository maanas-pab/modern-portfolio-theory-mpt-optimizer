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


def plot_weights_pies(tables: dict,
                      out_path: str = "figures/weights_pies.png") -> str:
    """Side-by-side pies (top-6 + 'other') for up to three strategies."""
    setup_style()
    names = list(tables)[:3]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.6))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        df = tables[name].sort_values("weight", ascending=False)
        top = df.head(6).copy()
        rest = df["weight"].iloc[6:].sum()
        if rest > 1e-6:
            top = pd.concat([top, pd.DataFrame(
                {"ticker": ["other"], "weight": [rest],
                 "weight_pct": [rest * 100], "portfolio": [name]})], ignore_index=True)
        wedges, _, autotexts = ax.pie(
            top["weight"], labels=top["ticker"], autopct="%1.1f%%",
            colors=[asset_color(i) for i in range(len(top))],
            startangle=90, pctdistance=0.75,
            wedgeprops={"linewidth": 1, "edgecolor": "white"})
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title(name)
    return finish(fig, axes[0], "Weight Breakdown by Strategy", out_path)


def plot_risk_contribution(weights, cov_matrix, tickers,
                           name: str = "Max Sharpe",
                           out_path: str = "figures/risk_contribution.png") -> str:
    """Per-asset percent contribution to total portfolio risk."""
    setup_style()
    _, pct = risk_contributions(weights, cov_matrix)
    order = np.argsort(pct)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [asset_color(i) for i in range(len(order))]
    ax.bar([tickers[i] for i in order], pct[order], color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("% of total risk")
    percent_axis(ax, "y")
    ax.axhline(1.0 / len(pct), ls="--", c=PALETTE["negative"], lw=1.2,
               label=f"equal risk ({1.0 / len(pct):.1%})")
    ax.legend()
    return finish(fig, ax, f"Risk Contributions - {name}", out_path)


