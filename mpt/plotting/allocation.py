"""
Weight charts: grouped bars across strategies, pie breakdowns,
risk-contribution bars, and weight drift over a backtest.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics import risk_contributions
from .style import PALETTE, asset_color, finish, percent_axis, setup_style


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


def plot_weights_over_time(result,
                           out_path: str = "figures/weights_over_time.png") -> str:
    """Stacked area of weights drifting between rebalances."""
    setup_style()
    w = result.weights_over_time
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.stackplot(w.index, w.T.values, labels=w.columns,
                 colors=[asset_color(i) for i in range(w.shape[1])], alpha=0.9)
    if len(result.rebalance_dates):
        for d in result.rebalance_dates:
            ax.axvline(d, color="white", lw=0.8, alpha=0.7)
    ax.set_xlim(w.index[0], w.index[-1])
    ax.set_ylabel("Weight")
    percent_axis(ax, "y")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    return finish(fig, ax, "Weights Drifting Between Rebalances", out_path)


def plot_concentration_over_time(result,
                                 out_path: str = "figures/concentration_over_time.png") -> str:
    """
    Herfindahl-Hirschman index of the weight vector through the backtest.
    Rises when the optimizer piles into a few names, falls at rebalances.
    """
    setup_style()
    w = result.weights_over_time
    hhi = (w ** 2).sum(axis=1)
    n = w.shape[1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hhi.index, hhi.values, lw=1.8, color=PALETTE["max_sharpe"],
            label="HHI (portfolio)")
    ax.axhline(1.0 / n, ls="--", c=PALETTE["benchmark"], lw=1.4,
               label=f"equal weight (1/{n} = {1 / n:.3f})")
    if len(result.rebalance_dates):
        for d in result.rebalance_dates:
            ax.axvline(d, color="grey", lw=0.7, alpha=0.5)
    ax.set_ylabel("Concentration (HHI)")
    ax.set_ylim(0, max(float(hhi.max()) * 1.25, 2.0 / n))
    ax.legend()
    return finish(fig, ax, "Portfolio Concentration Over Time", out_path)


def plot_weight_trajectories(result, top: int = 6,
                             out_path: str = "figures/weight_trajectories.png") -> str:
    """
    Line chart of the ``top`` largest (by peak weight) holdings over the
    backtest - shows individual drift paths that a stacked area hides.
    """
    setup_style()
    w = result.weights_over_time
    keep = w.max().nlargest(top).index
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for i, t in enumerate(keep):
        ax.plot(w.index, w[t].values, lw=1.6, color=asset_color(i), label=t)
    if len(result.rebalance_dates):
        for d in result.rebalance_dates:
            ax.axvline(d, color="grey", lw=0.7, alpha=0.45)
    ax.set_ylabel("Weight")
    percent_axis(ax, "y")
    ax.legend(ncol=3, fontsize=9)
    title = f"Top {top} Holdings - Weight Trajectories"
    return finish(fig, ax, title, out_path)
