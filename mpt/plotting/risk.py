"""
Loss distributions and cross-strategy metric scoreboards:
VaR/CVaR histogram, Sharpe cloud histogram, metric bars.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics import conditional_var, value_at_risk
from .style import PALETTE, asset_color, finish, percent_axis, setup_style

def plot_var_cvar(returns: pd.Series, level: float = 0.95,
                  out_path: str = "figures/var_cvar.png") -> str:
    """Daily return histogram with VaR and CVaR threshold lines."""
    setup_style()
    r = returns.dropna()
    var = value_at_risk(r, level)
    cvar = conditional_var(r, level)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bins = np.histogram_bin_edges(r, bins=45)
    ax.hist(r, bins=bins, color=PALETTE["random"], alpha=0.8, edgecolor="white")
    ax.axvline(-var, color=PALETTE["negative"], lw=2,
               label=f"{level:.0%} VaR ({var:.2%})")
    ax.axvline(-cvar, color=PALETTE["max_sharpe"], lw=2, ls="--",
               label=f"{level:.0%} CVaR ({cvar:.2%})")
    ax.axvline(r.mean(), color=PALETTE["positive"], lw=1.6, ls=":",
               label=f"mean ({r.mean():.2%})")
    percent_axis(ax, "x")
    ax.set_xlabel("Daily return")
    ax.legend()
    return finish(fig, ax, "Loss Tail: VaR & Expected Shortfall", out_path)


def plot_sharpe_histogram(rand_s, max_sharpe: float,
                          out_path: str = "figures/sharpe_distribution.png") -> str:
    """Distribution of Sharpe ratios from random portfolios vs the optimum."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(rand_s, bins=50, color=PALETTE["random"], alpha=0.8,
            edgecolor="white", label="Random portfolios")
    ax.axvline(max_sharpe, color=PALETTE["max_sharpe"], lw=2.4,
               label=f"Max Sharpe = {max_sharpe:.2f}")
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("Count")
    ax.legend()
    return finish(fig, ax, "Sharpe Ratio of 5,000 Random Portfolios", out_path)


def plot_metric_comparison(summary_table: pd.DataFrame,
                           out_path: str = "figures/metric_comparison.png") -> str:
    """Grouped bars of return / vol / drawdown across strategies.

    summary_table: rows = metrics (annual_return, annual_vol, max_drawdown),
    columns = strategy names (already decimal-scaled).
    """
    setup_style()
    wanted = [m for m in ("annual_return", "annual_vol", "max_drawdown")
              if m in summary_table.index]
    labels = {"annual_return": "Return", "annual_vol": "Volatility",
              "max_drawdown": "Max drawdown"}
    x = np.arange(len(wanted))
    width = 0.8 / max(len(summary_table.columns), 1)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = [PALETTE["max_sharpe"], PALETTE["min_vol"], PALETTE["benchmark"], "#6a4c93"]
    for k, name in enumerate(summary_table.columns):
        vals = [float(summary_table.loc[m, name]) for m in wanted]
        ax.bar(x + k * width, vals, width, label=name, color=colors[k % len(colors)])
    ax.set_xticks(x + width * (len(summary_table.columns) - 1) / 2)
    ax.set_xticklabels([labels[m] for m in wanted])
    percent_axis(ax, "y")
    ax.legend()
    return finish(fig, ax, "Risk / Return Scoreboard", out_path)


def plot_asset_risk_return(exp_returns, vol,
                           out_path: str = "figures/asset_risk_return.png") -> str:
    """Every asset in risk-return space, labeled with its ticker."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(vol, exp_returns, s=110, c=PALETTE["benchmark"],
               edgecolors="black", zorder=4)
    for i, t in enumerate(vol.index):
        ax.annotate(t, (vol[t], exp_returns[t]), textcoords="offset points",
                    xytext=(6, 4), fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    percent_axis(ax, "x")
    percent_axis(ax, "y")
    title = "Individual Assets: Risk vs Expected Return"
    return finish(fig, ax, title, out_path)


def plot_rolling_beta(returns, benchmark, window: int = 63,
                      out_path: str = "figures/rolling_beta.png") -> str:
    """
    Rolling beta of a portfolio (or asset) against the benchmark.
    Above 1 means the series amplifies benchmark moves over the window.
    """
    setup_style()
    r, b = returns.dropna(), benchmark.dropna()
    joined = pd.concat([r, b], axis=1, keys=["r", "b"]).dropna()
    cov = joined["r"].rolling(window).cov(joined["b"])
    var = joined["b"].rolling(window).var()
    beta = cov / var
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(beta.index, beta.values, lw=1.7, color=PALETTE["max_sharpe"],
            label=f"Rolling {window}-day beta")
    ax.axhline(1.0, ls="--", c=PALETTE["benchmark"], lw=1.4, label="beta = 1")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Beta vs benchmark")
    ax.legend()
    return finish(fig, ax, "Rolling Beta vs Benchmark", out_path)


def plot_benchmark_scatter(returns, benchmark, freq: str = "M",
                           out_path: str = "figures/benchmark_scatter.png") -> str:
    """
    Compounded returns per period for the portfolio vs the benchmark,
    with the identity line and an OLS fit - a compact "tracking" view.
    """
    setup_style()
    r, b = returns.dropna(), benchmark.dropna()
    joined = pd.concat([r, b], axis=1, keys=["p", "b"]).dropna()
    grouped = (1.0 + joined).groupby(joined.index.to_period(freq)).prod() - 1.0
    x, y = grouped["b"].values, grouped["p"].values
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=70, c=PALETTE["benchmark"], alpha=0.85,
               edgecolors="black", linewidths=0.5, zorder=4)
    lo = float(min(x.min(), y.min())) * 1.1 - 0.01
    hi = float(max(x.max(), y.max())) * 1.1 + 0.01
    ax.plot([lo, hi], [lo, hi], ls="--", c="grey", lw=1.3, label="identity")
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(lo, hi, 50)
    ax.plot(grid, slope * grid + intercept, c=PALETTE["max_sharpe"], lw=1.8,
            label=f"fit: {slope:.2f}x {intercept:+.2%}")
    percent_axis(ax, "x")
    percent_axis(ax, "y")
    ax.set_xlabel("Benchmark return")
    ax.set_ylabel("Portfolio return")
    ax.legend()
    period = {"W": "weekly", "M": "monthly", "Q": "quarterly"}.get(freq, freq)
    title = f"Portfolio vs Benchmark ({period}) Returns"
    return finish(fig, ax, title, out_path)