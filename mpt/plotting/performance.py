"""
Time-series performance: cumulative growth, underwater drawdown,
rolling volatility and rolling Sharpe.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE, finish, percent_axis, setup_style

def plot_cumulative_returns(series: dict,
                            benchmark: pd.Series = None,
                            out_path: str = "figures/cumulative_returns.png") -> str:
    """
    Growth of $1 for each named return series (plus SPY when given).
    series maps name -> daily returns.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 6))
    colors = [PALETTE["max_sharpe"], PALETTE["min_vol"], PALETTE["benchmark"],
              "#6a4c93", "#8ac926"]
    for i, (name, rets) in enumerate(series.items()):
        equity = (1.0 + rets.dropna()).cumprod()
        ax.plot(equity.index, equity.values, lw=2,
                color=colors[i % len(colors)], label=name)
    if benchmark is not None:
        bench = (1.0 + benchmark.dropna()).cumprod()
        ax.plot(bench.index, bench.values, lw=1.6, ls="--",
                color=PALETTE["random"], label=benchmark.name or "benchmark")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    return finish(fig, ax, "Cumulative Growth by Strategy", out_path)


