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


def plot_drawdown(series: dict,
                  out_path: str = "figures/drawdown.png") -> str:
    """Filled underwater plot for each strategy."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = [PALETTE["max_sharpe"], PALETTE["min_vol"], PALETTE["benchmark"], "#6a4c93"]
    for i, (name, rets) in enumerate(series.items()):
        equity = (1.0 + rets.dropna()).cumprod()
        dd = equity / equity.cummax() - 1.0
        ax.fill_between(dd.index, dd.values, 0, alpha=0.25,
                        color=colors[i % len(colors)])
        ax.plot(dd.index, dd.values, lw=1.3, color=colors[i % len(colors)], label=name)
    percent_axis(ax, "y")
    ax.set_ylabel("Drawdown from peak")
    ax.legend()
    return finish(fig, ax, "Underwater Curves", out_path)


def plot_rolling_volatility(series: dict, window: int = 63,
                            out_path: str = "figures/rolling_volatility.png") -> str:
    """Annualized rolling standard deviation (default: one quarter)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = [PALETTE["max_sharpe"], PALETTE["min_vol"], PALETTE["benchmark"], "#6a4c93"]
    for i, (name, rets) in enumerate(series.items()):
        vol = rets.dropna().rolling(window).std() * np.sqrt(252)
        ax.plot(vol.index, vol.values, lw=1.6, color=colors[i % len(colors)], label=name)
    percent_axis(ax, "y")
    ax.set_ylabel(f"Rolling {window}-day vol (annualized)")
    ax.legend()
    return finish(fig, ax, "Rolling Volatility", out_path)


def plot_rolling_sharpe(series: dict, window: int = 126, risk_free: float = 0.02,
                        out_path: str = "figures/rolling_sharpe.png") -> str:
    """Annualized rolling Sharpe ratio (default: six months)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = [PALETTE["max_sharpe"], PALETTE["min_vol"], PALETTE["benchmark"], "#6a4c93"]
    daily_rf = (1.0 + risk_free) ** (1 / 252) - 1.0
    for i, (name, rets) in enumerate(series.items()):
        r = rets.dropna()
        excess = r - daily_rf
        roll = (excess.rolling(window).mean() * 252) / (
            excess.rolling(window).std() * np.sqrt(252))
        ax.plot(roll.index, roll.values, lw=1.6, color=colors[i % len(colors)], label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"Rolling {window}-day Sharpe")
    ax.legend()
    return finish(fig, ax, "Rolling Sharpe Ratio", out_path)


def plot_turnover(result, out_path: str = "figures/turnover.png") -> str:
    """
    Bar chart of L1 turnover paid at each rebalance of a backtest -
    the direct driver of `total_cost`.
    """
    setup_style()
    turn = result.turnover
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(turn) == 0:
        ax.text(0.5, 0.5, "No rebalances in this backtest",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return finish(fig, ax, "Turnover at Each Rebalance", out_path)
    ax.bar(turn.index, turn.values, width=18, color=PALETTE["min_vol"],
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Turnover (L1 distance)")
    percent_axis(ax, "y")
    for d in turn.index:
        ax.axvline(d, color="grey", lw=0.5, alpha=0.4)
    avg = float(turn.mean())
    ax.axhline(avg, ls="--", c=PALETTE["max_sharpe"], lw=1.4,
               label=f"average ({avg:.1%})")
    ax.legend()
    return finish(fig, ax, "Turnover at Each Rebalance", out_path)


def plot_walk_forward(wf, out_path: str = "figures/walk_forward.png") -> str:
    """
    Out-of-sample equity from :func:`mpt.walkforward.walk_forward`, with
    vertical separators at each re-fit date and the summary Sharpe.
    """
    setup_style()
    equity = wf.equity
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(equity.index, equity.values, lw=2, color=PALETTE["max_sharpe"],
            label="Out-of-sample equity")
    for train_end, _ in wf.windows[1:]:
        ax.axvline(train_end, color="grey", lw=0.7, alpha=0.5)
    s = wf.summary()
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_ylabel("Growth of $1 (out-of-sample)")
    ax.legend(loc="upper left")
    title = (f"Walk-Forward Equity - {wf.n_windows} windows, "
             f"Sharpe {s['sharpe']:.2f}, MaxDD {s['max_drawdown']:.1%}")
    return finish(fig, ax, title, out_path)
