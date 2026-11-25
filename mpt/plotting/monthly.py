"""
Calendar-style heatmaps of monthly returns for a portfolio
and for every asset.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE, finish, percent_axis, setup_style


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compound daily returns into a year x month table of decimals."""
    r = returns.dropna()
    grouped = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    table = pd.DataFrame({
        "year": [y for y, _ in grouped.index],
        "month": [m for _, m in grouped.index],
        "ret": grouped.values,
    }).pivot(index="year", columns="month", values="ret")
    table = table.reindex(columns=range(1, 13))
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return table


def plot_monthly_heatmap(returns: pd.Series, title: str = "Portfolio",
                         out_path: str = "figures/monthly_returns.png") -> str:
    """Diverging year x month heatmap of compounded returns."""
    setup_style()
    table = monthly_returns(returns)
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(table) + 2.5))
    vmax = float(np.nanmax(np.abs(table.values))) or 0.1
    im = ax.imshow(table.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="Monthly return")
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(len(table)))
    ax.set_xticklabels(table.columns, fontsize=9)
    ax.set_yticklabels(table.index)
    ax.grid(False)
    for i in range(len(table)):
        for j in range(12):
            v = table.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < vmax * 0.6 else "white")
    return finish(fig, ax, f"Monthly Returns - {title}", out_path)


def plot_asset_monthly_heatmap(daily: pd.DataFrame,
                               out_path: str = "figures/asset_monthly_heatmap.png") -> str:
    """Latest-12-months average daily return per asset x month."""
    setup_style()
    last_year = daily.loc[daily.index.max() - pd.DateOffset(years=1):]
    table = last_year.groupby([last_year.index.year, last_year.index.month]).mean() * 252
    # average across the two calendar years present, keep months as columns
    avg = table.groupby(level=1).mean()
    avg = avg.reindex(columns=last_year.columns)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = float(np.nanmax(np.abs(avg.values))) or 0.1
    im = ax.imshow(avg.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="Annualized daily mean")
    ax.set_xticks(np.arange(len(avg.columns)))
    ax.set_yticks(np.arange(12))
    ax.set_xticklabels(avg.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels[:len(avg)], fontsize=9)
    ax.grid(False)
    return finish(fig, ax, "Asset Returns by Month (annualized)", out_path)


def plot_quarterly_returns(returns: pd.Series,
                           out_path: str = "figures/quarterly_returns.png") -> str:
    """Compounded return per calendar quarter, colored by sign."""
    setup_style()
    r = returns.dropna()
    grouped = (1.0 + r).groupby([r.index.year, r.index.quarter]).prod() - 1.0
    labels = [f"{y} Q{q}" for y, q in grouped.index]
    vals = grouped.values
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in vals]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.9)
    percent_axis(ax, "y")
    ax.set_ylabel("Quarterly return")
    ax.tick_params(axis="x", rotation=45)
    return finish(fig, ax, "Returns by Calendar Quarter", out_path)


def plot_annual_returns(series: dict,
                        out_path: str = "figures/annual_returns.png") -> str:
    """Grouped bars: each strategy's compounded return per calendar year."""
    setup_style()
    years = sorted({y for rets in series.values() for y in rets.dropna().index.year})
    x = np.arange(len(years))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = [PALETTE["max_sharpe"], PALETTE["min_vol"], PALETTE["benchmark"],
              "#6a4c93", "#8ac926"]
    for k, (name, rets) in enumerate(series.items()):
        r = rets.dropna()
        vals = [float((1.0 + r[r.index.year == y]).prod() - 1.0) for y in years]
        ax.bar(x + k * width, vals, width, label=name,
               color=colors[k % len(colors)], alpha=0.9)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(years)
    ax.axhline(0, color="black", lw=0.9)
    percent_axis(ax, "y")
    ax.set_ylabel("Calendar-year return")
    ax.legend()
    return finish(fig, ax, "Calendar-Year Returns by Strategy", out_path)
