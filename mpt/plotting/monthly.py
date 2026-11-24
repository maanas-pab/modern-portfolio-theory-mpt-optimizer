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


