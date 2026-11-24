"""
Per-asset return distributions - histograms and violin plots
that show just how fat-tailed these niche names are.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import ASSET_COLORS, asset_color, finish, percent_axis, setup_style

def plot_return_distributions(daily: pd.DataFrame, cols: int = 4,
                              out_path: str = "figures/return_distributions.png") -> str:
    """Small-multiples of daily return histograms for every asset."""
    setup_style()
    n = daily.shape[1]
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    bins = np.histogram_bin_edges(daily.values.ravel(), bins=32)
    for i, ax in enumerate(axes):
        if i >= n:
            ax.set_visible(False)
            continue
        col = daily.columns[i]
        ax.hist(daily[col].dropna(), bins=bins, color=asset_color(i),
                alpha=0.9, edgecolor="white")
        ax.axvline(daily[col].mean(), color="black", lw=1, ls="--")
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=7)
        percent_axis(ax, "x")
    fig.suptitle("Daily Return Distributions", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved chart -> {out_path}")
    return out_path


def plot_sector_exposure(weights, tickers, sector_map: dict,
                         out_path: str = "figures/sector_exposure.png") -> str:
    """Aggregate weights into sector buckets and show one pie."""
    setup_style()
    buckets: dict = {}
    for t, w in zip(tickers, np.asarray(weights, dtype=float)):
        key = sector_map.get(t, "Other")
        buckets[key] = buckets.get(key, 0.0) + w
    labels = sorted(buckets, key=buckets.get, reverse=True)
    values = [buckets[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
           colors=[ASSET_COLORS[i % len(ASSET_COLORS)] for i in range(len(labels))],
           wedgeprops={"linewidth": 1, "edgecolor": "white"},
           pctdistance=0.78)
    ax.axis("equal")
    return finish(fig, ax, "Sector Exposure", out_path)