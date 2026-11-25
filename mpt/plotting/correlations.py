"""
Heatmaps for correlation and covariance - the fastest way to see
which names actually diversify each other.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import finish, setup_style


def plot_correlation_heatmap(corr: pd.DataFrame,
                             out_path: str = "figures/correlation_heatmap.png") -> str:
    """Triangular-ish full correlation matrix with annotated cells."""
    setup_style()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.grid(False)
    # annotate unless the matrix gets unreadably large
    if n <= 16:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 else "black")
    return finish(fig, ax, "Daily-Return Correlation Matrix", out_path)


def plot_covariance_heatmap(cov: pd.DataFrame,
                            out_path: str = "figures/covariance_heatmap.png") -> str:
    """Annualized covariance heatmap (diagonal = variances)."""
    setup_style()
    n = len(cov)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cov.values, cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Annualized covariance")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cov.columns, rotation=45, ha="right")
    ax.set_yticklabels(cov.index)
    ax.grid(False)
    if n <= 16:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cov.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cov.values[i, j] > cov.values.max() * 0.5 else "black")
    return finish(fig, ax, "Annualized Covariance Matrix", out_path)


def plot_correlation_vs_vol(corr: pd.DataFrame, vol: pd.Series,
                            out_path: str = "figures/correlation_vs_vol.png") -> str:
    """Average pairwise correlation of each asset against its own volatility."""
    setup_style()
    avg_corr = (corr.sum(axis=1) - 1.0) / (len(corr) - 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(vol, avg_corr, s=90, c="#1982c4", edgecolors="black", zorder=4)
    for t in avg_corr.index:
        ax.annotate(t, (vol[t], avg_corr[t]), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Avg. correlation with peers")
    return finish(fig, ax, "Diversification vs Risk", out_path)
