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


