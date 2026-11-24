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


