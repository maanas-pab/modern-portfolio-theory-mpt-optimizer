"""
The flagship chart: Monte-Carlo cloud, efficient frontier,
capital market line, and individual assets in risk-return space.
"""
import matplotlib.pyplot as plt
import numpy as np

from ..frontier import capital_market_line
from .style import PALETTE, finish, percent_axis, setup_style

def plot_frontier(frontier_rets, frontier_vols,
                  rand_r, rand_v, rand_s,
                  max_sharpe_pt, min_vol_pt,
                  asset_rets=None, asset_vols=None, asset_labels=None,
                  risk_free: float = 0.0,
                  out_path: str = "figures/efficient_frontier.png") -> str:
    """
    Frontier over the Monte-Carlo cloud, colored by Sharpe ratio,
    with the tangency/min-vol points and (optionally) the capital
    market line plus every individual asset.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    sc = ax.scatter(rand_v, rand_r, c=rand_s, cmap="viridis", alpha=0.35,
                    s=12, label="Random portfolios")
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")

    ax.plot(frontier_vols, frontier_rets, color=PALETTE["frontier"], lw=2.5,
            label="Efficient Frontier", zorder=4)

    if asset_vols is not None:
        ax.scatter(asset_vols, asset_rets, marker="o", s=55, c=PALETTE["random"],
                   edgecolors="black", linewidths=0.6, zorder=5, label="Individual assets")
        for label, v, r in zip(asset_labels, asset_vols, asset_rets):
            ax.annotate(label, (v, r), textcoords="offset points", xytext=(5, 4),
                        fontsize=8.5, alpha=0.85)

    cml_v, cml_r = capital_market_line(risk_free, max_sharpe_pt[0], max_sharpe_pt[1],
                                       vol_max=float(np.max(rand_v)) * 1.05)
    ax.plot(cml_v, cml_r, ls="--", lw=1.6, color=PALETTE["benchmark"],
            label="Capital Market Line", zorder=3)

    ax.scatter(*max_sharpe_pt[1::-1], c=PALETTE["max_sharpe"], s=160, marker="*",
               edgecolors="black", label="Max Sharpe", zorder=6)
    ax.scatter(*min_vol_pt[1::-1], c=PALETTE["min_vol"], s=120, marker="D",
               edgecolors="black", label="Min Volatility", zorder=6)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Expected Return")
    percent_axis(ax, "x")
    percent_axis(ax, "y")
    return finish(fig, ax, "Efficient Frontier - 12 Niche Stocks (MPT)", out_path)


