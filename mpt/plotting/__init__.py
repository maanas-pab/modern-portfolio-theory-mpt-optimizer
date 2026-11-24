"""Chart functions for every figure in the gallery."""
from .allocation import (plot_concentration_over_time, plot_risk_contribution,
                         plot_weight_trajectories, plot_weights_comparison,
                         plot_weights_over_time, plot_weights_pies)
from .correlations import (plot_correlation_heatmap, plot_correlation_vs_vol,
                           plot_covariance_heatmap)
from .distributions import plot_return_distributions, plot_sector_exposure
from .frontier import (plot_frontier, plot_frontier_clean,
                       plot_frontier_shrinkage)

from .monthly import (plot_annual_returns, plot_asset_monthly_heatmap,
                      plot_monthly_heatmap, plot_quarterly_returns)
from .performance import (plot_cumulative_returns, plot_drawdown,
                          plot_rolling_sharpe, plot_rolling_volatility,
                          plot_turnover, plot_walk_forward)

from .risk import (plot_asset_risk_return, plot_benchmark_scatter,
                   plot_metric_comparison, plot_rolling_beta,
                   plot_sharpe_histogram, plot_var_cvar)
from .style import PALETTE, setup_style

