"""
mpt - a Modern Portfolio Theory toolkit.

Public API::

    from mpt import get_prices, calculate_statistics, optimize_portfolio
"""
from .config import BENCHMARK, DEFAULT_TICKERS, THEMES, TRADING_DAYS, OptimizerConfig
from .data import fetch_prices, get_prices, load_cache, save_cache
from .stats import calculate_statistics, covariance_matrix, daily_returns

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "BENCHMARK",
    "DEFAULT_TICKERS",
    "THEMES",
    "TRADING_DAYS",
    "OptimizerConfig",
    "fetch_prices",
    "get_prices",
    "load_cache",
    "save_cache",
    "calculate_statistics",
    "covariance_matrix",
    "daily_returns",
]
