"""
Risk/return metrics: Sharpe, Sortino, drawdown, VaR/CVaR, beta,
concentration and risk contributions.

Functions come in two flavours: scalars that combine already-annualized
figures, and series-based helpers that work on daily return histories.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from .config import TRADING_DAYS

def sharpe_ratio(annual_return: float, annual_vol: float, risk_free: float = 0.0) -> float:
    """(mu - rf) / sigma with a zero-vol guard."""
    if annual_vol <= 0:
        return 0.0
    return float((annual_return - risk_free) / annual_vol)


def portfolio_performance(weights, exp_returns, cov_matrix, risk_free_rate: float = 0.0
                          ) -> Tuple[float, float, float]:
    """Annualized (return, volatility, sharpe) for a weight vector."""
    w = np.asarray(weights, dtype=float)
    ret = float(w @ exp_returns.values)
    vol = float(np.sqrt(w @ cov_matrix.values @ w))
    return ret, vol, sharpe_ratio(ret, vol, risk_free_rate)


