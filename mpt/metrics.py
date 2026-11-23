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


def downside_deviation(returns: pd.Series, risk_free: float = 0.0,
                       periods: int = TRADING_DAYS) -> float:
    """Annualized standard deviation of returns below the risk-free threshold."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    daily_rf = (1.0 + risk_free) ** (1.0 / periods) - 1.0
    downside = np.minimum(r - daily_rf, 0.0)
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0,
                  periods: int = TRADING_DAYS) -> float:
    """Annualized excess return per unit of downside deviation."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    annual_ret = float(r.mean() * periods)
    dd = downside_deviation(r, risk_free, periods)
    if dd <= 0:
        return 0.0
    return float((annual_ret - risk_free) / dd)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the compounded equity curve (<= 0)."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(annual_return: float, max_dd: float, risk_free: float = 0.0) -> float:
    """Excess annual return divided by the magnitude of max drawdown."""
    if max_dd >= 0:
        return 0.0
    return float((annual_return - risk_free) / abs(max_dd))


