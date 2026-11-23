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


def value_at_risk(returns: pd.Series, level: float = 0.95,
                  method: str = "historical") -> float:
    """
    Daily VaR as a positive loss fraction, e.g. 0.031 = 3.1% one-day loss.

    method='historical' uses the empirical quantile; method='parametric'
    assumes a normal distribution.
    """
    r = returns.dropna()
    if r.empty:
        return 0.0
    if not 0.5 < level < 1.0:
        raise ValueError("level must be in (0.5, 1)")
    if method == "historical":
        q = float(np.quantile(r, 1.0 - level))
    elif method == "parametric":
        from statistics import NormalDist

        z = NormalDist().inv_cdf(level)
        q = float(r.mean() - z * r.std(ddof=1))
    else:
        raise ValueError("method must be 'historical' or 'parametric'")
    return float(max(-q, 0.0))


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """Average loss beyond the VaR threshold (expected shortfall)."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    var = value_at_risk(r, level=level, method="historical")
    tail = r[r <= -var]
    if tail.empty:
        return var
    return float(max(-tail.mean(), 0.0))


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Covariance with the benchmark divided by benchmark variance."""
    joined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 2:
        return 0.0
    p, b = joined.iloc[:, 0], joined.iloc[:, 1]
    var = float(b.var())
    if var <= 0:
        return 0.0
    return float(p.cov(b) / var)


def alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
          risk_free: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualized alpha from the CAPM regression (intercept-based)."""
    joined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 2:
        return 0.0
    p, b = joined.iloc[:, 0], joined.iloc[:, 1]
    b_var = float(b.var())
    if b_var <= 0:
        return 0.0
    b_beta = float(p.cov(b) / b_var)
    daily_rf = (1.0 + risk_free) ** (1.0 / periods) - 1.0
    intercept = float(p.mean() - daily_rf - b_beta * (b.mean() - daily_rf))
    return float(((1.0 + intercept) ** periods - 1.0) - risk_free)


def concentration_hhi(weights) -> float:
    """Herfindahl-Hirschman index of weight concentration (1/N .. 1)."""
    w = np.asarray(weights, dtype=float)
    w = np.abs(w) / max(np.abs(w).sum(), 1e-12)
    return float(np.sum(w ** 2))


