"""
Return statistics: daily returns, annualized means, covariance
matrices, correlations, and shrinkage estimators.
"""

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Percentage change of prices, first row dropped.

    Forward-fills explicitly first: pandas < 3 did this inside
    ``pct_change()`` by default, pandas >= 3 does not, and the two
    must agree for the tests (and the backtest) to be version-stable.
    """
    return prices.ffill().pct_change().dropna(how="all")


def annualize_returns(daily: pd.DataFrame, days: int = TRADING_DAYS) -> pd.Series:
    """Mean daily return scaled to a yearly figure."""
    return daily.mean() * days


def annualize_covariance(daily: pd.DataFrame, days: int = TRADING_DAYS) -> pd.DataFrame:
    """Sample covariance of daily returns scaled to a yearly figure."""
    return daily.cov() * days


def calculate_statistics(prices: pd.DataFrame,
                         days: int = TRADING_DAYS) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Return (daily_returns, annualized_expected_returns, annualized_covariance).
    """
    daily = daily_returns(prices).dropna()
    exp_returns = annualize_returns(daily, days)
    cov_matrix = annualize_covariance(daily, days)
    return daily, exp_returns, cov_matrix


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily returns."""
    return daily_returns(prices).corr()


def annualized_volatility(cov_matrix: pd.DataFrame) -> pd.Series:
    """Per-asset annualized standard deviation from the covariance diagonal."""
    return pd.Series(np.sqrt(np.diag(cov_matrix.values)), index=cov_matrix.index)


def ledoit_wolf_intensity(daily: pd.DataFrame) -> float:
    """
    Analytic shrinkage intensity toward the scaled-identity target.

    Implements the Ledoit & Wolf (2004) well-conditioned estimator:
    lambda* = min(1, b^2 / d^2) where d^2 is the squared distance from
    the sample covariance to the target and b^2 estimates the squared
    estimation error of the sample covariance (Gaussian fourth moments).
    """
    x = daily.values - daily.values.mean(axis=0, keepdims=True)
    t, n = x.shape
    if t < 2 or n < 2:
        return 0.0
    s = x.T @ x / t
    mu = np.trace(s) / n
    delta = s - mu * np.eye(n)
    d2 = float(np.sum(delta ** 2)) / n
    if d2 <= 1e-18:
        return 0.0
    g2 = (np.trace(s) ** 2 + float(np.trace(s @ s))) / (t * n)
    return float(min(1.0, g2 / d2))


def ledoit_wolf_covariance(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Annualized covariance shrunk toward mu * I.

    Shrinking eigenvalues away from zero makes the matrix far better
    conditioned for inverting in optimizers and risk models.
    """
    daily = daily.dropna()
    lam = ledoit_wolf_intensity(daily)
    cov = daily.cov() * TRADING_DAYS
    mu = float(np.trace(cov.values)) / len(cov)
    shrunk = (1.0 - lam) * cov.values + lam * mu * np.eye(len(cov))
    out = pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)
    out.attrs["shrinkage_intensity"] = lam
    return out


def covariance_matrix(daily: pd.DataFrame, shrinkage: bool = False) -> pd.DataFrame:
    """Sample or shrunk annualized covariance depending on the flag."""
    if shrinkage:
        return ledoit_wolf_covariance(daily)
    return annualize_covariance(daily)
