"""
Return statistics: daily returns, annualized means, covariance
matrices, correlations, and shrinkage estimators.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from .config import TRADING_DAYS

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Percentage change of prices, first row dropped."""
    return prices.pct_change().dropna(how="all")


def annualize_returns(daily: pd.DataFrame, days: int = TRADING_DAYS) -> pd.Series:
    """Mean daily return scaled to a yearly figure."""
    return daily.mean() * days


def annualize_covariance(daily: pd.DataFrame, days: int = TRADING_DAYS) -> pd.DataFrame:
    """Sample covariance of daily returns scaled to a yearly figure."""
    return daily.cov() * days


def calculate_statistics(prices: pd.DataFrame,
                         days: int = TRADING_DAYS) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
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


