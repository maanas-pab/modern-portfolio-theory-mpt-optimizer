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


