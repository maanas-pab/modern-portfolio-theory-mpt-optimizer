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


