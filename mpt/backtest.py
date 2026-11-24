"""
Historical backtests: buy-and-hold vs periodically rebalanced
portfolios, with transaction costs and weight tracking.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .metrics import portfolio_performance

@dataclass
class BacktestResult:
    """Everything a backtest produces, ready for charts and reports."""

    equity: pd.Series                       # compounded equity curve
    returns: pd.Series                      # daily portfolio returns
    weights_over_time: pd.DataFrame         # weights after each day's drift
    rebalance_dates: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_cost: float = 0.0

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)


