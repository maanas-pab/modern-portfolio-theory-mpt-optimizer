"""
SLSQP-based portfolio optimization: max Sharpe, min volatility,
target-return minimum vol, and equal risk budgeting.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import portfolio_performance, risk_budget_loss

def _neg_sharpe(weights, exp_returns, cov_matrix, risk_free_rate) -> float:
    _, _, s = portfolio_performance(weights, exp_returns, cov_matrix, risk_free_rate)
    return -s


