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


def _portfolio_vol(weights, exp_returns, cov_matrix, risk_free_rate=None) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ cov_matrix.values @ w))


def _sum_to_one(weights) -> float:
    return float(np.sum(weights) - 1.0)


def _target_return(weights, exp_returns, target) -> float:
    return float(np.asarray(weights) @ exp_returns.values - target)


