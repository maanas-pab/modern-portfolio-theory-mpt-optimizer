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


def _base_constraints(exp_returns, target=None):
    constraints = [{"type": "eq", "fun": _sum_to_one}]
    if target is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w, t=target: _target_return(w, exp_returns, t)}
        )
    return constraints


def optimize_portfolio(exp_returns: pd.Series,
                       cov_matrix: pd.DataFrame,
                       risk_free_rate: float = 0.0,
                       objective: str = "sharpe",
                       target_return: Optional[float] = None,
                       bounds: Optional[Sequence[Tuple[float, float]]] = None,
                       budget: str = "sum",
                       x0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Single entry point for all SLSQP optimizations.

    objective: 'sharpe' | 'vol' | 'risk_parity'
    budget:    'sum' (fully invested) | 'gross' (sum |w| = 1, for shorts)
    x0:        optional warm start; defaults to equal weights.  For
               long-short ('gross') problems, seeding with a short
               avoids SLSQP stopping on the |w| kink at zero.
    """
    n = len(exp_returns)
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))  # long-only, no leverage
    if x0 is None:
        x0 = np.repeat(1.0 / n, n)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},)")
    budget_fun = _sum_to_one if budget == "sum" else (
        lambda w: float(np.sum(np.abs(w)) - 1.0))
    if budget not in ("sum", "gross"):
        raise ValueError("budget must be 'sum' or 'gross'")

    if objective == "sharpe":
        fun, args = _neg_sharpe, (exp_returns, cov_matrix, risk_free_rate)
        constraints = [{"type": "eq", "fun": budget_fun}]
    elif objective == "vol":
        fun, args = _portfolio_vol, (exp_returns, cov_matrix, None)
        constraints = [{"type": "eq", "fun": budget_fun}]
        if target_return is not None:
            constraints.append(
                {"type": "eq", "fun": lambda w, t=target_return: _target_return(w, exp_returns, t)}
            )
    elif objective == "risk_parity":
        fun, args = risk_budget_loss, (cov_matrix,)
        constraints = [{"type": "eq", "fun": budget_fun}]
    else:
        raise ValueError("objective must be 'sharpe', 'vol' or 'risk_parity'")

    res = minimize(fun, x0, args=args, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success:
        raise RuntimeError(f"Optimization failed ({objective}): {res.message}")
    w = np.asarray(res.x, dtype=float)
    return w


def max_sharpe_weights(exp_returns, cov_matrix, risk_free_rate: float = 0.0) -> np.ndarray:
    """Tangency portfolio: max (ret - rf) / vol."""
    return optimize_portfolio(exp_returns, cov_matrix, risk_free_rate, objective="sharpe")


