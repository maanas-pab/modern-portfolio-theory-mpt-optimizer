"""
Efficient frontier construction, Monte-Carlo portfolio clouds,
and the capital market line.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import portfolio_performance
from .optimizer import max_sharpe_weights, optimize_portfolio

def efficient_frontier(exp_returns: pd.Series,
                       cov_matrix: pd.DataFrame,
                       n_points: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trace the frontier by minimizing volatility over a grid of
    target returns spanning min..max single-asset expected return.

    Returns (returns, vols, weights) for the targets that solved.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    targets = np.linspace(float(exp_returns.min()), float(exp_returns.max()), n_points)
    rets, vols, weights = [], [], []
    for t in targets:
        try:
            w = optimize_portfolio(exp_returns, cov_matrix, objective="vol", target_return=t)
        except RuntimeError:
            continue
        r, v, _ = portfolio_performance(w, exp_returns, cov_matrix)
        rets.append(r)
        vols.append(v)
        weights.append(w)
    return np.array(rets), np.array(vols), np.array(weights)


def random_portfolios(exp_returns: pd.Series,
                      cov_matrix: pd.DataFrame,
                      risk_free_rate: float = 0.0,
                      n: int = 5000,
                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly random long-only weights -> (returns, vols, sharpes)."""
    rng = np.random.default_rng(seed)
    n_assets = len(exp_returns)
    out_r = np.empty(n)
    out_v = np.empty(n)
    out_s = np.empty(n)
    for i in range(n):
        w = rng.random(n_assets)
        w /= w.sum()
        r, v, s = portfolio_performance(w, exp_returns, cov_matrix, risk_free_rate)
        out_r[i], out_v[i], out_s[i] = r, v, s
    return out_r, out_v, out_s


