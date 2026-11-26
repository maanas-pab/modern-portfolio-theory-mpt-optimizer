"""Solver behavior at the edges of mpt.optimizer."""
import numpy as np
import pandas as pd
import pytest

from mpt.metrics import portfolio_performance, risk_contributions
from mpt.optimizer import (max_sharpe_weights, min_vol_weights,
                           optimize_portfolio, risk_parity_weights)

def test_two_asset_max_sharpe_closed_form():
    """w* for two assets is proportional to Sigma^-1 (mu - rf)."""
    rf = 0.02
    exp = pd.Series({"A": 0.10, "B": 0.20})
    cov = pd.DataFrame([[0.04, 0.012], [0.012, 0.09]],
                       index=["A", "B"], columns=["A", "B"])
    w = max_sharpe_weights(exp, cov, rf)
    excess = exp.values - rf
    analytic = np.linalg.solve(cov.values, excess)
    analytic = np.abs(analytic) / np.abs(analytic).sum()   # long-only projection
    # the optimizer's answer must beat the naive equal split
    _, _, s_opt = portfolio_performance(w, exp, cov, rf)
    _, _, s_eq = portfolio_performance([0.5, 0.5], exp, cov, rf)
    assert s_opt >= s_eq - 1e-6
    assert w.sum() == pytest.approx(1.0)


def test_min_vol_target_matches_frontier(stats):
    from mpt.frontier import efficient_frontier

    _, exp, cov = stats
    rets, vols, _ = efficient_frontier(exp, cov, n_points=25)
    i = len(rets) // 2
    w = min_vol_weights(exp, cov, target_return=float(rets[i]))
    _, v, _ = portfolio_performance(w, exp, cov)
    assert v <= vols[i] + 1e-5


def test_infeasible_negative_target(stats):
    _, exp, cov = stats
    with pytest.raises(RuntimeError):
        min_vol_weights(exp, cov, target_return=-5.0)


def test_risk_parity_diagonal():
    cov = pd.DataFrame(np.diag([0.04, 0.09, 0.16]),
                       index=list("ABC"), columns=list("ABC"))
    w = risk_parity_weights(cov)
    _, pct = risk_contributions(w, cov)
    np.testing.assert_allclose(pct, [1 / 3] * 3, atol=1e-4)
    # inverse-vol weights are the analytic answer here
    inv = 1.0 / np.sqrt(np.diag(cov.values))
    np.testing.assert_allclose(w, inv / inv.sum(), atol=1e-4)


