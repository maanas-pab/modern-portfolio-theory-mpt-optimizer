"""Tests for mpt.optimizer: constraints, objectives, edge cases."""
import numpy as np
import pandas as pd
import pytest

from mpt.metrics import portfolio_performance
from mpt.optimizer import (max_sharpe_weights, min_vol_weights,
                           optimize_portfolio, risk_parity_weights)

def test_weights_sum_to_one(stats):
    _, exp, cov = stats
    for w in (max_sharpe_weights(exp, cov, 0.02),
              min_vol_weights(exp, cov),
              risk_parity_weights(cov)):
        assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_long_only_bounds(stats):
    _, exp, cov = stats
    w = max_sharpe_weights(exp, cov, 0.02)
    assert w.min() >= -1e-8
    assert w.max() <= 1.0 + 1e-8


def test_max_sharpe_beats_equal(stats):
    _, exp, cov = stats
    rf = 0.02
    w = max_sharpe_weights(exp, cov, rf)
    _, _, s_opt = portfolio_performance(w, exp, cov, rf)
    eq = np.repeat(1 / len(exp), len(exp))
    _, _, s_eq = portfolio_performance(eq, exp, cov, rf)
    assert s_opt >= s_eq - 1e-6


def test_min_vol_is_minimum(stats):
    _, exp, cov = stats
    w = min_vol_weights(exp, cov)
    _, v_opt, _ = portfolio_performance(w, exp, cov)
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.random(len(exp))
        x /= x.sum()
        _, v, _ = portfolio_performance(x, exp, cov)
        assert v_opt <= v + 1e-6


def test_min_vol_target_return(stats):
    _, exp, cov = stats
    target = float(exp.mean())
    w = min_vol_weights(exp, cov, target_return=target)
    assert np.isclose(float(w @ exp.values), target, atol=1e-5)


def test_infeasible_target_raises(stats):
    _, exp, cov = stats
    with pytest.raises(RuntimeError):
        min_vol_weights(exp, cov, target_return=float(exp.max()) * 5)


def test_risk_parity_equal_contributions(stats):
    _, _, cov = stats
    w = risk_parity_weights(cov)
    from mpt.metrics import risk_contributions

    _, pct = risk_contributions(w, cov)
    assert pct.max() - pct.min() < 0.05


def test_unknown_objective(stats):
    _, exp, cov = stats
    with pytest.raises(ValueError, match="objective"):
        optimize_portfolio(exp, cov, objective="drawdown")


