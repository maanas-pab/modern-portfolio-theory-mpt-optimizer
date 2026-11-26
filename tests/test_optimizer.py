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


