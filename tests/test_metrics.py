"""Tests for mpt.metrics: ratios, drawdown, VaR, beta, contributions."""
import numpy as np
import pandas as pd
import pytest

from mpt.metrics import (alpha, beta, calmar_ratio, concentration_hhi,
                         conditional_var, downside_deviation, max_drawdown,
                         portfolio_performance, risk_budget_loss,
                         risk_contributions, sharpe_ratio, sortino_ratio,
                         value_at_risk)

def test_sharpe_ratio_arithmetic():
    assert np.isclose(sharpe_ratio(0.10, 0.20, 0.02), 0.4)


def test_sharpe_zero_vol_guard():
    assert sharpe_ratio(0.10, 0.0, 0.02) == 0.0


def test_portfolio_performance_equal_weights(two_asset):
    exp, cov = two_asset
    r, v, s = portfolio_performance([0.5, 0.5], exp, cov, 0.0)
    assert np.isclose(r, 0.15)
    expected_var = 0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.03
    assert np.isclose(v, np.sqrt(expected_var))
    assert np.isclose(s, 0.15 / v)


def test_portfolio_return_linear(two_asset):
    exp, cov = two_asset
    r1, _, _ = portfolio_performance([1.0, 0.0], exp, cov)
    r2, _, _ = portfolio_performance([0.0, 1.0], exp, cov)
    r3, _, _ = portfolio_performance([0.5, 0.5], exp, cov)
    assert np.isclose(r3, 0.5 * r1 + 0.5 * r2)


def test_downside_deviation_positive_returns():
    r = pd.Series([0.01] * 100)
    assert downside_deviation(r, risk_free=0.0) == 0.0


def test_downside_deviation_negative_series():
    r = pd.Series([-0.01] * 100)
    assert downside_deviation(r, risk_free=0.0) > 0.0


def test_sortino_asymmetric_series():
    rng = np.random.default_rng(3)
    r = pd.Series(np.where(rng.random(1000) > 0.7, -0.02, 0.01))
    s = sortino_ratio(r, 0.0)
    assert np.isfinite(s)


def test_sortino_zero_downside_guard():
    r = pd.Series([0.01] * 50)
    assert sortino_ratio(r, 0.0) == 0.0


