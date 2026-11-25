"""Edge cases and property tests for mpt.metrics."""
import numpy as np
import pandas as pd
import pytest

from mpt.metrics import (alpha, beta, calmar_ratio, concentration_hhi,
                         conditional_var, downside_deviation, max_drawdown,
                         portfolio_performance, risk_budget_loss,
                         risk_contributions, sharpe_ratio, sortino_ratio,
                         value_at_risk)

def test_sharpe_negative():
    assert np.isclose(sharpe_ratio(-0.05, 0.20, 0.02), -0.35)


def test_sharpe_zero_return():
    assert np.isclose(sharpe_ratio(0.0, 0.10, 0.02), -0.2)


def test_single_asset_performance(two_asset):
    exp, cov = two_asset
    r, v, s = portfolio_performance([1.0, 0.0], exp, cov, 0.02)
    assert np.isclose(r, 0.10)
    assert np.isclose(v, 0.20)
    assert np.isclose(s, (0.10 - 0.02) / 0.20)


def test_downside_deviation_empty():
    assert downside_deviation(pd.Series(dtype=float)) == 0.0


def test_downside_deviation_periods_scaling():
    r = pd.Series([-0.01] * 50)
    daily = downside_deviation(r, 0.0, periods=252)
    weekly = downside_deviation(r, 0.0, periods=52)
    assert np.isclose(daily / weekly, np.sqrt(252 / 52))


def test_sortino_empty():
    assert sortino_ratio(pd.Series(dtype=float)) == 0.0


