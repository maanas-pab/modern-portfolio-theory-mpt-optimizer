"""Tests for mpt.metrics: ratios, drawdown, VaR, beta, contributions."""
import numpy as np
import pandas as pd
import pytest

from mpt.metrics import (
    alpha,
    beta,
    calmar_ratio,
    concentration_hhi,
    conditional_var,
    downside_deviation,
    max_drawdown,
    portfolio_performance,
    risk_budget_loss,
    risk_contributions,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


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


def test_max_drawdown_known_series():
    r = pd.Series([0.10, -0.50, 0.10, 0.10])
    assert np.isclose(max_drawdown(r), -0.50)


def test_max_drawdown_no_losses():
    r = pd.Series([0.01, 0.02, 0.03, 0.04])
    assert max_drawdown(r) == 0.0


def test_calmar_arithmetic():
    assert np.isclose(calmar_ratio(0.20, -0.10), 2.0)


def test_calmar_no_drawdown_guard():
    assert calmar_ratio(0.20, 0.0) == 0.0


def test_historical_var_positive_loss():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0005, 0.01, 5000))
    var = value_at_risk(r, 0.95, "historical")
    assert var > 0
    # roughly the 5% quantile magnitude
    assert np.isclose(var, -np.quantile(r, 0.05), atol=1e-6)


def test_parametric_var_close_to_historical():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0, 0.01, 20000))
    hist = value_at_risk(r, 0.95, "historical")
    para = value_at_risk(r, 0.95, "parametric")
    assert abs(hist - para) < 0.002


def test_var_rejects_bad_level():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100))
    with pytest.raises(ValueError, match="level"):
        value_at_risk(r, 0.3)


def test_var_rejects_unknown_method():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100))
    with pytest.raises(ValueError, match="method"):
        value_at_risk(r, 0.95, "cornish-fisher")


def test_cvar_at_least_var():
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    assert conditional_var(r, 0.95) >= value_at_risk(r, 0.95) - 1e-12


def test_beta_identical_series():
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0, 0.01, 300))
    assert np.isclose(beta(r, r), 1.0)


def test_beta_uncorrelated():
    rng = np.random.default_rng(10)
    a = pd.Series(rng.normal(0, 0.01, 3000))
    b = pd.Series(rng.normal(0, 0.01, 3000))
    assert abs(beta(a, b)) < 0.1


def test_alpha_self_zero():
    rng = np.random.default_rng(12)
    r = pd.Series(rng.normal(0.0004, 0.01, 500))
    assert abs(alpha(r, r, risk_free=0.0)) < 1e-9


def test_hhi_equal_weights():
    assert np.isclose(concentration_hhi([0.25] * 4), 0.25)


def test_hhi_concentrated():
    assert np.isclose(concentration_hhi([1.0, 0.0, 0.0]), 1.0)


def test_risk_contributions_sum(two_asset):
    _, cov = two_asset
    w = np.array([0.6, 0.4])
    _, pct = risk_contributions(w, cov)
    assert np.isclose(pct.sum(), 1.0)


def test_risk_contributions_symmetric():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["A", "B"],
                       columns=["A", "B"])
    _, pct = risk_contributions([0.5, 0.5], cov)
    np.testing.assert_allclose(pct, [0.5, 0.5])


def test_budget_loss_zero_at_equal_risk():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["A", "B"],
                       columns=["A", "B"])
    assert np.isclose(risk_budget_loss([0.5, 0.5], cov), 0.0, atol=1e-12)
