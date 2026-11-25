"""Tests for mpt.stats: returns, annualization, correlation, shrinkage."""
import numpy as np
import pandas as pd

from mpt.stats import (
    annualize_covariance,
    annualize_returns,
    annualized_volatility,
    calculate_statistics,
    correlation_matrix,
    daily_returns,
)


def test_daily_returns_drops_first_row(prices):
    rets = daily_returns(prices)
    assert len(rets) == len(prices) - 1
    assert not rets.isna().any().any()


def test_daily_returns_values():
    s = pd.Series([100.0, 110.0, 99.0])
    r = daily_returns(s.to_frame("X"))["X"].dropna()
    assert np.isclose(r.iloc[0], 0.10)
    assert np.isclose(r.iloc[1], -0.10)


def test_annualize_returns(prices):
    rets = daily_returns(prices).dropna()
    ann = annualize_returns(rets)
    expected = rets.mean() * 252
    np.testing.assert_allclose(ann.values, expected.values)


def test_annualize_covariance(prices):
    rets = daily_returns(prices).dropna()
    cov = annualize_covariance(rets)
    np.testing.assert_allclose(cov.values, rets.cov().values * 252)


def test_calculate_statistics_consistency(prices):
    daily, exp, cov = calculate_statistics(prices)
    assert daily.shape[1] == prices.shape[1]
    assert len(exp) == prices.shape[1]
    assert cov.shape == (prices.shape[1], prices.shape[1])


def test_covariance_symmetric(stats):
    _, _, cov = stats
    np.testing.assert_allclose(cov.values, cov.values.T)


def test_covariance_psd(stats):
    _, _, cov = stats
    eigvals = np.linalg.eigvalsh(cov.values)
    assert eigvals.min() > -1e-8


def test_correlation_diagonal(prices):
    corr = correlation_matrix(prices)
    np.testing.assert_allclose(np.diag(corr.values), 1.0)


def test_correlation_bounds(prices):
    corr = correlation_matrix(prices)
    assert corr.values.min() >= -1.0
    assert corr.values.max() <= 1.0


def test_annualized_volatility(prices):
    _, _, cov = calculate_statistics(prices)
    vol = annualized_volatility(cov)
    np.testing.assert_allclose(vol.values, np.sqrt(np.diag(cov.values)))
