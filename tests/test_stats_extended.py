"""Boundary behavior for mpt.stats."""
import numpy as np
import pandas as pd
import pytest

from mpt.stats import (annualize_covariance, annualize_returns,
                       annualized_volatility, calculate_statistics,
                       correlation_matrix, covariance_matrix, daily_returns,
                       ledoit_wolf_covariance, ledoit_wolf_intensity)

def test_annualize_custom_days(prices):
    rets = daily_returns(prices).dropna()
    ann = annualize_returns(rets, days=100)
    np.testing.assert_allclose(ann.values, (rets.mean() * 100).values)


def test_annualize_cov_custom_days(prices):
    rets = daily_returns(prices).dropna()
    cov = annualize_covariance(rets, days=100)
    np.testing.assert_allclose(cov.values, rets.cov().values * 100)


def test_daily_returns_handles_nan():
    prices = pd.DataFrame({"A": [100.0, 101.0, np.nan, 103.0, 104.0],
                           "B": [50.0, 49.0, 48.0, 47.0, 46.0]},
                          index=pd.bdate_range("2024-01-01", periods=5))
    r = daily_returns(prices)
    # first row (all-NaN pct) is dropped
    assert r.index[0] == prices.index[1]
    # pandas forward-fills the NaN price gap: the gap day reads as flat
    assert r["A"].notna().all()
    assert r["A"].iloc[1] == 0.0
    assert r["B"].notna().all()


def test_calculate_statistics_nan_tolerant():
    prices = pd.DataFrame({
        "A": [100.0, 101.0, np.nan, 103.0, 104.0],
        "B": [50.0, 49.0, 48.0, 47.0, 46.0],
    }, index=pd.bdate_range("2024-01-01", periods=5))
    daily, exp, cov = calculate_statistics(prices)
    assert np.isfinite(exp).all()
    assert np.isfinite(cov.values).all()


def test_correlation_constant_series():
    prices = pd.DataFrame({
        "A": [100.0] * 10,
        "B": [float(i) for i in range(10)],
    })
    corr = correlation_matrix(prices)
    assert np.isnan(corr.loc["A", "A"]) or corr.loc["A", "A"] == 1.0


