"""Boundary behavior for mpt.stats."""
import numpy as np
import pandas as pd

from mpt.stats import (
    annualize_covariance,
    annualize_returns,
    annualized_volatility,
    calculate_statistics,
    correlation_matrix,
    covariance_matrix,
    daily_returns,
    ledoit_wolf_covariance,
    ledoit_wolf_intensity,
)


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


def test_volatility_zero_covariance():
    cov = pd.DataFrame([[0.0]], index=["CASH"], columns=["CASH"])
    assert annualized_volatility(cov).iloc[0] == 0.0


def test_intensity_single_asset():
    daily = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, 100)})
    lam = ledoit_wolf_intensity(daily)
    assert 0.0 <= lam <= 1.0


def test_shrinkage_conditions_collinear_assets():
    x = np.random.default_rng(1).normal(0, 0.01, 300)
    daily = pd.DataFrame({"A": x, "B": x * 1.0 + 1e-9})
    plain = covariance_matrix(daily, shrinkage=False)
    shrunk = covariance_matrix(daily, shrinkage=True)
    cond_plain = np.linalg.cond(plain.values)
    cond_shrunk = np.linalg.cond(shrunk.values)
    assert cond_shrunk < cond_plain


def test_shrunk_preserves_order(prices):
    daily = daily_returns(prices).dropna()
    shrunk = ledoit_wolf_covariance(daily)
    assert list(shrunk.index) == list(daily.columns)
    assert list(shrunk.columns) == list(daily.columns)


def test_dispatcher_default_is_sample(prices):
    daily = daily_returns(prices).dropna()
    default = covariance_matrix(daily)
    explicit = covariance_matrix(daily, shrinkage=False)
    np.testing.assert_allclose(default.values, explicit.values)


def test_dispatcher_attrs(prices):
    daily = daily_returns(prices).dropna()
    cov = covariance_matrix(daily, shrinkage=True)
    assert "shrinkage_intensity" in cov.attrs
    assert 0.0 <= cov.attrs["shrinkage_intensity"] <= 1.0
