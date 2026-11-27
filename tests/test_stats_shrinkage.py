"""Tests for the Ledoit-Wolf estimators in mpt.stats."""
import numpy as np

from mpt.stats import (calculate_statistics, covariance_matrix,
                       ledoit_wolf_covariance, ledoit_wolf_intensity)

def test_intensity_bounds(prices):
    daily, _, _ = calculate_statistics(prices)
    lam = ledoit_wolf_intensity(daily)
    assert 0.0 <= lam <= 1.0


def test_intensity_small_sample(prices):
    daily = prices.pct_change().dropna().head(3)
    lam = ledoit_wolf_intensity(daily)
    assert 0.0 <= lam <= 1.0


def test_shrunk_cov_symmetric(prices):
    daily = prices.pct_change().dropna()
    cov = ledoit_wolf_covariance(daily)
    np.testing.assert_allclose(cov.values, cov.values.T)


def test_shrunk_cov_positive_definite(prices):
    daily = prices.pct_change().dropna()
    cov = ledoit_wolf_covariance(daily)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert eigvals.min() > 0.0


def test_intensity_in_attrs(prices):
    daily = prices.pct_change().dropna()
    cov = ledoit_wolf_covariance(daily)
    assert "shrinkage_intensity" in cov.attrs


def test_dispatcher_flag(prices):
    daily = prices.pct_change().dropna()
    plain = covariance_matrix(daily, shrinkage=False)
    shrunk = covariance_matrix(daily, shrinkage=True)
    assert not np.allclose(plain.values, shrunk.values)
    assert list(plain.index) == list(shrunk.index)
    assert shrunk.attrs.get("shrinkage_intensity", -1) >= 0.0


