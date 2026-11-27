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


