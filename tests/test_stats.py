"""Tests for mpt.stats: returns, annualization, correlation, shrinkage."""
import numpy as np
import pandas as pd

from mpt.stats import (annualize_covariance, annualize_returns,
                       annualized_volatility, calculate_statistics,
                       covariance_matrix, correlation_matrix, daily_returns,
                       ledoit_wolf_covariance, ledoit_wolf_intensity)

def test_daily_returns_drops_first_row(prices):
    rets = daily_returns(prices)
    assert len(rets) == len(prices) - 1
    assert not rets.isna().any().any()


def test_daily_returns_values():
    s = pd.Series([100.0, 110.0, 99.0])
    r = daily_returns(s.to_frame("X"))["X"].dropna()
    assert np.isclose(r.iloc[0], 0.10)
    assert np.isclose(r.iloc[1], -0.10)


