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


