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


