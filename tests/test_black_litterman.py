"""Tests for mpt.black_litterman: priors, views, posteriors."""
import numpy as np
import pandas as pd
import pytest

from mpt.black_litterman import (DEFAULT_DELTA, black_litterman,
                                 black_litterman_covariance,
                                 equilibrium_returns, top_pick_views)

def test_equilibrium_matches_formula(two_asset):
    _, cov = two_asset
    pi = equilibrium_returns(cov, market_weights=[1.0, 0.0],
                             risk_aversion=DEFAULT_DELTA)
    expected = DEFAULT_DELTA * cov.values @ np.array([1.0, 0.0])
    np.testing.assert_allclose(pi.values, expected)


