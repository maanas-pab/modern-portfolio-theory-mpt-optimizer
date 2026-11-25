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


def test_equilibrium_equal_weight_default(two_asset):
    _, cov = two_asset
    pi = equilibrium_returns(cov, risk_aversion=2.0)
    expected = 2.0 * cov.values @ np.array([0.5, 0.5])
    np.testing.assert_allclose(pi.values, expected)


def test_equilibrium_weight_length(two_asset):
    _, cov = two_asset
    with pytest.raises(ValueError, match="market_weights"):
        equilibrium_returns(cov, market_weights=[1.0, 0.5, 0.2])


def test_top_pick_views_shape():
    exp = pd.Series({"A": 0.05, "B": 0.20, "C": 0.10, "D": 0.15})
    P, Q = top_pick_views(exp, k=2, uplift=0.07)
    assert P.shape == (2, 4)
    assert Q.shape == (2,)
    np.testing.assert_allclose(P.sum(axis=1), 0.0, atol=1e-12)
    assert (Q == 0.07).all()


def test_top_pick_views_selects_top():
    exp = pd.Series({"A": 0.05, "B": 0.20, "C": 0.10, "D": 0.15})
    P, _ = top_pick_views(exp, k=1)
    # the single view longs B (the largest) against the rest
    assert P[0, list(exp.index).index("B")] == 1.0


def test_top_pick_views_bounds():
    exp = pd.Series({"A": 0.05, "B": 0.10})
    with pytest.raises(ValueError, match="k"):
        top_pick_views(exp, k=0)
    with pytest.raises(ValueError, match="k"):
        top_pick_views(exp, k=5)


def test_posterior_near_prior_with_diffuse_views(two_asset):
    exp, cov = two_asset
    pi = equilibrium_returns(cov)
    P = np.array([[1.0, -1.0]])
    Q = np.array([0.05])
    mu = black_litterman(pi, cov, P, Q, tau=0.05, omega_scalar=10.0)
    np.testing.assert_allclose(mu.values, pi.values, atol=0.02)


def test_confident_view_pulls_posterior(two_asset):
    exp, cov = two_asset
    pi = equilibrium_returns(cov)
    P = np.array([[1.0, -1.0]])   # A outperforms B
    Q = np.array([0.20])          # by 20 points - confident
    mu = black_litterman(pi, cov, P, Q, tau=0.05, omega_scalar=0.001)
    assert mu["A"] - mu["B"] > (pi["A"] - pi["B"]) + 0.05


