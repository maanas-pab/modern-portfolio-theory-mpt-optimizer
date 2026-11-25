"""Shape and property coverage for mpt.frontier."""
import numpy as np
import pandas as pd
import pytest

from mpt.frontier import (capital_market_line, efficient_frontier,
                          frontier_key_points, random_portfolios)

def test_frontier_weights_normalized(stats):
    _, exp, cov = stats
    _, _, weights = efficient_frontier(exp, cov, n_points=12)
    for w in weights:
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)
        assert w.min() >= -1e-8


def test_frontier_return_range(stats):
    _, exp, cov = stats
    rets, _, _ = efficient_frontier(exp, cov, n_points=30)
    assert rets.min() >= exp.min() - 1e-6
    assert rets.max() <= exp.max() + 1e-6


def test_frontier_single_asset():
    exp = pd.Series({"A": 0.10})
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    rets, vols, _ = efficient_frontier(exp, cov, n_points=5)
    np.testing.assert_allclose(vols, 0.20, atol=1e-8)
    np.testing.assert_allclose(rets, 0.10, atol=1e-8)


def test_frontier_points_bounds(stats):
    _, exp, cov = stats
    with pytest.raises(ValueError, match="n_points"):
        efficient_frontier(exp, cov, n_points=1)


def test_random_zero_risk_free(stats):
    _, exp, cov = stats
    r, v, s = random_portfolios(exp, cov, 0.0, n=50, seed=3)
    assert np.isfinite(s).all()


def test_key_point_equal_weight(stats):
    _, exp, cov = stats
    pts = frontier_key_points(exp, cov, 0.02)
    assert np.isclose(pts["equal"]["return"], exp.mean())
    eq_vol = np.sqrt(np.repeat(1 / len(exp), len(exp)) @ cov.values @
                     np.repeat(1 / len(exp), len(exp)))
    assert np.isclose(pts["equal"]["vol"], eq_vol)


def test_min_vol_beats_equal(stats):
    _, exp, cov = stats
    pts = frontier_key_points(exp, cov, 0.02)
    assert pts["min_vol"]["vol"] <= pts["equal"]["vol"] + 1e-8


def test_cml_extends_past_tangency():
    vols, rets = capital_market_line(0.02, 0.12, 0.20, vol_max=0.4)
    assert vols[-1] == 0.4
    assert rets[-1] > 0.12
