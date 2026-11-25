"""Tests for mpt.frontier: tracing, sampling, anchors, CML."""
import numpy as np

from mpt.frontier import (capital_market_line, efficient_frontier,
                          frontier_key_points, random_portfolios)

def test_frontier_alignment(stats):
    _, exp, cov = stats
    rets, vols, weights = efficient_frontier(exp, cov, n_points=25)
    assert len(rets) == len(vols) == len(weights)
    assert len(rets) >= 20


def test_frontier_positive_vol(stats):
    _, exp, cov = stats
    _, vols, _ = efficient_frontier(exp, cov, n_points=15)
    assert (vols > 0).all()


def test_frontier_monotone_above_min_vol(stats):
    _, exp, cov = stats
    rets, vols, _ = efficient_frontier(exp, cov, n_points=40)
    i_min = int(np.argmin(vols))
    upper_rets = rets[i_min:]
    assert (np.diff(upper_rets) >= -1e-6).all()
    assert (np.diff(vols[i_min:]) >= -1e-6).all()


def test_random_portfolios_shape(stats):
    _, exp, cov = stats
    r, v, s = random_portfolios(exp, cov, n=300, seed=1)
    assert r.shape == v.shape == s.shape == (300,)
    assert np.isfinite(r).all() and np.isfinite(v).all()


def test_random_seed_reproducible(stats):
    _, exp, cov = stats
    a = random_portfolios(exp, cov, n=50, seed=42)
    b = random_portfolios(exp, cov, n=50, seed=42)
    np.testing.assert_allclose(a[0], b[0])


def test_random_long_only(stats):
    _, exp, cov = stats
    _, vols, _ = random_portfolios(exp, cov, n=200, seed=2)
    max_asset_vol = np.sqrt(np.diag(cov.values)).max()
    assert (vols > 0).all()
    assert vols.max() <= max_asset_vol + 1e-9


def test_key_points_present(stats):
    _, exp, cov = stats
    pts = frontier_key_points(exp, cov, 0.02)
    assert set(pts) == {"max_sharpe", "min_vol", "equal"}
    for p in pts.values():
        assert {"weights", "return", "vol", "sharpe"} <= set(p)


