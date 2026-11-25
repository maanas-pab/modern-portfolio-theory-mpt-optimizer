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


