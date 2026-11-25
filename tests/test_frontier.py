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


