"""Tests for mpt.frontier: tracing, sampling, anchors, CML."""
import numpy as np

from mpt.frontier import (capital_market_line, efficient_frontier,
                          frontier_key_points, random_portfolios)

def test_frontier_alignment(stats):
    _, exp, cov = stats
    rets, vols, weights = efficient_frontier(exp, cov, n_points=25)
    assert len(rets) == len(vols) == len(weights)
    assert len(rets) >= 20


