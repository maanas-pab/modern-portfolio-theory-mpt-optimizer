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


