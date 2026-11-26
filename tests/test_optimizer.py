"""Tests for mpt.optimizer: constraints, objectives, edge cases."""
import numpy as np
import pandas as pd
import pytest

from mpt.metrics import portfolio_performance
from mpt.optimizer import (max_sharpe_weights, min_vol_weights,
                           optimize_portfolio, risk_parity_weights)

def test_weights_sum_to_one(stats):
    _, exp, cov = stats
    for w in (max_sharpe_weights(exp, cov, 0.02),
              min_vol_weights(exp, cov),
              risk_parity_weights(cov)):
        assert np.isclose(w.sum(), 1.0, atol=1e-6)


