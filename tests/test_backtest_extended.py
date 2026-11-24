"""More properties of the rebalancing engine."""
import numpy as np
import pandas as pd
import pytest

from mpt.backtest import (backtest, buy_and_hold, compare_strategies,
                          rebalance_schedule)

def test_weights_sum_to_one_always(prices):
    bt = backtest(prices, [0.4, 0.3, 0.2, 0.1], rebalance="monthly")
    np.testing.assert_allclose(bt.weights_over_time.sum(axis=1).values, 1.0,
                               atol=1e-9)


