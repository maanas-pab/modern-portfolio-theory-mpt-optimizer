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


def test_rebalance_dates_are_trading_days(prices):
    bt = backtest(prices, [0.25] * 4, rebalance="weekly")
    for d in bt.rebalance_dates:
        assert d in prices.index


def test_turnover_bounds(prices):
    bt = backtest(prices, [0.5, 0.3, 0.1, 0.1], rebalance="monthly")
    assert (bt.turnover >= 0).all()
    assert (bt.turnover <= 2.0).all()


def test_equity_longer_than_returns(prices):
    bt = backtest(prices, [0.25] * 4, rebalance="none")
    assert len(bt.equity) == len(bt.returns) + 1


