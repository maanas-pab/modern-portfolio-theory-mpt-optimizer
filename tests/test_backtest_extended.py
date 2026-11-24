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


def test_flat_prices_flat_equity():
    flat = pd.DataFrame({"A": [100.0] * 30, "B": [50.0] * 30},
                        index=pd.bdate_range("2024-01-01", periods=30))
    bt = backtest(flat, [0.5, 0.5], rebalance="monthly")
    np.testing.assert_allclose(bt.equity.values, 1.0, atol=1e-12)


def test_single_asset_backtest():
    rng = np.random.default_rng(0)
    px = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 40))},
                      index=pd.bdate_range("2024-01-01", periods=40))
    bt = backtest(px, [1.0], rebalance="none")
    direct = px["A"].pct_change().dropna()
    # equity compounds from day 2, so reported returns start one day later
    np.testing.assert_allclose(bt.returns.values, direct.values[1:], atol=1e-12)


def test_cost_monotonicity(prices):
    w = [0.4, 0.3, 0.2, 0.1]
    eq = [backtest(prices, w, rebalance="quarterly", transaction_cost=c).equity.iloc[-1]
          for c in (0.0, 0.005, 0.02)]
    assert eq[0] >= eq[1] >= eq[2]


def test_start_before_history(prices):
    bt = backtest(prices, [0.25] * 4, rebalance="none", start="2000-01-01")
    assert len(bt.returns) == len(prices) - 2


