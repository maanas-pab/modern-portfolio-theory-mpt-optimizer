"""Tests for mpt.backtest: schedules, drift, costs, baselines."""
import numpy as np
import pandas as pd
import pytest

from mpt.backtest import (BacktestResult, backtest, buy_and_hold,
                          compare_strategies, rebalance_schedule)

def test_weekly_schedule(prices):
    idx = prices.index
    sched = rebalance_schedule(idx, "weekly")
    assert len(sched) >= 100
    assert sched.is_monotonic_increasing
    assert sched.isin(idx).all()


def test_monthly_schedule(prices):
    idx = prices.index
    sched = rebalance_schedule(idx, "monthly")
    span_months = (idx[-1].year - idx[0].year) * 12 + (idx[-1].month - idx[0].month)
    assert abs(len(sched) - span_months) <= 2
    assert sched.isin(idx).all()


def test_quarterly_schedule(prices):
    idx = prices.index
    sched = rebalance_schedule(idx, "quarterly")
    span_quarters = ((idx[-1].year - idx[0].year) * 12 +
                     (idx[-1].month - idx[0].month)) // 3
    assert abs(len(sched) - span_quarters) <= 1
    assert sched.isin(idx).all()


def test_none_schedule(prices):
    assert len(rebalance_schedule(prices.index, "none")) == 0


def test_bad_schedule_freq(prices):
    with pytest.raises(ValueError, match="freq"):
        rebalance_schedule(prices.index, "yearly-ish")


def test_backtest_equity_starts_at_one(prices):
    eq = backtest(prices, [0.5, 0.5, 0, 0], rebalance="none").equity
    assert np.isclose(eq.iloc[0], 1.0 + prices.pct_change().dropna().iloc[0] @
                      np.array([0.5, 0.5, 0, 0]), atol=1e-9)


def test_backtest_returns_consistent(prices):
    bt = backtest(prices, [0.25] * 4, rebalance="monthly")
    np.testing.assert_allclose(
        bt.returns.values,
        bt.equity.pct_change().dropna().values,
        atol=1e-12)


