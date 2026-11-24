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


