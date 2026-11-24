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


