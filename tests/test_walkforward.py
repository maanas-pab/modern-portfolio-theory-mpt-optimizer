"""Tests for mpt.walkforward: windows, simulation, summaries."""
import numpy as np
import pandas as pd
import pytest

from mpt.walkforward import (WalkForwardResult, rolling_windows,
                             walk_forward, window_weight_frame)

def test_window_count(prices):
    idx = prices.index
    n, train, test = len(idx), 100, 25
    wins = rolling_windows(idx, train, test)
    expected = (n - train - test) // test + 1
    assert len(wins) == expected


def test_windows_slide_by_test_days(prices):
    wins = rolling_windows(prices.index, 100, 25)
    starts = [w[2] for w in wins[:-1]]
    ends = [w[2] for w in wins[1:]]
    # consecutive test ends are exactly test_days apart
    gap = prices.index.get_loc(ends[0]) - prices.index.get_loc(starts[0])
    assert gap == 25


def test_window_ordering(prices):
    for train_start, train_end, test_end in rolling_windows(prices.index, 120, 30):
        assert train_start <= train_end < test_end


def test_window_validation(prices):
    idx = prices.index
    with pytest.raises(ValueError, match="train_days"):
        rolling_windows(idx, 1, 10)
    with pytest.raises(ValueError, match="test_days"):
        rolling_windows(idx, 10, 0)
    with pytest.raises(ValueError, match="exceeds"):
        rolling_windows(idx, len(idx), 10)


def test_walk_forward_too_short(prices):
    with pytest.raises(ValueError, match="observations"):
        walk_forward(prices, train_days=len(prices), test_days=10)


def test_walk_forward_alignment(prices):
    wf = walk_forward(prices, train_days=150, test_days=40)
    assert isinstance(wf, WalkForwardResult)
    assert len(wf.equity) == len(wf.returns)
    assert wf.n_windows >= 1
    assert wf.returns.index.is_monotonic_increasing
    assert not wf.returns.index.has_duplicates


def test_walk_forward_equity_positive(prices):
    wf = walk_forward(prices, train_days=150, test_days=40)
    assert (wf.equity > 0).all()


def test_walk_forward_total_return(prices):
    wf = walk_forward(prices, train_days=150, test_days=40)
    assert np.isclose(wf.total_return, wf.equity.iloc[-1] - 1.0)


