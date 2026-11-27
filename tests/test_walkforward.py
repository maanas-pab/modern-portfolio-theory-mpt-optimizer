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


