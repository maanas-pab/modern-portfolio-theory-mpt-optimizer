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


