"""Tests for mpt.report: tables, exports, summaries."""
import numpy as np
import pandas as pd

from mpt.report import (export_weights, performance_summary,
                        print_report, summary_table, weights_table)

def test_weights_table_sorted():
    df = weights_table([0.2, 0.5, 0.3], ["A", "B", "C"], "p")
    assert list(df["ticker"]) == ["B", "C", "A"]


def test_weights_table_pct_sum():
    df = weights_table([0.2, 0.5, 0.3], ["A", "B", "C"], "p")
    assert np.isclose(df["weight"].sum(), 1.0)


def test_weights_table_name_column():
    df = weights_table([1.0], ["A"], "Max Sharpe")
    assert (df["portfolio"] == "Max Sharpe").all()


def test_export_weights_csv(tmp_path):
    path = export_weights([0.6, 0.4], ["A", "B"], str(tmp_path), "x")
    loaded = pd.read_csv(path)
    assert list(loaded["ticker"]) == ["A", "B"]


def test_summary_keys(prices):
    rets = prices.pct_change().mean(axis=1).dropna()
    s = performance_summary(rets, 0.02)
    for key in ("annual_return", "annual_vol", "sharpe", "sortino",
                "max_drawdown", "calmar", "var_95", "cvar_95"):
        assert key in s


def test_summary_with_benchmark(prices, benchmark):
    rets = prices.pct_change().mean(axis=1).dropna()
    s = performance_summary(rets, 0.02, benchmark)
    assert "beta" in s and "alpha" in s


def test_summary_positive_days_bounds(prices):
    rets = prices.pct_change().mean(axis=1).dropna()
    s = performance_summary(rets, 0.0)
    assert 0.0 <= s["positive_days"] <= 1.0


