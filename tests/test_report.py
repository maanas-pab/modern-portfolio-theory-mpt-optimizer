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


