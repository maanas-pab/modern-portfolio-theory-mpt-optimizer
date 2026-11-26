"""More coverage for mpt.report tables and summaries."""
import numpy as np
import pandas as pd

from mpt.report import (export_weights, performance_summary, print_report,
                        summary_table, weights_table)

def test_weights_table_single():
    df = weights_table([1.0], ["ONLY"], "P")
    assert len(df) == 1
    assert df["weight"].iloc[0] == 1.0
    assert df["weight_pct"].iloc[0] == 100.0


def test_weights_table_alignment():
    df = weights_table([0.1, 0.9], ["AAA", "BBB"], "P")
    lookup = dict(zip(df["ticker"], df["weight"]))
    assert lookup["AAA"] == 0.1
    assert lookup["BBB"] == 0.9


def test_export_creates_dir(tmp_path):
    path = export_weights([1.0], ["A"], str(tmp_path / "deep" / "out"), "x")
    assert (tmp_path / "deep" / "out").exists()
    assert path.endswith(".csv")


def test_summary_positive_constant():
    r = pd.Series([0.001] * 300)
    s = performance_summary(r, 0.0)
    assert s["annual_return"] > 0
    assert s["max_drawdown"] == 0.0
    assert s["sharpe"] == 0.0     # zero vol guard
    assert s["calmar"] == 0.0     # zero drawdown guard


