"""More coverage for mpt.report tables and summaries."""
import numpy as np
import pandas as pd

from mpt.report import (
    export_weights,
    performance_summary,
    print_report,
    summary_table,
    weights_table,
)


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


def test_summary_benchmark_keys(prices, benchmark):
    r = prices.pct_change().mean(axis=1).dropna()
    s = performance_summary(r, 0.02, benchmark)
    assert {"beta", "alpha"} <= set(s)
    plain = performance_summary(r, 0.02)
    assert "beta" not in plain


def test_summary_table_precision():
    df = summary_table({"P": {"sharpe": 1.23456, "var_95": 0.0312}})
    assert df.loc["sharpe", "P"] == 1.235
    assert df.loc["var_95", "P"] == 3.12


def test_summary_table_row_order():
    df = summary_table({"P": {"a": 1.0, "b": 2.0}})
    assert list(df.index) == ["a", "b"]


def test_print_report_labels(capsys, prices):
    r = prices.pct_change().mean(axis=1).dropna()
    s = performance_summary(r, 0.02)
    print_report("X", s, weights_table([1.0], ["A"], "X"))
    out = capsys.readouterr().out
    for label in ("Return", "Vol", "Sharpe", "MaxDD"):
        assert label in out


def test_summary_positive_days_exact():
    r = pd.Series([0.01, -0.01, 0.02, -0.03, 0.04])
    s = performance_summary(r, 0.0)
    assert np.isclose(s["positive_days"], 3 / 5)
