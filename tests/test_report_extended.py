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


