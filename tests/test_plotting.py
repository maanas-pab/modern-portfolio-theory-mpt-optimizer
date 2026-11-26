"""Every chart function must write a non-empty PNG (Agg backend)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mpt import plotting as P
from mpt.backtest import backtest
from mpt.frontier import (efficient_frontier, frontier_key_points,
                          random_portfolios)
from mpt.report import performance_summary, weights_table, summary_table
from mpt.stats import calculate_statistics


def test_plot_frontier(tmp_path, prices):
    _, exp, cov = calculate_statistics(prices)
    f_rets, f_vols, _ = efficient_frontier(exp, cov, n_points=15)
    r, v, s = random_portfolios(exp, cov, n=100, seed=1)
    pts = frontier_key_points(exp, cov, 0.02)
    out = str(tmp_path / "f.png")
    path = P.plot_frontier(
        f_rets, f_vols, r, v, s,
        max_sharpe_pt=(pts["max_sharpe"]["return"], pts["max_sharpe"]["vol"]),
        min_vol_pt=(pts["min_vol"]["return"], pts["min_vol"]["vol"]),
        asset_rets=exp.values, asset_vols=np.sqrt(np.diag(cov.values)),
        asset_labels=list(exp.index), out_path=out)
    assert path == out
    assert (tmp_path / "f.png").stat().st_size > 10_000


def test_plot_frontier_clean(tmp_path, prices):
    _, exp, cov = calculate_statistics(prices)
    f_rets, f_vols, _ = efficient_frontier(exp, cov, n_points=15)
    pts = frontier_key_points(exp, cov, 0.02)
    out = str(tmp_path / "c.png")
    P.plot_frontier_clean(f_rets, f_vols, pts, out_path=out)
    assert (tmp_path / "c.png").stat().st_size > 5_000


def test_plot_correlation_heatmap(tmp_path, prices):
    from mpt.stats import correlation_matrix

    out = str(tmp_path / "corr.png")
    P.plot_correlation_heatmap(correlation_matrix(prices), out_path=out)
    assert (tmp_path / "corr.png").stat().st_size > 5_000


def test_plot_covariance_heatmap(tmp_path, stats):
    out = str(tmp_path / "cov.png")
    P.plot_covariance_heatmap(stats[2], out_path=out)
    assert (tmp_path / "cov.png").stat().st_size > 5_000


