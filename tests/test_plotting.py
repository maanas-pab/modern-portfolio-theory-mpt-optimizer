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


def test_plot_correlation_vs_vol(tmp_path, prices):
    from mpt.stats import correlation_matrix

    out = str(tmp_path / "cv.png")
    vol = prices.pct_change().std() * np.sqrt(252)
    P.plot_correlation_vs_vol(correlation_matrix(prices), vol, out_path=out)
    assert (tmp_path / "cv.png").stat().st_size > 5_000


def test_plot_weights_comparison(tmp_path):
    tables = {
        "Max Sharpe": weights_table([0.7, 0.3], ["A", "B"], "Max Sharpe"),
        "Min Vol": weights_table([0.4, 0.6], ["A", "B"], "Min Vol"),
    }
    out = str(tmp_path / "w.png")
    P.plot_weights_comparison(tables, out_path=out)
    assert (tmp_path / "w.png").stat().st_size > 5_000


def test_plot_weights_pies(tmp_path):
    tables = {"P1": weights_table([0.5, 0.5], ["A", "B"], "P1")}
    out = str(tmp_path / "p.png")
    P.plot_weights_pies(tables, out_path=out)
    assert (tmp_path / "p.png").stat().st_size > 5_000


def test_plot_risk_contribution(tmp_path, stats):
    _, exp, cov = stats
    w = np.array([0.5, 0.3, 0.1, 0.1])
    out = str(tmp_path / "rc.png")
    P.plot_risk_contribution(w, cov, list(exp.index), out_path=out)
    assert (tmp_path / "rc.png").stat().st_size > 5_000


def test_plot_cumulative_returns(tmp_path, prices):
    rets = prices.pct_change().dropna()
    series = {"A": rets.iloc[:, 0], "B": rets.iloc[:, 1]}
    out = str(tmp_path / "cum.png")
    P.plot_cumulative_returns(series, out_path=out)
    assert (tmp_path / "cum.png").stat().st_size > 5_000


def test_plot_drawdown(tmp_path, prices):
    rets = prices.pct_change().dropna()
    out = str(tmp_path / "dd.png")
    P.plot_drawdown({"A": rets.iloc[:, 0]}, out_path=out)
    assert (tmp_path / "dd.png").stat().st_size > 5_000


def test_plot_rolling_volatility(tmp_path, prices):
    rets = prices.pct_change().dropna()
    out = str(tmp_path / "rv.png")
    P.plot_rolling_volatility({"A": rets.iloc[:, 0]}, window=21, out_path=out)
    assert (tmp_path / "rv.png").stat().st_size > 5_000


def test_plot_rolling_sharpe(tmp_path, prices):
    rets = prices.pct_change().dropna()
    out = str(tmp_path / "rs.png")
    P.plot_rolling_sharpe({"A": rets.iloc[:, 0]}, window=21, out_path=out)
    assert (tmp_path / "rs.png").stat().st_size > 5_000


def test_plot_var_cvar(tmp_path, prices):
    rets = prices.pct_change().mean(axis=1).dropna()
    out = str(tmp_path / "var.png")
    P.plot_var_cvar(rets, out_path=out)
    assert (tmp_path / "var.png").stat().st_size > 5_000


def test_plot_sharpe_histogram(tmp_path):
    out = str(tmp_path / "sh.png")
    P.plot_sharpe_histogram(np.random.default_rng(0).normal(0.5, 0.2, 500), 1.2,
                            out_path=out)
    assert (tmp_path / "sh.png").stat().st_size > 5_000


def test_plot_metric_comparison(tmp_path):
    raw = pd.DataFrame({"A": {"annual_return": 0.1, "annual_vol": 0.2,
                              "max_drawdown": -0.15},
                        "B": {"annual_return": 0.2, "annual_vol": 0.3,
                              "max_drawdown": -0.25}})
    out = str(tmp_path / "mc.png")
    P.plot_metric_comparison(raw, out_path=out)
    assert (tmp_path / "mc.png").stat().st_size > 5_000


def test_plot_asset_risk_return(tmp_path, stats):
    _, exp, cov = stats
    vol = pd.Series(np.sqrt(np.diag(cov.values)), index=exp.index)
    out = str(tmp_path / "ar.png")
    P.plot_asset_risk_return(exp, vol, out_path=out)
    assert (tmp_path / "ar.png").stat().st_size > 5_000


def test_plot_monthly_heatmap(tmp_path, prices):
    rets = prices.pct_change().mean(axis=1).dropna()
    out = str(tmp_path / "m.png")
    P.plot_monthly_heatmap(rets, out_path=out)
    assert (tmp_path / "m.png").stat().st_size > 5_000


