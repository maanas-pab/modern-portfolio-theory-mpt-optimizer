"""End-to-end tests wiring config -> run -> report -> figures."""
import numpy as np
import pandas as pd
import pytest

from mpt.cli import (build_parser, config_from_args, emit_figures,
                     emit_walkforward, run)
from mpt.config import OptimizerConfig

def test_run_result_keys():
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=100,
                          no_plot=True, no_backtest=True).validate()
    result = run(cfg)
    for key in ("prices", "assets", "daily", "bench", "exp_returns", "cov",
                "weights", "points", "frontier", "random", "summaries",
                "backtests", "cfg", "bl_returns", "walkforward"):
        assert key in result


def test_run_defaults_have_no_extras():
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=100,
                          no_plot=True, no_backtest=True).validate()
    result = run(cfg)
    assert result["bl_returns"] is None
    assert result["walkforward"] is None


def test_run_includes_benchmark():
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=100,
                          no_plot=True, no_backtest=True).validate()
    result = run(cfg)
    assert result["bench"] is not None
    assert "beta" in result["summaries"]["Max Sharpe"]["summary"]


def test_run_black_litterman_strategy():
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=100,
                          no_plot=True, no_backtest=True,
                          black_litterman=True).validate()
    result = run(cfg)
    assert "BL Max Sharpe" in result["weights"]
    assert result["bl_returns"] is not None
    assert np.isclose(result["weights"]["BL Max Sharpe"].sum(), 1.0)


def test_run_walk_forward_flag():
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=100,
                          no_plot=True, no_backtest=True,
                          walk_forward=True, train_days=200,
                          test_days=50).validate()
    result = run(cfg)
    wf = result["walkforward"]
    assert wf is not None and wf.n_windows >= 1


def test_all_weights_normalized():
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=100,
                          no_plot=True, no_backtest=True,
                          black_litterman=True).validate()
    result = run(cfg)
    for name, w in result["weights"].items():
        assert np.isclose(w.sum(), 1.0, atol=1e-5), name


def test_emit_walkforward_noop(capsys):
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=100,
                          no_plot=True, no_backtest=True).validate()
    emit_walkforward(run(cfg))
    assert "Walk-forward" not in capsys.readouterr().out


def test_emit_walkforward_prints(capsys):
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=100,
                          no_plot=True, no_backtest=True,
                          walk_forward=True, train_days=200,
                          test_days=50).validate()
    emit_walkforward(run(cfg))
    out = capsys.readouterr().out
    assert "Walk-forward" in out
    assert "OOS annual return" in out
    assert "look-ahead" in out


def test_full_gallery(tmp_path):
    cfg = OptimizerConfig(
        frontier_points=10, random_portfolios=200,
        figure_dir=str(tmp_path / "figs"),
        rebalance="monthly", transaction_cost=0.001,
        walk_forward=True, train_days=200, test_days=50,
        black_litterman=True,
    ).validate()
    result = run(cfg)
    paths = emit_figures(result)
    pngs = [p for p in paths if p.endswith(".png")]
    assert len(pngs) == 30
    for p in pngs:
        assert (tmp_path / "figs" / p.split("/")[-1]).stat().st_size > 3000


