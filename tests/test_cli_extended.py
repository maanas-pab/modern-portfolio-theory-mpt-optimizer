"""Flag handling and pipeline-guard coverage for mpt.cli."""
import pytest

from mpt.cli import build_parser, config_from_args, emit_backtests, run
from mpt.config import OptimizerConfig

@pytest.mark.parametrize("flag,extra", [
    ("--period", ["3y"]), ("--interval", ["1d"]), ("--tickers", ["AAPL"]),
    ("--refresh", []), ("--cache", ["/tmp/x.csv"]),
    ("--risk-free", ["0.03"]), ("--points", ["10"]), ("--random", ["25"]),
    ("--shrinkage", []), ("--benchmark", ["QQQ"]),
    ("--black-litterman", []), ("--rebalance", ["monthly"]),
    ("--cost", ["0.001"]), ("--walk-forward", []),
    ("--train", ["180"]), ("--test", ["30"]),
    ("--out-dir", ["/tmp/o"]), ("--figure-dir", ["/tmp/f"]),
    ("--no-plot", []), ("--no-backtest", []),
])
def test_flag_exists(flag, extra):
    args = build_parser().parse_args([flag] + extra)
    assert args is not None


def test_unknown_flag_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--definitely-not-a-flag"])


def test_benchmark_flag():
    args = build_parser().parse_args(["--benchmark", "QQQ"])
    assert config_from_args(args).benchmark == "QQQ"


def test_ticker_flag():
    args = build_parser().parse_args(["--tickers", "AAPL", "MSFT"])
    cfg = config_from_args(args)
    assert cfg.tickers == ["AAPL", "MSFT"]


def test_ticker_flag_empty_fallback():
    args = build_parser().parse_args(["--tickers"])
    cfg = config_from_args(args)
    assert len(cfg.tickers) == 12


def test_output_flags():
    args = build_parser().parse_args(["--no-plot", "--no-backtest"])
    cfg = config_from_args(args)
    assert cfg.no_plot and cfg.no_backtest


def test_emit_backtests_noop(capsys):
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=50,
                          no_plot=True, no_backtest=True).validate()
    emit_backtests(run(cfg))
    assert "Backtest" not in capsys.readouterr().out


def test_emit_backtests_prints(capsys):
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=50,
                          no_plot=True, rebalance="monthly").validate()
    emit_backtests(run(cfg))
    out = capsys.readouterr().out
    assert "Backtest" in out
    assert "n_rebalances" in out
    assert "total_cost" in out


def test_run_no_shrinkage_attrs():
    cfg = OptimizerConfig(frontier_points=8, random_portfolios=50,
                          no_plot=True, no_backtest=True).validate()
    result = run(cfg)
    assert "shrinkage_intensity" not in result["cov"].attrs


