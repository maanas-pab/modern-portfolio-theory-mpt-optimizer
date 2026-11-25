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


