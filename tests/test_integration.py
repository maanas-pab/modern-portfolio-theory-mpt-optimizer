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


