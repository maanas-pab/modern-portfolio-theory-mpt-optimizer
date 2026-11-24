"""
Command-line entry point: parse flags, run the pipeline,
print the report, export CSVs and draw every chart.
"""
import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import backtest, buy_and_hold
from .config import BENCHMARK, DEFAULT_TICKERS, SECTORS, OptimizerConfig
from .data import get_prices
from .frontier import efficient_frontier, frontier_key_points, random_portfolios
from .metrics import concentration_hhi, portfolio_performance
from .optimizer import max_sharpe_weights, min_vol_weights, risk_parity_weights
from .report import (export_weights, performance_summary, print_report,
                     summary_table, weights_table)
from .stats import calculate_statistics, covariance_matrix

def build_parser() -> argparse.ArgumentParser:
    """All CLI flags in one place (used by tests and docs)."""
    ap = argparse.ArgumentParser(
        prog="mpt",
        description="MPT optimizer: efficient frontier, max-Sharpe and min-vol portfolios")
    # data window
    ap.add_argument("--period", default="3y", help="history window (1y, 3y, 5y, max)")
    ap.add_argument("--interval", default="1d", help="price interval (1d, 1wk, ...)")
    ap.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS),
                    help="space-separated universe")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, refetch")
    ap.add_argument("--cache", default="data/prices_3y.csv", dest="cache_path",
                    help="price cache CSV path")
    # model
    ap.add_argument("--risk-free", type=float, default=0.02, dest="risk_free")
    ap.add_argument("--points", type=int, default=60, help="frontier resolution")
    ap.add_argument("--random", type=int, default=5000, help="Monte-Carlo portfolios")
    ap.add_argument("--shrinkage", action="store_true",
                    help="use Ledoit-Wolf shrunk covariance")
    ap.add_argument("--benchmark", default=BENCHMARK)
    ap.add_argument("--black-litterman", action="store_true", dest="black_litterman",
                    help="add a Black-Litterman max-Sharpe strategy")
    # backtest
    ap.add_argument("--rebalance", default="none",
                    choices=["none", "weekly", "monthly", "quarterly"])
    ap.add_argument("--cost", type=float, default=0.0,
                    help="one-way transaction cost per unit turnover")
    # out-of-sample
    ap.add_argument("--walk-forward", action="store_true", dest="walk_forward",
                    help="rolling train/test evaluation (no look-ahead)")
    ap.add_argument("--train", type=int, default=252, dest="train_days",
                    help="walk-forward training window (days)")
    ap.add_argument("--test", type=int, default=63, dest="test_days",
                    help="walk-forward test window (days)")
    # output
    ap.add_argument("--out-dir", default="outputs", dest="out_dir")
    ap.add_argument("--figure-dir", default="figures", dest="figure_dir")
    ap.add_argument("--no-plot", action="store_true", dest="no_plot")
    ap.add_argument("--no-backtest", action="store_true", dest="no_backtest")
    return ap


