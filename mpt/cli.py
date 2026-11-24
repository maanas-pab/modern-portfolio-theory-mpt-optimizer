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


def config_from_args(args: argparse.Namespace) -> OptimizerConfig:
    """Translate parsed flags into a validated OptimizerConfig."""
    cfg = OptimizerConfig(
        tickers=list(args.tickers) or list(DEFAULT_TICKERS),
        period=args.period,
        interval=args.interval,
        risk_free=args.risk_free,
        shrinkage=args.shrinkage,
        benchmark=args.benchmark,
        black_litterman=args.black_litterman,
        frontier_points=args.points,
        random_portfolios=args.random,
        rebalance=args.rebalance,
        transaction_cost=args.cost,
        walk_forward=args.walk_forward,
        train_days=args.train_days,
        test_days=args.test_days,
        out_dir=args.out_dir,
        figure_dir=args.figure_dir,
        no_plot=args.no_plot,
        no_backtest=args.no_backtest,
        cache_path=args.cache_path,
        refresh=args.refresh,
    )
    return cfg.validate()


def build_portfolios(exp_returns: pd.Series, cov: pd.DataFrame, cfg: OptimizerConfig) -> dict:
    """Optimize every named strategy used downstream."""
    rf = cfg.risk_free
    weights = {
        "Max Sharpe": max_sharpe_weights(exp_returns, cov, rf),
        "Min Volatility": min_vol_weights(exp_returns, cov),
        "Risk Parity": risk_parity_weights(cov),
        "Equal Weight": np.repeat(1.0 / len(exp_returns), len(exp_returns)),
    }
    return weights


def summarize_all(named_weights: dict, daily: pd.DataFrame,
                  exp_returns: pd.Series, cov: pd.DataFrame,
                  cfg: OptimizerConfig,
                  benchmark: Optional[pd.Series] = None) -> dict:
    """performance_summary for each strategy (and its weight table)."""
    out = {}
    tickers = list(exp_returns.index)
    for name, w in named_weights.items():
        rets = pd.Series(
            (daily.values @ np.asarray(w, dtype=float)), index=daily.index, name=name)
        out[name] = {
            "weights": weights_table(w, tickers, name),
            "summary": performance_summary(rets, cfg.risk_free, benchmark),
            "hhi": concentration_hhi(w),
            "returns": rets,
        }
    return out


def run(cfg: OptimizerConfig) -> dict:
    """
    Execute the whole pipeline and return every artifact:

    prices, daily, exp_returns, cov, weights, points, frontier,
    random cloud, summaries, backtests.
    """
    wanted = list(dict.fromkeys(list(cfg.tickers) + [cfg.benchmark]))
    prices = get_prices(wanted, period=cfg.period, interval=cfg.interval,
                        cache_path=cfg.cache_path, refresh=cfg.refresh)
    tickers = [t for t in prices.columns if t != cfg.benchmark]
    bench = prices[cfg.benchmark] if cfg.benchmark in prices.columns else None
    assets = prices[tickers]

    daily, exp_returns, cov_sample = calculate_statistics(assets)
    cov = covariance_matrix(daily, shrinkage=cfg.shrinkage)
    if cfg.shrinkage:
        lam = cov.attrs.get("shrinkage_intensity", float("nan"))
        print(f"Ledoit-Wolf shrinkage intensity: {lam:.3f}")

    named_weights = build_portfolios(exp_returns, cov, cfg)
    if cfg.black_litterman:
        from .black_litterman import (black_litterman as bl_posterior,
                                      equilibrium_returns, top_pick_views)

        pi = equilibrium_returns(cov)
        P, Q = top_pick_views(exp_returns, k=3, uplift=0.05)
        mu_bl = bl_posterior(pi, cov, P, Q, tau=0.05)
        named_weights["BL Max Sharpe"] = max_sharpe_weights(mu_bl, cov, cfg.risk_free)
        result_bl = mu_bl
    else:
        result_bl = None
    points = frontier_key_points(exp_returns, cov, cfg.risk_free)
    f_rets, f_vols, _ = efficient_frontier(exp_returns, cov, cfg.frontier_points)
    rand_r, rand_v, rand_s = random_portfolios(
        exp_returns, cov, cfg.risk_free, n=cfg.random_portfolios)

    bench_rets = bench.pct_change().dropna() if bench is not None else None
    summaries = summarize_all(named_weights, daily, exp_returns, cov, cfg, bench_rets)

    backtests = {}
    if not cfg.no_backtest:
        for name, w in named_weights.items():
            backtests[name] = backtest(assets, w, rebalance=cfg.rebalance,
                                       transaction_cost=cfg.transaction_cost)
        backtests["Buy & Hold (equal)"] = buy_and_hold(assets)

    wf = None
    if cfg.walk_forward:
        from .walkforward import walk_forward as _walk_forward

        wf = _walk_forward(assets, train_days=cfg.train_days,
                           test_days=cfg.test_days, objective="sharpe",
                           risk_free=cfg.risk_free)

    return {
        "prices": prices, "assets": assets, "daily": daily, "bench": bench,
        "exp_returns": exp_returns, "cov": cov, "cov_sample": cov_sample,
        "weights": named_weights, "points": points,
        "frontier": (f_rets, f_vols), "random": (rand_r, rand_v, rand_s),
        "summaries": summaries, "backtests": backtests, "cfg": cfg,
        "bl_returns": result_bl, "walkforward": wf,
    }


