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


def emit_report(result: dict) -> None:
    """Print the console report and write weights CSVs."""
    cfg: OptimizerConfig = result["cfg"]
    tickers = list(result["exp_returns"].index)
    print("\n--- Annualized Expected Returns ---")
    print((result["exp_returns"] * 100).round(2).astype(str) + " %")
    print("\n--- Covariance (first 5x5) ---")
    print(result["cov"].iloc[:5, :5].round(4))

    for name, block in result["summaries"].items():
        print_report(name, block["summary"], block["weights"])
        path = export_weights(block["weights"]["weight"].values, tickers,
                              cfg.out_dir, name.lower().replace(" ", "_"))
        print(f"Wrote {path}")

    eq = result["summaries"]["Equal Weight"]["summary"]
    print(f"\nHHI concentration - Max Sharpe "
          f"{result['summaries']['Max Sharpe']['hhi']:.3f} vs "
          f"Equal {result['summaries']['Equal Weight']['hhi']:.3f}")


def emit_backtests(result: dict) -> None:
    """Print one-line stats for each backtest."""
    cfg: OptimizerConfig = result["cfg"]
    if not result["backtests"]:
        return
    print(f"\n--- Backtest (rebalance={cfg.rebalance}, cost={cfg.transaction_cost:.3%}) ---")
    rows = {}
    for name, bt in result["backtests"].items():
        s = performance_summary(bt.returns, cfg.risk_free)
        s["total_return"] = bt.total_return
        s["n_rebalances"] = len(bt.rebalance_dates)
        s["total_cost"] = bt.total_cost
        rows[name] = s
    df = summary_table(pd.DataFrame(rows))
    print(df.loc[[m for m in ("total_return", "annual_return", "sharpe",
                              "max_drawdown", "calmar", "n_rebalances", "total_cost")
                  if m in df.index]].to_string())


def emit_walkforward(result: dict) -> None:
    """Print the honest out-of-sample summary when --walk-forward is on."""
    cfg: OptimizerConfig = result["cfg"]
    wf = result.get("walkforward")
    if wf is None:
        return
    s = wf.summary(cfg.risk_free)
    print(f"\n--- Walk-forward (train={cfg.train_days}d, "
          f"test={cfg.test_days}d, {wf.n_windows} windows) ---")
    print(f"OOS annual return {s['annual_return']:.2%} | "
          f"vol {s['annual_vol']:.2%} | Sharpe {s['sharpe']:.3f} | "
          f"MaxDD {s['max_drawdown']:.2%}")
    in_sample = result["summaries"]["Max Sharpe"]["summary"]
    print(f"In-sample (look-ahead) Sharpe {in_sample['sharpe']:.3f} "
          f"- the gap is the cost of overfitting")


def emit_figures(result: dict) -> list:
    """Draw every chart in the gallery; returns the list of paths."""
    from . import plotting as P

    cfg: OptimizerConfig = result["cfg"]
    if cfg.no_plot:
        return []
    os.makedirs(cfg.figure_dir, exist_ok=True)
    f = lambda name: os.path.join(cfg.figure_dir, name)  # noqa: E731

    f_rets, f_vols = result["frontier"]
    rand_r, rand_v, rand_s = result["random"]
    pts = result["points"]
    exp_ret, cov, daily = result["exp_returns"], result["cov"], result["daily"]
    asset_vol = np.sqrt(np.diag(cov.values))
    bench = result["bench"]
    bench_rets = bench.pct_change().dropna() if bench is not None else None

    # frontier family
    P.plot_frontier(
        f_rets, f_vols, rand_r, rand_v, rand_s,
        max_sharpe_pt=(pts["max_sharpe"]["return"], pts["max_sharpe"]["vol"]),
        min_vol_pt=(pts["min_vol"]["return"], pts["min_vol"]["vol"]),
        asset_rets=exp_ret.values, asset_vols=asset_vol,
        asset_labels=list(exp_ret.index), risk_free=cfg.risk_free,
        out_path=f("efficient_frontier.png"))
    P.plot_frontier_clean(f_rets, f_vols, pts, out_path=f("frontier_clean.png"))
    P.plot_sharpe_histogram(rand_s, pts["max_sharpe"]["sharpe"],
                            out_path=f("sharpe_distribution.png"))

    # risk structure
    P.plot_correlation_heatmap(daily.corr(), out_path=f("correlation_heatmap.png"))
    P.plot_covariance_heatmap(cov, out_path=f("covariance_heatmap.png"))
    P.plot_correlation_vs_vol(daily.corr(), pd.Series(asset_vol, index=exp_ret.index),
                              out_path=f("correlation_vs_vol.png"))
    P.plot_asset_risk_return(exp_ret, pd.Series(asset_vol, index=exp_ret.index),
                             out_path=f("asset_risk_return.png"))
    P.plot_return_distributions(daily, out_path=f("return_distributions.png"))
    P.plot_monthly_heatmap(result["summaries"]["Max Sharpe"]["returns"],
                           title="Max Sharpe", out_path=f("monthly_returns.png"))
    P.plot_asset_monthly_heatmap(daily, out_path=f("asset_monthly_heatmap.png"))

    # allocations
    tables = {name: result["summaries"][name]["weights"]
              for name in ("Max Sharpe", "Min Volatility", "Equal Weight")}
    P.plot_weights_comparison(tables, out_path=f("weights_comparison.png"))
    P.plot_weights_pies(tables, out_path=f("weights_pies.png"))
    w_sharpe = result["weights"]["Max Sharpe"]
    P.plot_risk_contribution(w_sharpe, cov, list(exp_ret.index),
                             name="Max Sharpe", out_path=f("risk_contribution.png"))
    P.plot_sector_exposure(w_sharpe, list(exp_ret.index), SECTORS,
                           out_path=f("sector_exposure.png"))

    # performance over time
    series = {name: block["returns"] for name, block in result["summaries"].items()}
    P.plot_cumulative_returns(series, benchmark=bench_rets,
                              out_path=f("cumulative_returns.png"))
    P.plot_drawdown(series, out_path=f("drawdown.png"))
    P.plot_rolling_volatility(series, out_path=f("rolling_volatility.png"))
    P.plot_rolling_sharpe(series, risk_free=cfg.risk_free,
                          out_path=f("rolling_sharpe.png"))
    P.plot_var_cvar(result["summaries"]["Max Sharpe"]["returns"],
                    out_path=f("var_cvar.png"))

    # scoreboard + backtest weights
    raw = pd.DataFrame({name: block["summary"]
                        for name, block in result["summaries"].items()})
    P.plot_metric_comparison(raw, out_path=f("metric_comparison.png"))
    if result["backtests"]:
        bt = result["backtests"].get("Max Sharpe")
        if bt is not None and len(bt.rebalance_dates):
            P.plot_weights_over_time(bt, out_path=f("weights_over_time.png"))
            P.plot_concentration_over_time(bt,
                                           out_path=f("concentration_over_time.png"))
            P.plot_weight_trajectories(bt,
                                       out_path=f("weight_trajectories.png"))
            P.plot_turnover(bt, out_path=f("turnover.png"))

    # benchmark relationship
    if bench_rets is not None:
        ms = result["summaries"]["Max Sharpe"]["returns"]
        P.plot_rolling_beta(ms, bench_rets, out_path=f("rolling_beta.png"))
        P.plot_benchmark_scatter(ms, bench_rets,
                                 out_path=f("benchmark_scatter.png"))

    # calendar views
    P.plot_quarterly_returns(result["summaries"]["Max Sharpe"]["returns"],
                             out_path=f("quarterly_returns.png"))
    P.plot_annual_returns(series, out_path=f("annual_returns.png"))

    # covariance-shrinkage sensitivity (always both, whatever --shrinkage says)
    cov_shrunk = covariance_matrix(daily, shrinkage=True)
    samp_rets, samp_vols, _ = efficient_frontier(exp_ret, result["cov_sample"],
                                                 cfg.frontier_points)
    shr_rets, shr_vols, _ = efficient_frontier(exp_ret, cov_shrunk,
                                               cfg.frontier_points)
    P.plot_frontier_shrinkage(samp_rets, samp_vols, shr_rets, shr_vols,
                              out_path=f("frontier_shrinkage.png"))

    # out-of-sample equity
    if result.get("walkforward") is not None:
        P.plot_walk_forward(result["walkforward"],
                            out_path=f("walk_forward.png"))

    return sorted(os.path.join(cfg.figure_dir, n)
                  for n in os.listdir(cfg.figure_dir) if n.endswith(".png"))


def main(argv=None) -> int:
    """Parse args, run the pipeline, emit everything. Returns exit code."""
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    result = run(cfg)
    emit_report(result)
    emit_backtests(result)
    emit_walkforward(result)
    paths = emit_figures(result)
    if paths:
        print(f"\n{len(paths)} charts in {cfg.figure_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
