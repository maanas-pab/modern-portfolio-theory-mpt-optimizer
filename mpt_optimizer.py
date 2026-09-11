"""
Modern Portfolio Theory (MPT) Optimizer
=======================================
Pulls historical prices via yfinance for 10-15 niche stocks,
computes daily returns / expected returns / covariance matrix,
builds the Efficient Frontier with scipy.optimize, and finds
the Max-Sharpe and Min-Volatility portfolios.

Usage:
    python mpt_optimizer.py
    python mpt_optimizer.py --period 3y --risk-free 0.02 --points 60
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

TRADING_DAYS = 252

# 12 niche / thematic growth stocks across clean energy, biotech, cloud, fintech
DEFAULT_TICKERS = [
    "CELH",  # Celsius Holdings (energy drinks)
    "ENPH",  # Enphase Energy (solar microinverters)
    "FSLR",  # First Solar
    "CRSP",  # CRISPR Therapeutics
    "BEAM",  # Beam Therapeutics
    "ROKU",  # Roku
    "ETSY",  # Etsy
    "PLTR",  # Palantir
    "NET",   # Cloudflare
    "DDOG",  # Datadog
    "U",     # Unity Software
    "AFRM",  # Affirm Holdings
]


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def fetch_prices(tickers, period="3y", interval="1d", auto_adjust=True):
    """Download adjusted close prices for all tickers. Returns DataFrame (dates x tickers)."""
    print(f"Downloading {len(tickers)} tickers ({period}, {interval}) ...")
    data = yf.download(
        tickers, period=period, interval=interval,
        auto_adjust=auto_adjust, progress=False, threads=True,
    )
    # yfinance returns MultiIndex columns for multi-ticker; normalize to Close/Adj Close
    if isinstance(data.columns, pd.MultiIndex):
        for level in ("Close", "Adj Close", "Price"):
            if level in data.columns.get_level_values(0):
                data = data[level]
                break
    elif "Close" in data.columns:
        data = data[["Close"]]
    prices = data.dropna(how="all").dropna(axis=1, how="all").ffill().dropna()
    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"Warning: no data for {sorted(missing)} (dropped).")
    print(f"Got {prices.shape[0]} trading days x {prices.shape[1]} assets "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")
    return prices


# ---------------------------------------------------------------------------
# 2. Returns statistics
# ---------------------------------------------------------------------------
def calculate_statistics(prices):
    """
    Returns:
        daily_returns : DataFrame of pct-change daily returns
        exp_returns   : Series of annualized expected returns (mean * 252)
        cov_matrix    : DataFrame of annualized covariance (cov * 252)
    """
    daily_returns = prices.pct_change().dropna()
    exp_returns = daily_returns.mean() * TRADING_DAYS
    cov_matrix = daily_returns.cov() * TRADING_DAYS
    return daily_returns, exp_returns, cov_matrix


# ---------------------------------------------------------------------------
# 3. Portfolio math
# ---------------------------------------------------------------------------
def portfolio_performance(weights, exp_returns, cov_matrix, risk_free_rate=0.0):
    """Annualized (return, volatility, sharpe) for a weight vector."""
    w = np.asarray(weights)
    ret = float(np.dot(w, exp_returns.values))
    vol = float(np.sqrt(w @ cov_matrix.values @ w))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _neg_sharpe(weights, exp_returns, cov_matrix, risk_free_rate):
    r, v, s = portfolio_performance(weights, exp_returns, cov_matrix, risk_free_rate)
    return -s


def _portfolio_vol(weights, exp_returns, cov_matrix, risk_free_rate=None):
    return float(np.sqrt(np.asarray(weights) @ cov_matrix.values @ np.asarray(weights)))


def _base_constraints(exp_returns, target=None):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # fully invested
    if target is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, exp_returns.values) - t}
        )
    return constraints


def optimize_portfolio(exp_returns, cov_matrix, risk_free_rate=0.0,
                       objective="sharpe", target_return=None, bounds=None):
    """Single optimizer entry point: objective in {'sharpe', 'vol'}."""
    n = len(exp_returns)
    bounds = bounds or tuple((0.0, 1.0) for _ in range(n))  # long-only, no leverage
    x0 = np.repeat(1.0 / n, n)
    if objective == "sharpe":
        fun = _neg_sharpe
        args = (exp_returns, cov_matrix, risk_free_rate)
        constraints = _base_constraints(exp_returns)
    elif objective == "vol":
        fun = _portfolio_vol
        args = (exp_returns, cov_matrix, None)
        constraints = _base_constraints(exp_returns, target=target_return)
    else:
        raise ValueError("objective must be 'sharpe' or 'vol'")
    res = minimize(fun, x0, args=args, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"Optimization failed ({objective}): {res.message}")
    return res.x


def max_sharpe_weights(exp_returns, cov_matrix, risk_free_rate=0.0):
    return optimize_portfolio(exp_returns, cov_matrix, risk_free_rate, objective="sharpe")


def min_vol_weights(exp_returns, cov_matrix):
    return optimize_portfolio(exp_returns, cov_matrix, objective="vol")


def efficient_frontier(exp_returns, cov_matrix, n_points=60):
    """Trace the frontier by minimizing vol over a grid of target returns."""
    min_ret = float(exp_returns.min())
    max_ret = float(exp_returns.max())
    targets = np.linspace(min_ret, max_ret, n_points)
    rets, vols, weights = [], [], []
    for t in targets:
        try:
            w = optimize_portfolio(exp_returns, cov_matrix, objective="vol", target_return=t)
            r, v, _ = portfolio_performance(w, exp_returns, cov_matrix)
            rets.append(r)
            vols.append(v)
            weights.append(w)
        except RuntimeError:
            continue
    return np.array(rets), np.array(vols), np.array(weights)


def random_portfolios(exp_returns, cov_matrix, risk_free_rate=0.0, n=5000, seed=42):
    """Monte-Carlo cloud for visual context behind the frontier."""
    rng = np.random.default_rng(seed)
    n_assets = len(exp_returns)
    out_r, out_v, out_s = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        w = rng.random(n_assets)
        w /= w.sum()
        r, v, s = portfolio_performance(w, exp_returns, cov_matrix, risk_free_rate)
        out_r[i], out_v[i], out_s[i] = r, v, s
    return out_r, out_v, out_s


# ---------------------------------------------------------------------------
# 4. Reporting / plotting
# ---------------------------------------------------------------------------
def weights_table(weights, tickers):
    df = pd.DataFrame({"ticker": tickers, "weight": np.asarray(weights)})
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    df["weight_pct"] = (df["weight"] * 100).round(2)
    return df


def plot_frontier(frontier_rets, frontier_vols, rand_r, rand_v, rand_s,
                  max_sharpe_pt, min_vol_pt, out_path="efficient_frontier.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(rand_v, rand_r, c=rand_s, cmap="viridis", alpha=0.35, s=12,
                    label="Random portfolios")
    plt.colorbar(sc, ax=ax, label="Sharpe ratio")
    ax.plot(frontier_vols, frontier_rets, "r-", lw=2.5, label="Efficient Frontier")
    ax.scatter(*max_sharpe_pt[1::-1], c="red", s=140, marker="*",
               edgecolors="black", label="Max Sharpe", zorder=5)
    ax.scatter(*min_vol_pt[1::-1], c="gold", s=120, marker="D",
               edgecolors="black", label="Min Volatility", zorder=5)
    ax.set_xlabel("Annualized Volatility (std)")
    ax.set_ylabel("Annualized Expected Return")
    ax.set_title("Efficient Frontier — 12 Niche Stocks (MPT)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved frontier chart -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MPT Optimizer: frontier + max Sharpe ratio")
    ap.add_argument("--period", default="3y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--risk-free", type=float, default=0.02, dest="risk_free")
    ap.add_argument("--points", type=int, default=60)
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    prices = fetch_prices(args.tickers, period=args.period, interval=args.interval)
    tickers = list(prices.columns)

    daily_returns, exp_returns, cov_matrix = calculate_statistics(prices)

    print("\n--- Annualized Expected Returns ---")
    print((exp_returns * 100).round(2).astype(str) + " %")
    print("\n--- Annualized Covariance Matrix (first 5x5) ---")
    print((cov_matrix.iloc[:5, :5]).round(4))

    w_sharpe = max_sharpe_weights(exp_returns, cov_matrix, args.risk_free)
    w_minvol = min_vol_weights(exp_returns, cov_matrix)
    r_s, v_s, s_s = portfolio_performance(w_sharpe, exp_returns, cov_matrix, args.risk_free)
    r_m, v_m, s_m = portfolio_performance(w_minvol, exp_returns, cov_matrix, args.risk_free)

    print("\n=== MAX SHARPE PORTFOLIO ===")
    print(weights_table(w_sharpe, tickers).to_string(index=False))
    print(f"Expected return: {r_s:.2%} | Volatility: {v_s:.2%} | Sharpe: {s_s:.3f}")

    print("\n=== MIN VOLATILITY PORTFOLIO ===")
    print(weights_table(w_minvol, tickers).to_string(index=False))
    print(f"Expected return: {r_m:.2%} | Volatility: {v_m:.2%} | Sharpe: {s_m:.3f}")

    # Equal-weight baseline
    w_eq = np.repeat(1 / len(tickers), len(tickers))
    r_e, v_e, s_e = portfolio_performance(w_eq, exp_returns, cov_matrix, args.risk_free)
    print(f"\nEqual-weight baseline: return {r_e:.2%} | vol {v_e:.2%} | Sharpe {s_e:.3f}")

    # Frontier
    f_rets, f_vols, _ = efficient_frontier(exp_returns, cov_matrix, n_points=args.points)
    print(f"\nEfficient Frontier: {len(f_rets)} points from {f_rets.min():.2%} to {f_rets.max():.2%} return")

    weights_table(w_sharpe, tickers).to_csv("max_sharpe_weights.csv", index=False)
    weights_table(w_minvol, tickers).to_csv("min_vol_weights.csv", index=False)
    print("Saved max_sharpe_weights.csv, min_vol_weights.csv")

    if not args.no_plot:
        rand_r, rand_v, rand_s = random_portfolios(exp_returns, cov_matrix, args.risk_free)
        plot_frontier(f_rets, f_vols, rand_r, rand_v, rand_s,
                      max_sharpe_pt=(r_s, v_s), min_vol_pt=(r_m, v_m))


if __name__ == "__main__":
    main()
