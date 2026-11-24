"""
Human-readable tables, CSV exports, and the console summary report.
"""
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .metrics import (calmar_ratio, conditional_var, concentration_hhi,
                      max_drawdown, portfolio_performance, sortino_ratio,
                      value_at_risk)

def weights_table(weights, tickers, name: str = "portfolio") -> pd.DataFrame:
    """Sorted ticker/weight table with percentage column."""
    df = pd.DataFrame({"ticker": list(tickers), "weight": np.asarray(weights, dtype=float)})
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    df["weight_pct"] = (df["weight"] * 100).round(2)
    df.insert(0, "portfolio", name)
    return df


def export_weights(weights, tickers, out_dir: str, name: str) -> str:
    """Write a weights table to outputs/<name>.csv and return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    weights_table(weights, tickers, name).to_csv(path, index=False)
    return path


def performance_summary(returns: pd.Series,
                        risk_free: float = 0.0,
                        benchmark: Optional[pd.Series] = None,
                        periods: int = 252) -> Dict[str, float]:
    """One flat dict of every headline metric for a return series."""
    r = returns.dropna()
    annual_ret = float(r.mean() * periods)
    annual_vol = float(r.std(ddof=1) * np.sqrt(periods))
    dd = max_drawdown(r)
    summary = {
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        # guard both true zero and floating-point dust (constant series)
        "sharpe": (annual_ret - risk_free) / annual_vol if annual_vol > 1e-12 else 0.0,
        "sortino": sortino_ratio(r, risk_free, periods),
        "max_drawdown": dd,
        "calmar": calmar_ratio(annual_ret, dd, risk_free),
        "var_95": value_at_risk(r, 0.95),
        "cvar_95": conditional_var(r, 0.95),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "positive_days": float((r > 0).mean()),
    }
    if benchmark is not None:
        from .metrics import alpha as alpha_fn
        from .metrics import beta as beta_fn

        joined = pd.concat([r, benchmark], axis=1).dropna()
        if len(joined) > 2:
            summary["beta"] = beta_fn(joined.iloc[:, 0], joined.iloc[:, 1])
            summary["alpha"] = alpha_fn(joined.iloc[:, 0], joined.iloc[:, 1], risk_free, periods)
    return summary


def summary_table(named_summaries: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Pivot {strategy: {metric: value}} into a metric-by-strategy table."""
    df = pd.DataFrame(named_summaries)
    pct_rows = {"annual_return", "annual_vol", "max_drawdown", "var_95", "cvar_95",
                "positive_days", "best_day", "worst_day", "total_return", "total_cost"}
    for row in df.index:
        if row in pct_rows:
            df.loc[row] = (df.loc[row] * 100).round(2)
        else:
            df.loc[row] = df.loc[row].round(3)
    return df


def print_report(title: str, summary: Dict[str, float],
                 weights_df: Optional[pd.DataFrame] = None) -> None:
    """Pretty-print one strategy block to stdout."""
    print(f"\n=== {title} ===")
    if weights_df is not None:
        print(weights_df[["ticker", "weight_pct"]].to_string(index=False))
    print(f"Return {summary['annual_return']:.2%} | Vol {summary['annual_vol']:.2%} "
          f"| Sharpe {summary['sharpe']:.3f} | Sortino {summary['sortino']:.3f}")
    print(f"MaxDD {summary['max_drawdown']:.2%} | Calmar {summary['calmar']:.3f} "
          f"| VaR95 {summary['var_95']:.2%} | CVaR95 {summary['cvar_95']:.2%}")
    if "beta" in summary:
        print(f"Beta {summary['beta']:.3f} | Alpha {summary['alpha']:.2%}")