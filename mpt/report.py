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


