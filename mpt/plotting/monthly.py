"""
Calendar-style heatmaps of monthly returns for a portfolio
and for every asset.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE, finish, percent_axis, setup_style

def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compound daily returns into a year x month table of decimals."""
    r = returns.dropna()
    grouped = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    table = pd.DataFrame({
        "year": [y for y, _ in grouped.index],
        "month": [m for _, m in grouped.index],
        "ret": grouped.values,
    }).pivot(index="year", columns="month", values="ret")
    table = table.reindex(columns=range(1, 13))
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return table


