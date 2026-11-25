"""
Historical backtests: buy-and-hold vs periodically rebalanced
portfolios, with transaction costs and weight tracking.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Everything a backtest produces, ready for charts and reports."""

    equity: pd.Series                       # compounded equity curve
    returns: pd.Series                      # daily portfolio returns
    weights_over_time: pd.DataFrame         # weights after each day's drift
    rebalance_dates: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_cost: float = 0.0

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)


def rebalance_schedule(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """First trading day of each week/month/quarter in `index`."""
    if freq == "none":
        return pd.DatetimeIndex([])
    rule = {"weekly": "W-MON", "monthly": "MS", "quarterly": "QS"}.get(freq)
    if rule is None:
        raise ValueError("freq must be none/weekly/monthly/quarterly")
    anchors = pd.date_range(index[0], index[-1] + pd.Timedelta(days=7), freq=rule)
    # first index entry on/after each anchor
    pos = index.searchsorted(anchors)
    pos = pos[pos < len(index)]
    return index[np.unique(pos)]


def backtest(prices: pd.DataFrame,
             weights,
             rebalance: str = "monthly",
             transaction_cost: float = 0.0,
             start: Optional[str] = None) -> BacktestResult:
    """
    Run a daily backtest of a static target weight vector.

    Between rebalances weights drift with the market; at each rebalance
    they snap back to target and pay `transaction_cost` per unit turnover
    (one-way, e.g. 0.001 = 10 bps).
    """
    prices = prices.loc[start:] if start else prices
    rets = prices.ffill().pct_change().dropna(how="all").fillna(0.0)
    target = np.asarray(weights, dtype=float)
    target = target / target.sum()

    sched = rebalance_schedule(rets.index, rebalance)
    w = target.copy()
    w_rows = []
    daily = []
    turnover_rows = []
    total_cost = 0.0
    rebalance_seen = []

    for dt in rets.index:
        row = rets.loc[dt].values
        # apply previous weights to today's returns
        port_ret = float(w @ row)
        w = w * (1.0 + row)
        w = w / max(w.sum(), 1e-12)          # drift (no cash drag)
        if dt in sched:
            turn = float(np.abs(target - w).sum())
            cost = transaction_cost * turn
            w = target.copy()
            port_ret -= cost
            total_cost += cost
            turnover_rows.append((dt, turn))
            rebalance_seen.append(dt)
        w_rows.append((dt, w.copy()))
        daily.append((dt, port_ret))

    equity = (1.0 + pd.Series(dict(daily))).cumprod()
    wdf = pd.DataFrame([r[1] for r in w_rows], index=[r[0] for r in w_rows],
                       columns=prices.columns)
    turnover = pd.Series(dict(turnover_rows)) if turnover_rows else pd.Series(dtype=float)
    return BacktestResult(equity=equity, returns=equity.pct_change().dropna(),
                          weights_over_time=wdf,
                          rebalance_dates=pd.DatetimeIndex(rebalance_seen),
                          turnover=turnover, total_cost=total_cost)


def buy_and_hold(prices: pd.DataFrame, weights=None) -> BacktestResult:
    """Backtest with rebalance='none' - the passive baseline."""
    n = prices.shape[1]
    w = np.repeat(1.0 / n, n) if weights is None else np.asarray(weights, dtype=float)
    return backtest(prices, w, rebalance="none")


def compare_strategies(prices: pd.DataFrame,
                       named_weights: dict[str, np.ndarray],
                       rebalance: str = "monthly",
                       transaction_cost: float = 0.0) -> dict[str, BacktestResult]:
    """Run the same schedule for several named weight vectors."""
    return {name: backtest(prices, w, rebalance=rebalance,
                           transaction_cost=transaction_cost)
            for name, w in named_weights.items()}
