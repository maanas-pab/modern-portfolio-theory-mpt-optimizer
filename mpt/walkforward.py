"""
Walk-forward (rolling out-of-sample) optimization
==================================================
The in-sample backtest in :mod:`mpt.backtest` chooses weights on the
*same* history it evaluates — results are look-ahead.  This module
closes that hole:

1. split the history into rolling **train** / **test** windows,
2. optimize on the train window only,
3. hold those weights through the next test window,
4. stitch the out-of-sample equity curve together.

The honest performance number is whatever the strategy earned on data
the optimizer had never seen.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import backtest
from .metrics import max_drawdown, sharpe_ratio
from .optimizer import max_sharpe_weights, min_vol_weights, risk_parity_weights
from .stats import calculate_statistics


@dataclass
class WalkForwardResult:
    """Out-of-sample equity plus the per-window weights that produced it."""

    equity: pd.Series
    returns: pd.Series
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    weights_by_window: List[dict] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def total_return(self) -> float:
        # equity compounds from 1.0 before the first out-of-sample day
        return float(self.equity.iloc[-1] - 1.0)

    def summary(self, risk_free: float = 0.02) -> dict:
        """Headline out-of-sample metrics."""
        r = self.returns
        ann = float((1.0 + r.mean()) ** 252 - 1.0) if len(r) else 0.0
        vol = float(r.std() * np.sqrt(252)) if len(r) else 0.0
        return {
            "annual_return": ann,
            "annual_vol": vol,
            "sharpe": sharpe_ratio(ann, vol, risk_free),
            "max_drawdown": max_drawdown(r),
            "n_windows": self.n_windows,
        }


def rolling_windows(index: pd.DatetimeIndex, train_days: int,
                    test_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Rolling ``(train_start, train_end, test_end)`` triples.

    The first window starts at ``index[0]``; each subsequent window slides
    forward by ``test_days``.  The final window is dropped if fewer than
    ``test_days`` observations remain.
    """
    if train_days < 2:
        raise ValueError("train_days must be at least 2")
    if test_days < 1:
        raise ValueError("test_days must be at least 1")
    if train_days + test_days > len(index):
        raise ValueError(
            f"window ({train_days}+{test_days}) exceeds the {len(index)} observations")

    out, start = [], 0
    while start + train_days + test_days <= len(index):
        train_start = index[start]
        train_end = index[start + train_days - 1]
        test_end = index[start + train_days + test_days - 1]
        out.append((train_start, train_end, test_end))
        start += test_days
    if not out:
        raise ValueError("no complete train/test window fits this history")
    return out


_OBJECTIVES = {
    "sharpe": lambda exp, cov, rf: max_sharpe_weights(exp, cov, rf),
    "vol": lambda exp, cov, rf: min_vol_weights(exp, cov),
    "risk_parity": lambda exp, cov, rf: risk_parity_weights(cov),
}


def _fit_weights(train: pd.DataFrame, objective: str, risk_free: float) -> np.ndarray:
    _, exp, cov = calculate_statistics(train)
    if objective not in _OBJECTIVES:
        raise ValueError(
            f"objective must be one of {sorted(_OBJECTIVES)}, got {objective!r}")
    return _OBJECTIVES[objective](exp, cov, risk_free)


def walk_forward(prices: pd.DataFrame,
                 train_days: int = 252,
                 test_days: int = 63,
                 objective: str = "sharpe",
                 risk_free: float = 0.02,
                 start: Optional[pd.Timestamp] = None) -> WalkForwardResult:
    """
    Optimize on each train window, hold through the following test window.

    Parameters
    ----------
    prices : full price history (the test slices come from it)
    train_days : observations used to estimate mu/Sigma per window
    test_days  : observations each fitted weight set is held for
    objective  : 'sharpe' | 'vol' | 'risk_parity'
    risk_free  : passed to the Sharpe objective
    start      : optional first date (slice the history beforehand)

    Returns
    -------
    WalkForwardResult whose equity compounds from 1.0 before the first
    out-of-sample day.
    """
    if start is not None:
        prices = prices.loc[start:]
    if len(prices) < train_days + test_days:
        raise ValueError(
            f"need at least {train_days + test_days} observations, got {len(prices)}")

    windows = rolling_windows(prices.index, train_days, test_days)
    chunks, weight_sets, spans = [], [], []
    for train_start, train_end, test_end in windows:
        train = prices.loc[train_start:train_end]
        test = prices.loc[train_end:test_end]
        if len(test) < 2:
            continue
        w = _fit_weights(train, objective, risk_free)
        bt = backtest(test, w, rebalance="none")
        chunks.append(bt.returns)
        weight_sets.append(dict(zip(prices.columns, np.round(w, 6))))
        spans.append((train_end, test_end))

    if not chunks:
        raise ValueError("no test window produced returns")

    rets = pd.concat(chunks)
    rets = rets[~rets.index.duplicated(keep="first")]
    rets = rets.sort_index()
    # cumprod compounds from 1.0 placed before the first test day
    equity = (1.0 + rets).cumprod()
    return WalkForwardResult(equity=equity, returns=rets,
                             windows=spans, weights_by_window=weight_sets)


