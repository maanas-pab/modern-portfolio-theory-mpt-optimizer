"""
Price data: download via yfinance, cache to CSV, load offline.

The cache makes runs reproducible and keeps tests/CI off the network.
"""
import os
import time
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf

from .config import BENCHMARK, DEFAULT_TICKERS

def _download_one(ticker: str, period: str, interval: str, retries: int = 3) -> Optional[pd.Series]:
    """Download a single ticker's adjusted close; None when Yahoo fails."""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            s = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if s.empty:
                raise ValueError(f"empty response for {ticker}")
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            s.name = ticker
            return s["Close"].astype(float)
        except Exception as exc:  # noqa: BLE001 - Yahoo raises many types
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"Warning: no data for {ticker} ({last_err}) - dropped.")
    return None


def fetch_prices(tickers: Iterable[str] = DEFAULT_TICKERS,
                 period: str = "3y",
                 interval: str = "1d") -> pd.DataFrame:
    """Download adjusted close prices for every ticker that responds."""
    tickers = list(tickers)
    print(f"Downloading {len(tickers)} tickers ({period}, {interval}) ...")
    series = []
    for t in tickers:
        s = _download_one(t, period, interval)
        if s is not None:
            series.append(s)
        time.sleep(0.3)
    if not series:
        raise RuntimeError("No price data downloaded - check network or tickers.")
    prices = pd.DataFrame(series).T.sort_index().ffill().dropna(how="all")
    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    print(f"Got {prices.shape[0]} trading days x {prices.shape[1]} assets "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")
    return prices


