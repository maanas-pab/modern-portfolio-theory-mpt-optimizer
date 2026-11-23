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


