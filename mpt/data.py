"""
Price data: download via yfinance, cache to CSV, load offline.

The cache makes runs reproducible and keeps tests/CI off the network.
"""
import os
import time
from collections.abc import Iterable
from typing import Optional

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


def load_cache(path: str) -> pd.DataFrame:
    """Read a cached price CSV (index = dates)."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    if prices.empty:
        raise ValueError(f"cache {path} is empty")
    return prices.astype(float)


def save_cache(prices: pd.DataFrame, path: str) -> str:
    """Persist prices to CSV, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prices.to_csv(path)
    return path


def get_prices(tickers: Iterable[str] = DEFAULT_TICKERS,
               period: str = "3y",
               interval: str = "1d",
               cache_path: Optional[str] = "data/prices_3y.csv",
               refresh: bool = False) -> pd.DataFrame:
    """Return prices from cache when possible, else download (and refresh cache)."""
    tickers = list(tickers)
    if cache_path and os.path.exists(cache_path) and not refresh:
        prices = load_cache(cache_path)
        missing = [t for t in tickers if t not in prices.columns]
        if not missing:
            print(f"Loaded {prices.shape[0]} x {prices.shape[1]} from cache ({cache_path})")
            return prices[tickers] if set(tickers) <= set(prices.columns) else prices
        print(f"Cache missing {missing} - refetching.")
    prices = fetch_prices(tickers, period=period, interval=interval)
    if cache_path:
        save_cache(prices, cache_path)
    return prices


def attach_benchmark(prices: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Return prices including the benchmark column when it is available."""
    if benchmark in prices.columns:
        return prices
    bench = _download_one(benchmark, "3y", "1d")
    if bench is None:
        print(f"Warning: benchmark {benchmark} unavailable - continuing without it.")
        return prices
    return prices.join(bench, how="inner")
