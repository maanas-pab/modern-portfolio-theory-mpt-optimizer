"""Tests for mpt.data cache round-trips (no network)."""
import pandas as pd
import pytest

from mpt.data import get_prices, load_cache, save_cache

def test_save_cache_creates_dirs(tmp_path, prices):
    path = tmp_path / "nested" / "cache.csv"
    out = save_cache(prices, str(path))
    assert (tmp_path / "nested" / "cache.csv").exists()
    assert out.endswith("cache.csv")


def test_cache_roundtrip(tmp_path, prices):
    path = str(tmp_path / "p.csv")
    save_cache(prices, path)
    loaded = load_cache(path)
    pd.testing.assert_frame_equal(loaded, prices, check_freq=False)


def test_load_empty_cache_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(Exception):
        load_cache(str(path))


def test_get_prices_uses_cache(tmp_path, prices):
    path = str(tmp_path / "cache.csv")
    save_cache(prices, path)
    out = get_prices(list(prices.columns), cache_path=path)
    assert list(out.columns) == list(prices.columns)
    assert len(out) == len(prices)


def test_get_prices_missing_ticker_refetches(tmp_path, prices, monkeypatch):
    """A ticker absent from cache must trigger a refetch, not stale data."""
    path = str(tmp_path / "cache.csv")
    save_cache(prices[["AAA", "BBB"]], path)
    fetched = {}
    with_zzz = prices.copy()
    with_zzz["ZZZ"] = with_zzz["AAA"] * 1.01

    def fake_fetch(tickers, period="3y", interval="1d"):
        fetched["tickers"] = list(tickers)
        return with_zzz

    monkeypatch.setattr("mpt.data.fetch_prices", fake_fetch)
    out = get_prices(["AAA", "BBB", "ZZZ"], cache_path=path)
    assert "ZZZ" in fetched["tickers"]
    assert "ZZZ" in out.columns

