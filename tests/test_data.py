"""Tests for mpt.data cache round-trips (no network)."""
import pandas as pd
import pytest

from mpt.data import get_prices, load_cache, save_cache

def test_save_cache_creates_dirs(tmp_path, prices):
    path = tmp_path / "nested" / "cache.csv"
    out = save_cache(prices, str(path))
    assert (tmp_path / "nested" / "cache.csv").exists()
    assert out.endswith("cache.csv")


