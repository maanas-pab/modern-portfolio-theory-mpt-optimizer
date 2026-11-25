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


