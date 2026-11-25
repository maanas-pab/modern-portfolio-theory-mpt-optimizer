"""Tests for mpt.config: universe, themes, validation."""
import pytest

from mpt.config import DEFAULT_TICKERS, SECTORS, THEMES, OptimizerConfig

def test_default_universe_size():
    assert len(DEFAULT_TICKERS) == 12


def test_default_universe_unique():
    assert len(set(DEFAULT_TICKERS)) == len(DEFAULT_TICKERS)


