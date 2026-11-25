"""Tests for mpt.config: universe, themes, validation."""
import pytest

from mpt.config import DEFAULT_TICKERS, SECTORS, THEMES, OptimizerConfig

def test_default_universe_size():
    assert len(DEFAULT_TICKERS) == 12


