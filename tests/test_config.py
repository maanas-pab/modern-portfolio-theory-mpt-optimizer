"""Tests for mpt.config: universe, themes, validation."""
import pytest

from mpt.config import DEFAULT_TICKERS, SECTORS, THEMES, OptimizerConfig


def test_default_universe_size():
    assert len(DEFAULT_TICKERS) == 12


def test_default_universe_unique():
    assert len(set(DEFAULT_TICKERS)) == len(DEFAULT_TICKERS)


def test_themes_cover_universe():
    assert set(THEMES) >= set(DEFAULT_TICKERS)


def test_sectors_cover_universe():
    assert set(SECTORS) >= set(DEFAULT_TICKERS)


def test_universe_uses_duol_not_roku():
    assert "DUOL" in DEFAULT_TICKERS
    assert "ROKU" not in DEFAULT_TICKERS


def test_default_config_validates():
    cfg = OptimizerConfig().validate()
    assert cfg.risk_free == 0.02
    assert cfg.rebalance == "none"


def test_invalid_rebalance_rejected():
    with pytest.raises(ValueError, match="rebalance"):
        OptimizerConfig(rebalance="daily").validate()


def test_negative_cost_rejected():
    with pytest.raises(ValueError, match="transaction_cost"):
        OptimizerConfig(transaction_cost=-0.01).validate()


def test_too_few_frontier_points_rejected():
    with pytest.raises(ValueError, match="frontier_points"):
        OptimizerConfig(frontier_points=1).validate()


def test_risk_free_bounds():
    with pytest.raises(ValueError, match="risk_free"):
        OptimizerConfig(risk_free=1.5).validate()
    with pytest.raises(ValueError, match="risk_free"):
        OptimizerConfig(risk_free=-0.1).validate()


def test_validate_is_chainable():
    cfg = OptimizerConfig()
    assert cfg.validate() is cfg
