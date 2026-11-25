"""Tests for mpt.cli: parsing, config wiring, end-to-end on cache."""
import numpy as np
import pytest

from mpt.cli import build_parser, config_from_args, main, run
from mpt.config import OptimizerConfig

def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.period == "3y"
    assert args.risk_free == 0.02
    assert args.rebalance == "none"


def test_config_from_args_roundtrip():
    args = build_parser().parse_args(
        ["--period", "1y", "--risk-free", "0.03", "--points", "25",
         "--shrinkage", "--rebalance", "quarterly", "--cost", "0.002"])
    cfg = config_from_args(args)
    assert cfg.period == "1y"
    assert cfg.risk_free == 0.03
    assert cfg.frontier_points == 25
    assert cfg.shrinkage is True
    assert cfg.rebalance == "quarterly"
    assert cfg.transaction_cost == 0.002


def test_parser_rejects_bad_rebalance():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--rebalance", "yearly"])


def test_config_from_args_validates():
    args = build_parser().parse_args(["--points", "1"])
    with pytest.raises(ValueError):
        config_from_args(args)


