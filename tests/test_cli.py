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


