"""
Configuration for the MPT optimizer.

Central place for the stock universe, sector/theme tags, and
default parameters so every module and CLI flag agrees.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

TRADING_DAYS = 252

# 12 niche / thematic growth stocks across clean energy, biotech,
# cloud, fintech and edtech.  DUOL replaced ROKU in Nov 2026 after
# Yahoo Finance stopped serving usable ROKU history.
DEFAULT_TICKERS: List[str] = [
    "CELH",  # Celsius Holdings (energy drinks)
    "ENPH",  # Enphase Energy (solar microinverters)
    "FSLR",  # First Solar
    "CRSP",  # CRISPR Therapeutics
    "BEAM",  # Beam Therapeutics
    "DUOL",  # Duolingo (language learning)
    "ETSY",  # Etsy
    "PLTR",  # Palantir
    "NET",   # Cloudflare
    "DDOG",  # Datadog
    "U",     # Unity Software
    "AFRM",  # Affirm Holdings
]

BENCHMARK = "SPY"

THEMES = {
    "CELH": "Consumer / energy drinks",
    "ENPH": "Solar microinverters",
    "FSLR": "Solar manufacturing",
    "CRSP": "Gene editing",
    "BEAM": "Gene editing",
    "DUOL": "Edtech / language learning",
    "ETSY": "E-commerce",
    "PLTR": "AI / data analytics",
    "NET": "Cloud / edge",
    "DDOG": "Cloud observability",
    "U": "Gaming / 3D engine",
    "AFRM": "Fintech / BNPL",
    "SPY": "US large-cap index",
}

SECTORS = {
    "CELH": "Consumer",
    "ENPH": "Clean energy",
    "FSLR": "Clean energy",
    "CRSP": "Biotech",
    "BEAM": "Biotech",
    "DUOL": "Technology",
    "ETSY": "Consumer",
    "PLTR": "Technology",
    "NET": "Technology",
    "DDOG": "Technology",
    "U": "Technology",
    "AFRM": "Financials",
    "SPY": "Benchmark",
}


@dataclass
class OptimizerConfig:
    """Everything the CLI needs in one object."""

    tickers: Sequence[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    period: str = "3y"
    interval: str = "1d"

    risk_free: float = 0.02
    shrinkage: bool = False          # Ledoit-Wolf style covariance shrinkage
    benchmark: str = BENCHMARK

    frontier_points: int = 60
    random_portfolios: int = 5000
    rebalance: str = "none"          # none | weekly | monthly | quarterly
    transaction_cost: float = 0.0    # one-way cost per unit turnover

    out_dir: str = "outputs"
    figure_dir: str = "figures"
    no_plot: bool = False
    no_backtest: bool = False

    cache_path: Optional[str] = "data/prices_3y.csv"
    refresh: bool = False            # bypass cache, refetch from Yahoo

