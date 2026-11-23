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

