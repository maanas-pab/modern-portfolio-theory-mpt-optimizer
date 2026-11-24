"""
Shared fixtures: synthetic GBM price paths so tests never touch
the network, plus a few derived statistics bundles.
"""
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def prices() -> pd.DataFrame:
    """4 assets x 500 days of geometric-Brownian-motion prices (seeded)."""
    rng = np.random.default_rng(7)
    n_days, n_assets = 500, 4
    mu = np.array([0.08, 0.12, 0.05, 0.10]) / 252
    sigma = np.array([0.20, 0.35, 0.15, 0.25]) / np.sqrt(252)
    corr = np.array([
        [1.0, 0.3, 0.1, 0.2],
        [0.3, 1.0, 0.2, 0.4],
        [0.1, 0.2, 1.0, 0.1],
        [0.2, 0.4, 0.1, 1.0],
    ])
    cov = np.outer(sigma, sigma) * corr
    drift = np.diag(cov)
    log_rets = rng.multivariate_normal(mu - drift / 2, cov, size=n_days)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    prices = pd.DataFrame(np.exp(np.cumsum(log_rets, axis=0)) * 100.0,
                          index=dates,
                          columns=["AAA", "BBB", "CCC", "DDD"])
    prices.index.name = "Date"
    return prices


@pytest.fixture
def stats(prices):
    """(daily, exp_returns, cov) computed from the synthetic prices."""
    from mpt.stats import calculate_statistics

    return calculate_statistics(prices)


@pytest.fixture
def benchmark(prices) -> pd.Series:
    """A synthetic benchmark that correlates with the assets."""
    rng = np.random.default_rng(11)
    rets = prices.pct_change().mean(axis=1) + rng.normal(0, 0.004, len(prices))
    return pd.Series(rets.values, index=prices.index, name="BENCH")


