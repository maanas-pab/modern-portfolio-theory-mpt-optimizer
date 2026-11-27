# `mpt.config`

```python
from mpt.config import DEFAULT_TICKERS, SECTORS, THEMES, OptimizerConfig
```

## Constants

| Name | Value | Meaning |
|------|-------|---------|
| `DEFAULT_TICKERS` | 12 tickers | the niche-growth universe |
| `BENCHMARK` | `"SPY"` | benchmark for beta/alpha |
| `TRADING_DAYS` | `252` | annualization factor |
| `THEMES` | dict | ticker -> company/theme description |
| `SECTORS` | dict | ticker -> sector label (used by the sector pie) |

## `OptimizerConfig`

Dataclass holding every CLI flag. Key fields:

```python
cfg = OptimizerConfig(
    period="3y", risk_free=0.02, shrinkage=False,
    frontier_points=60, random_portfolios=5000,
    rebalance="none", transaction_cost=0.0,
    walk_forward=False, train_days=252, test_days=63,
    black_litterman=False,
    out_dir="outputs", figure_dir="figures",
    no_plot=False, no_backtest=False,
).validate()   # returns self, raises ValueError on nonsense
```

`validate()` checks the rebalance frequency, cost sign, frontier point
count, risk-free range and walk-forward window sizes — before any
network or optimizer work happens.
