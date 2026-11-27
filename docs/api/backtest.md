# `mpt.backtest`

```python
from mpt.backtest import backtest, buy_and_hold, compare_strategies
```

`backtest(prices, weights, rebalance="monthly",
transaction_cost=0.0, start=None) -> BacktestResult`

The result carries:

| Field | Meaning |
|-------|---------|
| `equity` | compounded growth of $1 |
| `returns` | daily portfolio returns |
| `weights_over_time` | drifted (and snapped) weights per day |
| `turnover` | L1 distance paid at each rebalance |
| `rebalance_dates` | dates the schedule fired |
| `total_return`, `total_cost` | headline figures |

