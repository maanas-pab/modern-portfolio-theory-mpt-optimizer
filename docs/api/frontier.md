# `mpt.frontier`

```python
from mpt.frontier import efficient_frontier, random_portfolios
```

| Function | Purpose |
|----------|---------|
| `efficient_frontier(mu, cov, n_points)` | `(rets, vols, weights)` traced over a target-return grid |
| `random_portfolios(mu, cov, rf, n, seed)` | Monte-Carlo cloud `(r, v, sharpe)` |
| `frontier_key_points(mu, cov, rf)` | dict of `max_sharpe` / `min_vol` / `equal` anchors |
| `capital_market_line(rf, ret, vol, vol_max)` | `(vols, rets)` of the CML |

