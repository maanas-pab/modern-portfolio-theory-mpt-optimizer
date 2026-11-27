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

All anchors carry `weights`, `return`, `vol`, `sharpe` — the same
shape `plot_frontier_clean` consumes.

The frontier is traced by minimizing volatility at a grid of target
returns, so it is monotone by construction above the minimum-vol point
(asserted in the test suite).
