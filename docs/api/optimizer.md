# `mpt.optimizer`

```python
from mpt.optimizer import (optimize_portfolio, max_sharpe_weights,
                           min_vol_weights, risk_parity_weights)
```

| Function | Objective | Constraints |
|----------|-----------|-------------|
| `max_sharpe_weights(mu, cov, rf)` | maximize Sharpe | long-only, Σw = 1 |
| `min_vol_weights(mu, cov, target_return=None)` | minimize σ | Σw = 1 (+ target) |
| `risk_parity_weights(cov)` | equal risk budgets | Σw = 1 |
| `optimize_portfolio(...)` | dispatcher | `objective`, `bounds`, `budget`, `x0` |

