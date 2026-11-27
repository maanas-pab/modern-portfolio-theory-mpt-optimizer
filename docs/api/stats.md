# `mpt.stats`

```python
from mpt.stats import calculate_statistics, covariance_matrix
```

| Function | Returns |
|----------|---------|
| `daily_returns(prices)` | pct change, first row dropped |
| `annualize_returns(daily, days=252)` | mean × days |
| `annualize_covariance(daily, days=252)` | cov × days |
| `calculate_statistics(prices)` | `(daily, exp_returns, cov)` — the trio the optimizer consumes |
| `correlation_matrix(prices)` | correlation of daily returns |
| `annualized_volatility(cov)` | `sqrt(diag(cov))` |
| `ledoit_wolf_intensity(daily)` | shrinkage λ in `[0, 1]` |
| `ledoit_wolf_covariance(daily)` | shrunk cov, λ in `.attrs` |
| `covariance_matrix(daily, shrinkage=False)` | dispatcher |

