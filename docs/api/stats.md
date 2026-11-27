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

```python
daily, exp_returns, cov = calculate_statistics(prices)
cov_shrunk = covariance_matrix(daily, shrinkage=True)
print(cov_shrunk.attrs["shrinkage_intensity"])
```

`calculate_statistics` is the single entry point the CLI uses; the
individual helpers exist so tests and notebooks can compose them.
