# `mpt.metrics`

All series-based functions take **daily return** `pd.Series` and return
`float` (positive = loss magnitude for VaR/CVaR/drawdown).

| Group | Functions |
|-------|-----------|
| Ratios | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio` |
| Drawdown | `max_drawdown` (≤ 0) |
| Tail risk | `value_at_risk(level, method)`, `conditional_var` |
| Benchmark | `beta`, `alpha` |
| Weighting | `concentration_hhi`, `risk_contributions`, `risk_budget_loss` |
| Combined | `portfolio_performance(w, mu, cov, rf) -> (ret, vol, sharpe)` |

```python
from mpt.metrics import portfolio_performance, value_at_risk

ret, vol, sharpe = portfolio_performance(weights, exp_returns, cov, 0.02)
var95 = value_at_risk(daily_returns, level=0.95, method="historical")
```

`value_at_risk` supports `"historical"` (empirical quantile) and
`"parametric"` (normal) methods; `conditional_var` is the expected
shortfall beyond that threshold.
