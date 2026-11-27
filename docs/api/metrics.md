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

