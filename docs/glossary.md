# Glossary

Terms used across the README, docs and console output, in alphabetical
order. Cross-references link to the deeper explanation.

## A – E

| Term | Meaning |
|------|---------|
| Alpha (α) | excess return over what CAPM beta predicts: `r_p − rf − β(r_m − rf)` |
| Annualization | scaling a daily statistic by 252 trading days (vol by `√252`) |
| Beta (β) | sensitivity to the benchmark: `cov(r_p, r_b) / var(r_b)` |
| Black-Litterman | blending equilibrium prior returns with investor views |
| Calmar | annualized return / \|max drawdown\| |
| Capital market line | risk-free rate to the tangency portfolio, extended out |
| CVaR | mean loss beyond the VaR threshold (expected shortfall) |

## F – M

| Term | Meaning |
|------|---------|
| Frontier | set of portfolios maximizing return for each volatility level |
| HHI | `Σ wᵢ²` concentration index, `1/n` (equal) → `1` (single asset) |
| Long-only | no short sales; every weight ≥ 0 |
| Look-ahead bias | using future information in a backtest — what `--walk-forward` removes |
| Marginal risk | `∂σ_p/∂wᵢ`, the derivative of portfolio vol w.r.t. one weight |
| Max drawdown | worst peak-to-trough equity decline (≤ 0) |
| Monte-Carlo cloud | thousands of random portfolios plotted for context |

