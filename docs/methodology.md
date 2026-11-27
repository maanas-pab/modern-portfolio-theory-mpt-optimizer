# Methodology

This document explains the math behind every number the optimizer prints.

## Notation

| Symbol | Meaning |
|--------|---------|
| `w` | vector of portfolio weights (sums to 1) |
| `μ` | vector of annualized expected returns |
| `Σ` | annualized covariance matrix |
| `rf` | annualized risk-free rate |
| `r_t` | daily return series |

## Core formulas

**Portfolio return** — linear in weights:

```
μ_p = wᵀ μ
```

**Portfolio volatility** — the square root of the quadratic form:

```
σ_p = √(wᵀ Σ w)
```

**Sharpe ratio** — excess return per unit of total risk:

```
S = (μ_p − rf) / σ_p
```

Annualization uses 252 trading days: daily mean × 252, daily covariance × 252.

