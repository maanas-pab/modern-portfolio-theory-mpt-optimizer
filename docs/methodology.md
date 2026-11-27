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

## The three optimizations

All problems are solved with `scipy.optimize.minimize` (SLSQP),
long-only bounds `0 ≤ wᵢ ≤ 1` unless stated otherwise.

| Portfolio | Objective | Constraints |
|-----------|-----------|-------------|
| Max Sharpe | minimize `−S(w)` | `Σwᵢ = 1` |
| Min Volatility | minimize `√(wᵀΣw)` | `Σwᵢ = 1` |
| Frontier point | minimize `√(wᵀΣw)` | `Σwᵢ = 1`, `wᵀμ = target` |
| Risk Parity | minimize `Σᵢ (RCᵢ − 1/n)²` | `Σwᵢ = 1` |

`RCᵢ` is the percentage risk contribution `(wᵢ · (Σw)ᵢ) / σ_p`.

