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

## Covariance shrinkage

Sample covariance from ~750 observations of 12 correlated assets is
noisy. The Ledoit–Wolf (2004) estimator shrinks it toward the
scaled-identity target `F = μ·I`:

```
λ* = min(1, b² / d²)
Σ_shrunk = (1 − λ*)·Σ_sample + λ*·μ·I
```

where `d²` is the squared distance from the sample to the target and
`b²` estimates the sample covariance's mean-squared estimation error
under Gaussian fourth moments. The result is better conditioned
(smaller eigenvalue spread), which stabilizes the optimizer.

Enable it with `--shrinkage`.

