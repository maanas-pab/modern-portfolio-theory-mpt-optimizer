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

## Risk metrics

| Metric | Definition |
|--------|-----------|
| Sortino | annualized excess return / downside deviation |
| Max drawdown | worst peak-to-trough equity decline |
| Calmar | annualized excess return / \|max drawdown\| |
| VaR 95% | 5th percentile of daily losses (historical or normal) |
| CVaR 95% | mean loss beyond the VaR threshold (expected shortfall) |
| Beta | `cov(r_p, r_b) / var(r_b)` |
| HHI | `Σ wᵢ²` — concentration from `1/n` (equal) to `1` (single asset) |

## Black-Litterman

Sample means are the noisiest input in the whole model. The
Black-Litterman (1992) approach replaces them with a precision-weighted
blend of a **prior** (market-implied equilibrium returns) and **views**
(an investor's relative opinions):

```
pi   = δ · Σ · w_mkt                            # equilibrium prior
mu   = [ (τΣ)⁻¹ + Pᵀ Ω⁻¹ P ]⁻¹ · [ (τΣ)⁻¹ pi + Pᵀ Ω⁻¹ Q ]
```

- `P` is the pick matrix (which assets each view concerns),
- `Q` the view returns, `τ` the prior scaling (0.05 by default),
- `Ω` the view uncertainty — from confidence levels or a fixed scalar;
  diffuse views (large Ω) make the posterior collapse back to the
  prior, confident views pull it toward `Q`.

The `--black-litterman` flag builds "top-3 outperformer" views from the
sample means and optimizes the max-Sharpe portfolio on the posterior.

## Assumptions & limitations

- Expected returns are backward-looking sample means — notoriously noisy
- Single-period model: no path-dependent utility, no rebalancing inside the optimizer
- Long-only, fully invested (shorts available via `budget="gross"`)
- No transaction costs inside the optimizer (costs modeled in the backtest)
- Past performance does not predict future results
