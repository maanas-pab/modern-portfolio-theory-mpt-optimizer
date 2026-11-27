# Backtesting

How the historical simulation works, and what it does *not* claim.

## Model

The engine walks the daily return series and applies the **previous
day's weights** to **today's returns**:

```
r_p(t) = w(t−1)ᵀ r(t) − cost·turnover(t)
```

Between rebalances, weights **drift** with the market:

```
w(t) = w(t−1) ⊙ (1 + r(t)) / Σ w(t−1) ⊙ (1 + r(t))
```

