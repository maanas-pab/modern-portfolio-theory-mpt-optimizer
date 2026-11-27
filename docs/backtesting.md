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

## Rebalancing

`--rebalance {none,weekly,monthly,quarterly}` snaps weights back to
the target on the first trading day of each period. Turnover is the
L1 distance between the drifted and target weights; each rebalance
pays `--cost` per unit of turnover (one-way, so `0.001` = 10 bps).

| Mode | Meaning |
|------|---------|
| `none` | buy-and-hold: the passive baseline |
| `weekly` | every week (≈52/yr) |
| `monthly` | first trading day of each month (≈12/yr) |
| `quarterly` | first trading day of each quarter (≈4/yr) |

