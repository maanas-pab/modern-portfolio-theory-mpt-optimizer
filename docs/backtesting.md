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

## Interpreting results

- `n_rebalances` counts how often the schedule fired
- `total_cost` is the cumulative drag from turnover
- With `rebalance=none` and equal weights, the backtest equals
  buy-and-hold exactly (a sanity check in the test suite)

## Caveats

- In-sample optimization: weights are chosen with the full history,
  then simulated on that same history — **results are look-ahead**
- No slippage, market impact, or taxes
- Survivorship bias: the universe is a fixed snapshot of tickers
- Treat the backtest as a *comparison between weighting schemes*,
  not a forecast

## Walk-forward: removing the look-ahead

`--walk-forward --train 252 --test 63` fixes the in-sample problem:

```
for each window:
    fit weights on train_days of history      # never sees the test period
    hold those weights for test_days          # pure out-of-sample
stitch the test slices into one equity curve
```

Windows slide forward by `test_days`, so every test observation is
scored with weights fitted only on earlier data. With the default
universe this gives 7 windows; the reported Sharpe (0.37) is what an
investor following the algorithm would actually have earned, versus
the 1.42 the in-sample backtest advertises.

The comparison is the point: **the gap between the two numbers is the
overfitting premium.**
