# FAQ

## Data

**Does it need network access?**
No. `data/prices_3y.csv` is bundled; pass `--cache data/prices_3y.csv`
(the default) and the pipeline runs offline. `--refresh` forces a
fresh Yahoo download.

**Can I use my own tickers?**
`--tickers AAPL MSFT ...` replaces the universe. Sector/theme colors
fall back to a generic palette for unknown tickers.

## Optimization

**Why is my max-Sharpe portfolio 90% one ticker?**
Expected returns are sample means — the noisiest input in the model.
The optimizer concentrates wherever the mean/vol ratio looks best.
Run `--shrinkage` and compare, or `--black-litterman` to temper the
priors.

**Why does SLSQP fail on some target returns?**
The target may be outside the feasible set (long-only convex hull of
asset means). The optimizer raises `RuntimeError` naming the target.

**What does `budget="gross"` do?**
Relaxes `Σw = 1` to `Σ|w| = 1`, allowing long-short solutions.

## Backtesting

**Are the backtest numbers tradable?**
No — they are in-sample. The weights were chosen on the same history
they are simulated on. Use `--walk-forward` for an honest estimate;
expect the Sharpe to drop a lot.

**How is turnover cost applied?**
At each rebalance, `cost = turnover × transaction_cost` is deducted
from that day's return. `total_cost` in the output is the cumulative
drag.

## Tests & CI

**How do I run the tests?**
`pytest tests/` — everything is offline and uses synthetic fixtures
plus the bundled cache. ~293 tests, a few seconds.

**Does CI need secrets?**
No. The workflow runs lint + pytest on Python 3.9–3.12 with no
network calls.

## Figures

**My figures look different from the README.**
Regenerate with `make figures` (the exact flags used are in the
Makefile). Style constants live in `mpt/plotting/style.py`.

**How do I add a new chart?**
Write a `plot_*` function in the right `mpt/plotting/` submodule
following the `setup_style()` → `finish()` contract, add a test in
`tests/test_plotting.py`, and document it in `docs/charts.md`.
