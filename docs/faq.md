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

