# FAQ

## Data

**Does it need network access?**
No. `data/prices_3y.csv` is bundled; pass `--cache data/prices_3y.csv`
(the default) and the pipeline runs offline. `--refresh` forces a
fresh Yahoo download.

**Can I use my own tickers?**
`--tickers AAPL MSFT ...` replaces the universe. Sector/theme colors
fall back to a generic palette for unknown tickers.

