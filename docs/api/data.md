# `mpt.data`

```python
from mpt.data import fetch_prices, get_prices, load_cache, save_cache
```

| Function | Signature | Notes |
|----------|-----------|-------|
| `fetch_prices` | `(tickers, period="3y", interval="1d") -> DataFrame` | network; per-ticker retries |
| `get_prices` | `(tickers, cache_path=..., refresh=False) -> DataFrame` | cache-first, refetches only missing tickers |
| `load_cache` | `(path) -> DataFrame` | parses the CSV, `Date` as index |
| `save_cache` | `(prices, path) -> str` | creates parent directories |
| `attach_benchmark` | `(prices, benchmark="SPY") -> DataFrame` | adds the benchmark when available |

`get_prices` is what the pipeline calls: it reads
`data/prices_3y.csv`, downloads anything missing, and writes the
cache back. `refresh=True` bypasses the cache entirely.

A ticker Yahoo has never heard of is dropped with a warning rather
than failing the whole run; if *nothing* usable comes back,
`RuntimeError` is raised.
