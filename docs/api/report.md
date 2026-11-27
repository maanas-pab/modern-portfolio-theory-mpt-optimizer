# `mpt.report`

```python
from mpt.report import weights_table, performance_summary, print_report
```

| Function | Purpose |
|----------|---------|
| `weights_table(w, tickers, portfolio)` | sorted ticker/weight/weight_pct frame |
| `export_weights(w, tickers, out_dir, name)` | writes `outputs/<name>.csv`, returns path |
| `performance_summary(rets, rf, benchmark=None)` | dict of 11+ metrics |
| `summary_table(dict_of_summaries)` | metric-rows × strategy-columns, rounded |
| `print_report(name, summary, weights)` | console block |

