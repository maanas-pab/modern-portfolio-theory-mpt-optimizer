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

`performance_summary` keys: `annual_return`, `annual_vol`, `sharpe`,
`sortino`, `max_drawdown`, `calmar`, `var_95`, `cvar_95`,
`positive_days`, plus `beta` and `alpha` when a benchmark series is
supplied.

`summary_table` scales percentages to readable numbers (12.34 for
12.34%) and leaves ratios like Sharpe at three decimals.
