# `mpt.plotting`

30 functions, one per figure, all with the same contract:

```python
plot_something(data, ..., out_path="figures/name.png") -> str
```

Each function:

1. calls `setup_style()` first (rcParams, grid, fonts),
2. builds one figure/axes pair,
3. ends with `finish(fig, ax, title, out_path)` — title, tight layout,
   save, close, `Saved chart -> ...` log line.

`PALETTE` and `ASSET_COLORS` keep the family looking consistent;
`percent_axis(ax)` formats an axis as percentages; `asset_color(i)`
gives the stable i-th categorical color.

| Submodule | Charts |
|-----------|--------|
| `frontier` | efficient frontier, clean anchors, shrinkage sensitivity |
| `correlations` | correlation/covariance heatmaps, correlation-vs-vol |
| `allocation` | grouped bars, pies, risk contributions, drift, concentration, trajectories |
| `performance` | cumulative growth, drawdown, rolling vol/Sharpe, turnover, walk-forward equity |
| `risk` | VaR/CVaR, Sharpe histogram, scoreboard, asset scatter, rolling beta, benchmark scatter |
| `monthly` | monthly/quarterly/annual calendars, asset-by-month |
| `distributions` | return histograms, sector pie |

See [the chart gallery](../charts.md) for what every figure shows.
