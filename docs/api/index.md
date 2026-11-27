# API Reference

Generated-by-hand reference for every public module. Each page lists the
signatures, what they return, and a one-liner example.

| Module | Purpose |
|--------|---------|
| [config.md](config.md) | universe, themes, `OptimizerConfig` |
| [data.md](data.md) | download, cache, benchmark attach |
| [stats.md](stats.md) | returns, covariance, shrinkage |
| [metrics.md](metrics.md) | ratios, drawdown, VaR, beta, HHI |
| [optimizer.md](optimizer.md) | SLSQP weight solvers |
| [frontier.md](frontier.md) | frontier tracing, Monte-Carlo, CML |
| [backtest.md](backtest.md) | rebalancing simulation |
| [walkforward.md](walkforward.md) | out-of-sample evaluation |
| [black_litterman.md](black_litterman.md) | posterior expected returns |
| [report.md](report.md) | tables, exports, console report |
| [cli.md](cli.md) | flags and pipeline stages |
| [plotting.md](plotting.md) | the 30-chart gallery |

## Conventions

- All return values are pandas objects indexed by ticker or date unless
  a plain `float`/`np.ndarray` is documented.
- All angles are **annualized** with 252 trading days.
- Every function validates its inputs and raises `ValueError` with a
  message naming the offending parameter.

## Error handling

- `mpt.data` raises `RuntimeError` when Yahoo returns nothing usable.
- `mpt.optimizer` raises `RuntimeError` when SLSQP cannot meet a
  target return, `ValueError` for unknown objectives/budgets.
- `mpt.config.OptimizerConfig.validate()` raises `ValueError` before
  any network or optimizer work happens.
