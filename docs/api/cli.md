# `mpt.cli`

```python
from mpt.cli import build_parser, config_from_args, run, main
```

| Function | Purpose |
|----------|---------|
| `build_parser()` | the `argparse` factory (single source of truth for flags) |
| `config_from_args(args)` | flags -> validated `OptimizerConfig` |
| `run(cfg)` | full pipeline -> artifact dict |
| `emit_report` / `emit_backtests` / `emit_walkforward` | console output |
| `emit_figures(result)` | draws the gallery, returns paths |
| `main(argv)` | end-to-end, returns the exit code |

`python3 -m mpt.cli`, the `mpt` console script (installed via
`pyproject`), and the legacy `python3 mpt_optimizer.py` all converge
on `main()`.

`run()` returns every artifact the emitters need — prices, daily
returns, covariance, weights, frontier, Monte-Carlo cloud, summaries,
backtests, the Black-Litterman posterior (when enabled) and the
walk-forward result (when enabled) — so notebooks can call it
directly instead of shelling out.
