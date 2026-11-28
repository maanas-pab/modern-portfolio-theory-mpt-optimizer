# Troubleshooting

Common failure modes, what causes them, and the fix.

## Yahoo download fails or returns empty

Yahoo rate-limits aggressively. The pipeline retries per ticker and
falls back to the bundled cache. If you genuinely need fresh data,
run `--refresh` once and let `save_cache` rewrite
`data/prices_3y.csv`; subsequent runs are offline again.

## `RuntimeError: could not reach target return`

The requested target sits outside the long-only feasible region —
no combination of the universe's assets can earn it without shorts.
Pick a target between `min(exp_returns)` and `max(exp_returns)`,
or use `budget="gross"`.

## Charts don't render

Pass `--no-plot` to skip the gallery entirely, or check that
`matplotlib` is installed in the environment running the CLI. The
gallery needs a writable `--figure-dir` (default `figures/`).

## Tests fail on a fresh clone

Run `pytest tests/` from the repo root so `data/prices_3y.csv` resolves.
The suite is fully offline — if it hangs, something is trying to
reach the network; check that `mpt.data.get_prices` is using the
cache path, not `--refresh`.

## Pandas `pct_change()` warnings

`daily_returns` intentionally forward-fills NaN gaps before taking
pct change (asserted in the test suite). Deprecation warnings from
pandas about the fill method are cosmetic and safe to ignore.
