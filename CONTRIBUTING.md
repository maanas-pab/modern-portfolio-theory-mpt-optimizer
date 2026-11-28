# Contributing

Thanks for considering a contribution!

## Development setup

```bash
git clone https://github.com/maanas-pab/modern-portfolio-theory-mpt-optimizer.git
cd modern-portfolio-theory-mpt-optimizer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Tests

The suite is fully offline: fixtures generate synthetic GBM price paths,
so no network or API key is needed.

```bash
make test          # or: python3 -m pytest tests/
```

Please add a test for every new function. Aim for the same granularity
as the existing suite - one behavior per test.

## Lint

```bash
make lint          # or: python3 -m ruff check mpt tests
```

## Charts

Charts live in `mpt/plotting/` and follow one convention:

1. one function per figure,
2. `out_path` keyword with a default under `figures/`,
3. `setup_style()` first, `finish()` last,
4. a smoke test in `tests/test_plotting.py`.

Regenerate the gallery with `make figures`.

