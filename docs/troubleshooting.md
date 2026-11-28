# Troubleshooting

Common failure modes, what causes them, and the fix.

## Yahoo download fails or returns empty

Yahoo rate-limits aggressively. The pipeline retries per ticker and
falls back to the bundled cache. If you genuinely need fresh data,
run `--refresh` once and let `save_cache` rewrite
`data/prices_3y.csv`; subsequent runs are offline again.

