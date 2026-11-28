# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [2.0.0] - 2026-11-27

### Added
- Full `mpt/` package: config, data, stats, metrics, optimizer, frontier,
  backtest, walkforward, black_litterman, report, cli modules
- 30-chart gallery: frontier with capital market line, correlation and
  covariance heatmaps, cumulative growth, underwater drawdown, rolling
  volatility/Sharpe/beta, monthly/quarterly/annual views, benchmark
  scatter, risk contributions, turnover, concentration drift, VaR/CVaR
  histogram, Sharpe distribution, frontier shrinkage sensitivity,
  walk-forward equity, sector exposure, weight drift, and more
- New metrics: Sortino, Calmar, max drawdown, historical/parametric VaR,
  CVaR, beta, alpha, HHI concentration, risk contributions
- Ledoit-Wolf covariance shrinkage (`--shrinkage`)
- Risk-parity (equal risk budgeting) portfolio
- Black-Litterman expected returns from equilibrium prior + top-pick
  views (`--black-litterman`)
- Walk-forward rolling train/test evaluation with no look-ahead
  (`--walk-forward --train --test`)
- Backtesting engine with weekly/monthly/quarterly rebalancing and
  transaction costs (`--rebalance`, `--cost`)
- Warm-start `x0` and long-short `budget="gross"` optimizer options
- 293 offline tests, GitHub Actions CI, ruff lint, Makefile, pyproject
- Cached price data for reproducible offline runs

