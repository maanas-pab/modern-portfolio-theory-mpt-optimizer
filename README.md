# Modern Portfolio Theory (MPT) Optimizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/maanas-pab/modern-portfolio-theory-mpt-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/maanas-pab/modern-portfolio-theory-mpt-optimizer/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-293%20passing-brightgreen.svg)](#testing)
[![Charts](https://img.shields.io/badge/charts-30-ff69b4.svg)](docs/charts.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Find the **exact asset weights that maximize the Sharpe Ratio** across a universe of 12 niche growth stocks — trace the full **Efficient Frontier** with `scipy.optimize`, stress-test it with a **rebalancing backtest**, validate it honestly with **walk-forward out-of-sample windows**, and explore **30 charts** of risk, return, and correlation.

![Efficient Frontier](figures/efficient_frontier.png)

## Highlights

- **Live or cached data** — pulls 3 years of daily prices via `yfinance`, or runs fully offline from the bundled `data/prices_3y.csv`
- **Five strategies** — Max Sharpe, Min Volatility, **Risk Parity** (equal risk budgeting), Equal Weight, and an optional **Black-Litterman** max-Sharpe, all compared against buy-and-hold
- **Honest evaluation** — `--walk-forward` re-fits on rolling train windows and reports only out-of-sample returns; the in-sample Sharpe of 1.42 becomes **0.37 out-of-sample**, and the gap is printed every run
- **30 charts** — frontier with capital market line, correlation & covariance heatmaps, cumulative growth, underwater drawdowns, rolling volatility/Sharpe/beta, monthly/quarterly/annual views, VaR/CVaR, risk contributions, turnover, concentration drift, sector exposure, weight drift — [see the full gallery](docs/charts.md)
- **15+ risk metrics** — Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR, beta, alpha, HHI concentration and more
- **Ledoit-Wolf shrinkage** — `--shrinkage` conditions the covariance matrix before optimizing
- **Backtesting engine** — weekly/monthly/quarterly rebalancing with transaction costs
- **293 offline tests + CI** — synthetic fixtures, GitHub Actions across Python 3.9–3.12

