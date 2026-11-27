# Chart Gallery

Every figure the pipeline generates, what it shows, and which flag
controls it. Regenerate the whole gallery with:

```bash
make figures
```

## Frontier family

![Efficient frontier](../figures/efficient_frontier.png)

**Efficient frontier** — 5,000 random portfolios (colored by Sharpe),
the traced frontier, the capital market line from the risk-free rate
through the tangency portfolio, and all 12 assets in risk-return space.

![Clean frontier](../figures/frontier_clean.png)

**Anchor portfolios** — max Sharpe, min volatility and equal weight
on a minimal frontier, for slides and docs.

![Sharpe distribution](../figures/sharpe_distribution.png)

**Sharpe distribution** — histogram of the 5,000 random portfolios
with the optimized maximum marked. Shows how rare a top-decile
portfolio is by chance alone.

![Frontier shrinkage](../figures/frontier_shrinkage.png)

**Frontier sensitivity** — the same expected returns traced against the
sample covariance and against the Ledoit-Wolf shrunk covariance. The
gap between the two lines is how much the optimizer's answer depends
on a noisy covariance estimate.

## Risk structure

![Correlation heatmap](../figures/correlation_heatmap.png)

**Correlation matrix** — daily-return correlations. CRSP/BEAM (0.72)
and NET/DDOG (0.52) are the diversification failures; most pairs sit
below 0.30, which is why the frontier bends so much.

![Covariance heatmap](../figures/covariance_heatmap.png)

**Covariance matrix** — annualized covariances feeding the optimizer.

![Correlation vs volatility](../figures/correlation_vs_vol.png)

**Diversification vs risk** — average correlation of each asset
against its own volatility: the upper-right corner is the worst place
to be (high vol, high correlation).

![Asset risk-return](../figures/asset_risk_return.png)

**Asset risk-return** — every ticker labeled, before any optimization.

## Distributions

![Return distributions](../figures/return_distributions.png)

**Return distributions** — per-asset daily histograms with mean lines;
note the fat tails in CELH, AFRM and BEAM.

![VaR/CVaR](../figures/var_cvar.png)

**VaR & expected shortfall** — the loss tail of the max-Sharpe
portfolio: 95% VaR and the average loss beyond it.

![Monthly returns](../figures/monthly_returns.png)

**Monthly returns** — year-by-year calendar heat for the max-Sharpe
portfolio.

![Asset monthly heatmap](../figures/asset_monthly_heatmap.png)

**Asset returns by month** — which assets carry which months.

## Calendar views

![Quarterly returns](../figures/quarterly_returns.png)

**Quarterly returns** — compounded return per calendar quarter, green
for gains and red for losses.

![Annual returns](../figures/annual_returns.png)

**Calendar-year returns** — every strategy's compounded return per
year, side by side.

## Allocations

![Weights comparison](../figures/weights_comparison.png)

**Allocation comparison** — grouped bars across max Sharpe, min
volatility and equal weight.

![Weight pies](../figures/weights_pies.png)

**Weight breakdown** — pies with a top-6 + other rollup.

![Risk contribution](../figures/risk_contribution.png)

**Risk contributions** — percent of total portfolio risk per asset
for the max-Sharpe portfolio, against the equal-risk line. PLTR and
NET dominate both weight *and* risk.

![Sector exposure](../figures/sector_exposure.png)

**Sector exposure** — weights aggregated by theme.

![Weight drift](../figures/weights_over_time.png)

**Weight drift** — stacked weights over a monthly-rebalanced
backtest; white lines mark rebalance events.

![Concentration](../figures/concentration_over_time.png)

**Concentration over time** — the Herfindahl-Hirschman index of the
weight vector through the backtest, against the equal-weight floor.
It rises as the optimizer piles into PLTR/NET and drops at rebalances.

![Weight trajectories](../figures/weight_trajectories.png)

**Weight trajectories** — line chart of the top holdings, showing
individual drift paths a stacked area hides.

