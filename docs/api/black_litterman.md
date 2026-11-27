# `mpt.black_litterman`

```python
from mpt.black_litterman import (equilibrium_returns, black_litterman,
                                 black_litterman_covariance, top_pick_views)
```

| Function | Purpose |
|----------|---------|
| `equilibrium_returns(cov, market_weights=None, risk_aversion=2.5)` | prior `π = δ·Σ·w` |
| `black_litterman(pi, cov, P, Q, tau=0.05, ...)` | posterior means |
| `black_litterman_covariance(cov, P, tau=0.05, ...)` | posterior Σ on the same annualized scale |
| `top_pick_views(expected, k=3, uplift=0.05)` | builds `P`, `Q` from the top-k estimates |

`Ω` (view uncertainty) can be given as:

1. `omega_scalar` — one fixed standard deviation for every view,
2. `confidences` — per-view confidence in `[0, 1]` mapped through
   Idzorek's formula,
3. omitted — the He-Litterman proportional default
   `ωᵢᵢ = pᵢ (τΣ) pᵢ'`.

Diffuse views (large Ω) collapse the posterior back to the prior;
confident views pull it toward `Q`.
