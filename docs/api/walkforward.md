# `mpt.walkforward`

```python
from mpt.walkforward import walk_forward, rolling_windows

wf = walk_forward(prices, train_days=252, test_days=63,
                  objective="sharpe", risk_free=0.02)
```

- `rolling_windows(index, train, test)` -> list of
  `(train_start, train_end, test_end)` triples sliding by `test_days`;
  the final partial window is dropped.
- `WalkForwardResult.equity` / `.returns` — stitched **out-of-sample**
  series; `.summary(risk_free)` returns annual return, vol, Sharpe,
  max drawdown and window count.
- `WalkForwardResult.weights_by_window` — the fitted weight vector for
  each window; `window_weight_frame(wf)` turns it into a DataFrame.

