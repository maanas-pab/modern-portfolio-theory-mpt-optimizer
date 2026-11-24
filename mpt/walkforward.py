"""
Walk-forward (rolling out-of-sample) optimization
==================================================
The in-sample backtest in :mod:`mpt.backtest` chooses weights on the
*same* history it evaluates — results are look-ahead.  This module
closes that hole:

1. split the history into rolling **train** / **test** windows,
2. optimize on the train window only,
3. hold those weights through the next test window,
4. stitch the out-of-sample equity curve together.

The honest performance number is whatever the strategy earned on data
the optimizer had never seen.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import backtest
from .metrics import max_drawdown, sharpe_ratio
from .optimizer import max_sharpe_weights, min_vol_weights, risk_parity_weights
from .stats import calculate_statistics


