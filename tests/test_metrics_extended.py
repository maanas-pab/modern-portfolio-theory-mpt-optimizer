"""Edge cases and property tests for mpt.metrics."""
import numpy as np
import pandas as pd
import pytest

from mpt.metrics import (alpha, beta, calmar_ratio, concentration_hhi,
                         conditional_var, downside_deviation, max_drawdown,
                         portfolio_performance, risk_budget_loss,
                         risk_contributions, sharpe_ratio, sortino_ratio,
                         value_at_risk)

def test_sharpe_negative():
    assert np.isclose(sharpe_ratio(-0.05, 0.20, 0.02), -0.35)


def test_sharpe_zero_return():
    assert np.isclose(sharpe_ratio(0.0, 0.10, 0.02), -0.2)


