"""
Black-Litterman expected returns
================================
Combines the market-implied equilibrium returns with investor views to
produce posterior expected returns that are far less noisy than raw
sample means.

The standard formulation (He & Litterman 1999, Idzorek 2004)::

    pi      = delta * Sigma * w_mkt          # equilibrium excess returns
    mu_BL   = [ (tau*Sigma)^-1 + P' Omega^-1 P ]^-1
              * [ (tau*Sigma)^-1 pi + P' Omega^-1 Q ]

where ``P`` is the pick matrix (n_views x n_assets), ``Q`` the view
returns, ``tau`` the prior scaling, and ``Omega`` the view uncertainty.

"""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# risk aversion used to turn covariance into equilibrium excess returns
DEFAULT_DELTA = 2.5

# Idzorek's closed form for Omega given confidence in [0, 1]
CONFIDENCE_TO_TAU = {0.0: 1.0, 0.25: 0.75, 0.5: 0.5, 0.75: 0.25, 1.0: 0.0}


def equilibrium_returns(cov: pd.DataFrame,
                        market_weights: Optional[Sequence[float]] = None,
                        risk_aversion: float = DEFAULT_DELTA) -> pd.Series:
    """
    ``pi = delta * Sigma * w`` — the returns the market is already pricing in.

    With no weights supplied, equal weights are used (the standard
    "unconstrained prior" for a personal-portfolio exercise).
    """
    n = len(cov)
    w = np.repeat(1.0 / n, n) if market_weights is None else np.asarray(
        market_weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"market_weights must have {n} entries")
    pi = risk_aversion * cov.values @ w
    return pd.Series(pi, index=cov.index, name="equilibrium")


def _as_pick_matrix(P, n_assets: int) -> np.ndarray:
    """Accept a 1-D view (single asset) or a 2-D pick matrix."""
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        if P.shape[0] != n_assets:
            raise ValueError("pick vector length must match the universe")
        return P.reshape(1, -1)
    if P.ndim != 2 or P.shape[1] != n_assets:
        raise ValueError("P must have shape (n_views, n_assets)")
    return P


