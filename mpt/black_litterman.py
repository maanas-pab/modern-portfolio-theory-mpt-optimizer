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


def _view_uncertainty(P: np.ndarray, tau: float, cov: np.ndarray,
                      confidences: Optional[Sequence[float]] = None,
                      omega_scalar: Optional[float] = None) -> np.ndarray:
    """
    Build the diagonal Omega variances (one per view).

    Three ways to specify it, in order of preference:

    1. ``omega_scalar`` — a fixed standard deviation for every view,
    2. ``confidences`` — per-view confidence in [0, 1], mapped through
       Idzorek's ``omega_ii = (1/c - 1) * tau * p_i Sigma p_i'``,
    3. default — the He-Litterman proportional scheme
       ``omega_ii = p_i (tau Sigma) p_i'``.
    """
    base = np.einsum("ij,jk,ik->i", P, cov, P)  # p_i Sigma p_i'
    if omega_scalar is not None:
        if omega_scalar <= 0:
            raise ValueError("omega_scalar must be positive")
        return np.full(len(base), omega_scalar ** 2)
    if confidences is not None:
        conf = np.asarray(confidences, dtype=float)
        if conf.shape != (len(base),):
            raise ValueError("confidences must have one entry per view")
        if (conf < 0).any() or (conf > 1).any():
            raise ValueError("confidences must lie in [0, 1]")
        # c = 1 / (1 + omega/(tau*base))  =>  omega = (1/c - 1) * tau * base
        with np.errstate(divide="ignore"):
            scale = np.where(conf > 0, 1.0 / conf - 1.0, np.inf)
        omega = scale * tau * base
        # zero-confidence views fall back to a very diffuse (but finite) variance
        return np.where(np.isfinite(omega), omega, max(tau * base.max(), 1.0) * 1e6)
    return tau * base


