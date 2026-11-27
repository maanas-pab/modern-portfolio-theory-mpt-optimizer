"""
Modern Portfolio Theory (MPT) Optimizer - backwards-compatible entry point
==========================================================================
The original single-file script has been restructured into the ``mpt``
package.  This module keeps every old public name importable and keeps
``python mpt_optimizer.py`` working, delegating to :mod:`mpt.cli`.

Usage:
    python mpt_optimizer.py
    python mpt_optimizer.py --period 3y --risk-free 0.02 --points 60

New code should import from the package instead::

    from mpt import max_sharpe_weights, min_vol_weights, efficient_frontier
"""

import warnings

warnings.warn(
    "mpt_optimizer.py is deprecated; use `python -m mpt.cli` or the `mpt` "
    "package directly. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from mpt.metrics import portfolio_performance  # noqa: F401
from mpt.optimizer import (  # noqa: F401
    max_sharpe_weights,
    min_vol_weights,
    optimize_portfolio,
)
from mpt.frontier import (  # noqa: F401
    efficient_frontier,
    random_portfolios,
)
from mpt.report import weights_table  # noqa: F401
from mpt.stats import calculate_statistics  # noqa: F401
from mpt.data import fetch_prices  # noqa: F401
from mpt.plotting import plot_frontier  # noqa: F401
from mpt.config import DEFAULT_TICKERS, TRADING_DAYS  # noqa: F401

__all__ = [
    "DEFAULT_TICKERS",
    "TRADING_DAYS",
    "fetch_prices",
    "calculate_statistics",
    "portfolio_performance",
    "optimize_portfolio",
    "max_sharpe_weights",
    "min_vol_weights",
    "efficient_frontier",
    "random_portfolios",
    "weights_table",
    "plot_frontier",
    "main",
]

def main(argv=None) -> int:
    """Run the full pipeline via :mod:`mpt.cli` (legacy signature)."""
    from mpt.cli import main as _cli_main

    return _cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
