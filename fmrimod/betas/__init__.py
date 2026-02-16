"""Beta extraction and trial-wise estimation.

Provides OLS and LSS (Least Squares Separate) methods for
single-trial beta estimation.
"""

from .extraction import (
    BetaMethod,
    BetaResult,
    estimate_betas,
    estimate_betas_lss,
    estimate_betas_ols,
)

__all__ = [
    "BetaMethod",
    "BetaResult",
    "estimate_betas",
    "estimate_betas_lss",
    "estimate_betas_ols",
]
