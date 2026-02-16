"""BIDS-Stats-Model export for GLM results.

Requires ``nibabel`` for NIfTI output.
"""

from .export import (
    BidsEntities,
    write_betas,
    write_contrasts,
)

__all__ = [
    "BidsEntities",
    "write_betas",
    "write_contrasts",
]
