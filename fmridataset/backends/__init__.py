"""Compatibility re-exports for storage backends."""

from fmrimod.dataset.backends import (
    BidsH5ScanBackend,
    InMemoryLatentBackend,
    LatentBackend,
    MatrixBackend,
    SharedH5Connection,
    bids_h5_scan_backend,
)

__all__ = [
    "BidsH5ScanBackend",
    "InMemoryLatentBackend",
    "LatentBackend",
    "MatrixBackend",
    "SharedH5Connection",
    "bids_h5_scan_backend",
]
