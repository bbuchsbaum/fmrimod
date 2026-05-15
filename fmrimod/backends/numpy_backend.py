"""Default numpy/scipy solver backend."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


class NumpyBackend:
    """Default solver backend using numpy and scipy.

    This wraps the core linear algebra operations so they could
    be swapped out for a JAX or GPU backend later.
    """

    @staticmethod
    def qr(X: NDArray[np.float64]) -> tuple[Any, ...]:
        """QR decomposition with column pivoting."""
        return cast("tuple[Any, ...]", linalg.qr(X, pivoting=True, mode="economic"))

    @staticmethod
    def cholesky(A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cholesky decomposition (lower triangular)."""
        return cast("NDArray[np.float64]", linalg.cholesky(A, lower=True))

    @staticmethod
    def cho_solve(L: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``A x = b`` given lower Cholesky factor ``L``."""
        return cast("NDArray[np.float64]", linalg.cho_solve((L, True), b))

    @staticmethod
    def svd(X: NDArray[np.float64]) -> tuple[Any, ...]:
        """Thin SVD."""
        return cast("tuple[Any, ...]", linalg.svd(X, full_matrices=False))

    @staticmethod
    def solve(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``A x = b``."""
        return cast("NDArray[np.float64]", linalg.solve(A, b))

    @staticmethod
    def matmul(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
        """Matrix multiplication."""
        return cast("NDArray[np.float64]", A @ B)
