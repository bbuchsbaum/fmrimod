"""Optional JAX solver backend for GPU-accelerated GLM fitting.

Requires ``jax`` and ``jaxlib`` (optional dependencies).
Install with: ``pip install fmrimod[jax]``

This backend mirrors the :class:`NumpyBackend` interface but uses
JAX's JIT-compiled linear algebra for GPU/TPU acceleration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _import_jax() -> tuple[Any, Any, Any]:
    """Lazily import JAX, raising a clear error if unavailable."""
    try:
        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]
        from jax import scipy as jsp
        return jax, jnp, jsp
    except ImportError as e:
        raise ImportError(
            "JAX is required for the JAX backend. "
            "Install with: pip install fmrimod[jax]"
        ) from e


class JaxBackend:
    """JAX-accelerated solver backend.

    Provides the same interface as :class:`~fmrimod.backends.NumpyBackend`
    but dispatches to JAX for GPU/TPU acceleration.  Arrays are
    automatically converted to/from JAX arrays at the boundary.

    Examples
    --------
    >>> backend = JaxBackend()
    >>> L = backend.cholesky(XtX)   # runs on GPU if available
    >>> betas = backend.cho_solve(L, XtY)
    """

    def __init__(self) -> None:
        self._jax, self._jnp, self._jsp = _import_jax()

    def _to_jax(self, arr: object) -> Any:
        return self._jnp.asarray(arr)

    def _to_numpy(self, arr: object) -> NDArray[np.float64]:
        return np.asarray(arr, dtype=np.float64)

    def qr(self, X: NDArray[np.float64]) -> tuple[Any, ...]:
        """QR decomposition (without pivoting — JAX limitation)."""
        X_j = self._to_jax(X)
        Q, R = self._jnp.linalg.qr(X_j, mode="reduced")
        return self._to_numpy(Q), self._to_numpy(R)

    def cholesky(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cholesky decomposition (lower triangular)."""
        A_j = self._to_jax(A)
        L = self._jnp.linalg.cholesky(A_j)
        return self._to_numpy(L)

    def cho_solve(
        self, L: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Solve ``A x = b`` given lower Cholesky factor ``L``."""
        L_j = self._to_jax(L)
        b_j = self._to_jax(b)
        x = self._jsp.linalg.cho_solve((L_j, True), b_j)
        return self._to_numpy(x)

    def svd(self, X: NDArray[np.float64]) -> tuple[Any, ...]:
        """Thin SVD."""
        X_j = self._to_jax(X)
        U, s, Vt = self._jnp.linalg.svd(X_j, full_matrices=False)
        return self._to_numpy(U), self._to_numpy(s), self._to_numpy(Vt)

    def solve(
        self, A: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Solve ``A x = b``."""
        A_j = self._to_jax(A)
        b_j = self._to_jax(b)
        x = self._jnp.linalg.solve(A_j, b_j)
        return self._to_numpy(x)

    def matmul(
        self, A: NDArray[np.float64], B: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Matrix multiplication."""
        A_j = self._to_jax(A)
        B_j = self._to_jax(B)
        return self._to_numpy(A_j @ B_j)
