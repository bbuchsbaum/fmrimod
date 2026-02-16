"""Nuisance projection via QR decomposition (Frisch-Waugh-Lovell theorem).

Provides memory-efficient projection that avoids forming the full
T x T projection matrix.  Shared by all single-trial estimation methods.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def project_nuisance(
    X_nuisance: NDArray[np.float64],
    *targets: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], ...]:
    """Project out the column space of *X_nuisance* from each target matrix.

    Uses the thin QR decomposition so the cost is O(n * p * V) rather
    than O(n^2) for forming the full projection matrix.

    Parameters
    ----------
    X_nuisance : NDArray, shape ``(n, p)``
        Nuisance regressor matrix (confounds, drift, etc.).
    *targets : NDArray
        One or more matrices of shape ``(n, ...)``.  Each is
        residualised with respect to *X_nuisance*.

    Returns
    -------
    tuple of NDArray
        Residualised versions of each *target*, in the same order.

    Examples
    --------
    >>> import numpy as np
    >>> n, p, V = 100, 3, 50
    >>> Z = np.random.randn(n, p)
    >>> Y = np.random.randn(n, V)
    >>> X = np.random.randn(n, 10)
    >>> Y_clean, X_clean = project_nuisance(Z, Y, X)
    >>> Y_clean.shape
    (100, 50)
    """
    if X_nuisance.ndim != 2:
        raise ValueError(
            f"X_nuisance must be 2-D, got shape {X_nuisance.shape}"
        )
    if X_nuisance.shape[1] == 0:
        return targets if len(targets) > 1 else targets[0]

    # Thin QR: Q is (n, p), avoids forming (n, n)
    Q, _ = np.linalg.qr(X_nuisance, mode="reduced")

    results = []
    for M in targets:
        # M_clean = M - Q @ (Q' @ M)
        results.append(M - Q @ (Q.T @ M))

    return tuple(results) if len(results) > 1 else results[0]
