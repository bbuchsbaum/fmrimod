"""Nuisance projection via QR decomposition (Frisch-Waugh-Lovell theorem).

Provides memory-efficient projection that avoids forming the full
T x T projection matrix.  Shared by all single-trial estimation methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NuisanceProjector:
    """Precomputed nuisance projector based on a thin-QR basis."""

    Q: NDArray[np.float64]

    @property
    def n_rows(self) -> int:
        return int(self.Q.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.Q.shape[1])

    def project(self, *targets: NDArray[np.float64]) -> Tuple[NDArray[np.float64], ...]:
        """Project each target matrix off the nuisance subspace."""
        results = []
        for M in targets:
            if M.shape[0] != self.n_rows:
                raise ValueError(
                    f"Target has {M.shape[0]} rows, expected {self.n_rows}."
                )
            results.append(M - self.Q @ (self.Q.T @ M))
        return cast(
            "Tuple[NDArray[np.float64], ...]",
            tuple(results) if len(results) > 1 else results[0],
        )


def build_nuisance_projector(
    X_nuisance: NDArray[np.float64],
) -> Optional[NuisanceProjector]:
    """Build a reusable nuisance projector from a nuisance design matrix."""
    X_nuisance = np.asarray(X_nuisance, dtype=np.float64)
    if X_nuisance.ndim != 2:
        raise ValueError(
            f"X_nuisance must be 2-D, got shape {X_nuisance.shape}"
        )
    if X_nuisance.shape[1] == 0:
        return None

    # Rank-aware thin basis.  A plain QR decomposition with rank-deficient
    # nuisance matrices can include arbitrary null-space columns and overproject.
    U, s, _ = np.linalg.svd(X_nuisance, full_matrices=False)
    if s.size == 0:
        return None
    tol = np.finfo(np.float64).eps * max(X_nuisance.shape) * float(s[0])
    rank = int(np.sum(s > tol))
    if rank == 0:
        return None
    Q = U[:, :rank]
    return NuisanceProjector(Q=Q)


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
    projector = build_nuisance_projector(X_nuisance)
    if projector is None:
        return cast(
            "Tuple[NDArray[np.float64], ...]",
            targets if len(targets) > 1 else targets[0],
        )
    return projector.project(*targets)
