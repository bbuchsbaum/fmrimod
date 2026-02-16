"""SBHM prepass: aggregate regression to get basis coefficients.

The prepass regresses aggregate regressors (sum of all trial regressors per
basis column) against the data to obtain average basis coefficients per voxel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def sbhm_prepass(
    Y: NDArray[np.float64],
    A_agg: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prepass regression for SBHM: fit aggregate regressors to data.

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
        Data matrix (time x voxels).
    A_agg : NDArray, shape ``(T, r)``
        Aggregate regressor matrix: sum of all trial regressors per basis column.
    confounds : NDArray, shape ``(T, q)``, optional
        Nuisance regressors to project out.

    Returns
    -------
    beta_bar : NDArray, shape ``(r, V)``
        Average basis coefficients per voxel.
    G : NDArray, shape ``(r, r)``
        Gram matrix ``A_agg.T @ A_agg`` (after nuisance projection).

    Notes
    -----
    This function solves the regression::

        Y = A_agg @ beta_bar + confounds @ gamma + E

    The Gram matrix ``G`` is cached for use in matching.

    Examples
    --------
    >>> import numpy as np
    >>> from fmrimod.single.sbhm import sbhm_prepass
    >>> Y = np.random.randn(100, 500)
    >>> A_agg = np.random.randn(100, 3)
    >>> beta_bar, G = sbhm_prepass(Y, A_agg)
    >>> beta_bar.shape
    (3, 500)
    >>> G.shape
    (3, 3)
    """
    Y = np.asarray(Y, dtype=np.float64)
    A_agg = np.asarray(A_agg, dtype=np.float64)

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if A_agg.ndim == 1:
        A_agg = A_agg[:, np.newaxis]

    T, V = Y.shape
    if A_agg.shape[0] != T:
        raise ValueError(
            f"Y has {T} timepoints, A_agg has {A_agg.shape[0]}."
        )

    r = A_agg.shape[1]

    # Project out confounds via QR
    Y_clean = Y.copy()
    A_clean = A_agg.copy()

    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.shape[0] != T:
            raise ValueError(
                f"confounds has {confounds.shape[0]} timepoints, expected {T}."
            )
        Q, _ = np.linalg.qr(confounds, mode="reduced")
        Y_clean -= Q @ (Q.T @ Y_clean)
        A_clean -= Q @ (Q.T @ A_clean)

    # Solve A_clean.T @ A_clean @ beta_bar = A_clean.T @ Y_clean
    G = A_clean.T @ A_clean  # (r, r)
    AtY = A_clean.T @ Y_clean  # (r, V)

    # Solve via Cholesky
    try:
        L = np.linalg.cholesky(G)
        beta_bar = np.linalg.solve(L, AtY)
        beta_bar = np.linalg.solve(L.T, beta_bar)
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        beta_bar = np.linalg.lstsq(G, AtY, rcond=None)[0]

    return beta_bar, G
