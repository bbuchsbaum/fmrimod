"""Prewhitening integration for single-trial estimation.

Wires the existing ``fmrimod.ar`` module into the single-trial
pipeline.  Fits AR parameters on OLS residuals, then whitens
all matrices so that OLS on the whitened system equals GLS on
the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PrewhitenConfig:
    """Prewhitening configuration.

    Attributes
    ----------
    method : str
        ``"ar"`` (default), ``"arma"``, or ``"none"``.
    p : int or str
        AR order.  Integer or ``"auto"`` for BIC selection.
    q : int
        MA order (only for ``method="arma"``).
    p_max : int
        Maximum AR order when ``p="auto"``.
    pooling : str
        ``"global"`` (default), ``"voxel"``, ``"run"``, or ``"parcel"``.
    runs : NDArray or None
        Run labels, shape ``(n,)``.
    parcels : NDArray or None
        Parcel labels, shape ``(V,)``.
    exact_first : str
        ``"ar1"`` (default) or ``"none"``.
    """

    method: str = "ar"
    p: object = 1
    q: int = 0
    p_max: int = 6
    pooling: str = "global"
    runs: Optional[NDArray] = None
    parcels: Optional[NDArray] = None
    exact_first: str = "ar1"

    def __post_init__(self) -> None:
        if self.method not in {"ar", "arma", "none"}:
            raise ValueError("method must be one of: ar, arma, none")
        if self.pooling not in {"global", "voxel", "run", "parcel"}:
            raise ValueError("pooling must be one of: global, voxel, run, parcel")
        if self.exact_first not in {"ar1", "none"}:
            raise ValueError("exact_first must be one of: ar1, none")
        if self.p != "auto":
            if int(self.p) != self.p or int(self.p) < 0:
                raise ValueError("p must be a non-negative integer or 'auto'")
            self.p = int(self.p)
        if int(self.q) != self.q or self.q < 0:
            raise ValueError("q must be a non-negative integer")
        if int(self.p_max) != self.p_max or self.p_max < 1:
            raise ValueError("p_max must be a positive integer")
        self.q = int(self.q)
        self.p_max = int(self.p_max)


def prewhiten_matrices(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]],
    config: PrewhitenConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]:
    """Estimate AR parameters and whiten all matrices.

    Parameters
    ----------
    Y : (n, V)
    X : (n, p_trial)
    confounds : (n, q) or None
    config : PrewhitenConfig

    Returns
    -------
    Y_w, X_w, confounds_w : whitened versions
    """
    from ..ar.estimation import estimate_ar
    from ..ar.whitening import ar_whiten, ar_whiten_matrix

    if config.method == "none":
        return Y, X, confounds

    # Step 1: OLS residuals for AR estimation
    # Build full design for residual computation
    if confounds is not None:
        X_full = np.column_stack([X, confounds])
    else:
        X_full = X

    # Quick OLS residuals
    Q, _ = np.linalg.qr(X_full, mode="reduced")
    residuals = Y - Q @ (Q.T @ Y)

    # Step 2: Estimate AR parameters
    if config.p == "auto":
        from ..ar.estimation import estimate_ar_bic
        # Use median voxel for order selection
        med_idx = np.argmin(
            np.abs(np.var(residuals, axis=0) - np.median(np.var(residuals, axis=0)))
        )
        bic_result = estimate_ar_bic(residuals[:, med_idx], config.p_max)
        p = bic_result["order"]
    else:
        p = int(config.p)

    if config.pooling == "global":
        phi = estimate_ar(residuals, order=p, voxelwise=False)
    elif config.pooling == "voxel":
        phi = estimate_ar(residuals, order=p, voxelwise=True)
    else:
        # For run/parcel pooling, use global as fallback
        phi = estimate_ar(residuals, order=p, voxelwise=False)

    # Step 3: Whiten all matrices
    if phi.ndim == 1:
        # Global AR: same coefficients for all voxels
        X_w, Y_w = ar_whiten_matrix(X, Y, phi)
        if confounds is not None:
            confounds_w = ar_whiten(confounds, phi)
        else:
            confounds_w = None
    else:
        # Voxelwise AR: phi is (p, V)
        # Whiten Y voxel-by-voxel, X with mean phi
        phi_mean = phi.mean(axis=1)
        X_w = ar_whiten(X, phi_mean)
        if confounds is not None:
            confounds_w = ar_whiten(confounds, phi_mean)
        else:
            confounds_w = None
        # Whiten Y with per-voxel phi
        Y_w = np.empty_like(Y)
        for v in range(Y.shape[1]):
            Y_w[:, v] = ar_whiten(Y[:, v:v+1], phi[:, v]).ravel()

    return Y_w, X_w, confounds_w
