"""Preprocessing steps applied before GLM fitting.

Includes volume weighting, soft subspace projection, and censoring.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import optimize


def apply_volume_weights(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply per-volume (scan) weights to design and data matrices.

    Multiplies each row by ``sqrt(w_i)`` so that subsequent OLS
    is equivalent to WLS with diagonal weight matrix.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    weights : NDArray
        Non-negative weight vector, shape ``(n,)``.

    Returns
    -------
    X_w, Y_w : tuple of NDArray
        Weighted design and data matrices.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2-D matrices")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if weights.ndim != 1:
        raise ValueError("weights must be a 1-D array")
    if weights.shape[0] != X.shape[0]:
        raise ValueError(
            f"weights length {weights.shape[0]} does not match matrix rows {X.shape[0]}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    w_sqrt = np.sqrt(np.maximum(weights, 0.0))
    X_w = X * w_sqrt[:, np.newaxis]
    Y_w = Y * w_sqrt[:, np.newaxis]
    return X_w, Y_w


def compute_dvars(
    Y: NDArray[np.float64],
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Compute DVARS (temporal derivative of voxelwise variance).

    Parity with fmrireg:
    - first timepoint is set to median of subsequent DVARS values
    - optional normalization by median DVARS (enabled by default)

    Parameters
    ----------
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    normalize : bool
        If ``True``, divide by median DVARS (fmrireg default).

    Returns
    -------
    NDArray
        DVARS values, shape ``(n,)``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.ndim != 2:
        raise ValueError("Y must be a 1-D or 2-D array")

    n = Y.shape[0]
    if n < 2:
        raise ValueError("DVARS requires at least 2 timepoints")

    # Temporal derivative: Y[t] - Y[t-1]
    dY = np.diff(Y, axis=0)
    # RMS across voxels for each timepoint
    dvars_raw = np.sqrt(np.mean(dY ** 2, axis=1))

    # First timepoint has no derivative; use median of subsequent values
    dvars = np.concatenate([[np.median(dvars_raw)], dvars_raw])

    if normalize:
        med = np.median(dvars)
        if med > 0:
            dvars = dvars / med

    return dvars


def dvars_weights(
    dvars: NDArray[np.float64],
    method: str = "inverse_squared",
    threshold: float = 1.5,
    steepness: float = 2.0,
) -> NDArray[np.float64]:
    """Convert DVARS into per-volume weights.

    Parameters
    ----------
    dvars : NDArray
        DVARS values, shape ``(n,)``.
    method : str
        ``"inverse_squared"``, ``"soft_threshold"``, or ``"tukey"``.
    threshold : float
        Threshold in MAD units.
    steepness : float
        Decay steepness for ``"soft_threshold"`` method.

    Returns
    -------
    NDArray
        Weights, shape ``(n,)``.
    """
    dvars = np.asarray(dvars, dtype=np.float64).ravel()
    if np.any(dvars < 0):
        raise ValueError("DVARS values must be non-negative")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    if method == "inverse_squared":
        # w = 1 / (1 + dvars^2)
        w = 1.0 / (1.0 + dvars ** 2)
    elif method == "soft_threshold":
        # Sigmoid-like decay above threshold
        w = 1.0 / (
            1.0 + ((np.maximum(dvars, threshold) - threshold) / threshold) ** steepness
        )
    elif method == "tukey":
        c_tukey = threshold * 2.0
        u = dvars / c_tukey
        w = np.where(np.abs(u) <= 1.0, (1.0 - u ** 2) ** 2, 0.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure weights are in [0, 1], then normalize mean to ~1
    w = np.clip(w, 0.0, 1.0)
    mean_w = np.mean(w)
    if mean_w > 0:
        w = w / mean_w
    return w


def apply_censoring(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    censor: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Remove censored volumes from design and data.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    censor : NDArray[bool]
        Boolean vector, shape ``(n,)``.  ``True`` = censored (excluded).

    Returns
    -------
    X_clean, Y_clean, keep_mask : tuple
        Subsetted matrices and the boolean keep mask.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    censor_arr = np.asarray(censor)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2-D matrices")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if censor_arr.ndim != 1 or censor_arr.shape[0] != X.shape[0]:
        raise ValueError(
            f"Censor vector length {censor_arr.shape[0]} does not match matrix rows {X.shape[0]}"
        )

    if np.issubdtype(censor_arr.dtype, np.bool_):
        censor_mask = censor_arr
    elif np.issubdtype(censor_arr.dtype, np.number):
        if not np.all(np.isfinite(censor_arr)):
            raise ValueError("Censor vector must contain finite values")
        if not np.all(np.isin(censor_arr, [0, 1])):
            raise ValueError("Censor vector must be boolean or binary (0/1)")
        censor_mask = censor_arr.astype(bool)
    else:
        raise ValueError("Censor vector must be boolean or binary (0/1)")

    keep = ~censor_mask
    return X[keep], Y[keep], keep


def extract_nuisance_timeseries(
    Y: NDArray[np.float64],
    nuisance_mask: object,
    dataset_mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float64]:
    """Extract nuisance timeseries columns from *Y* using a mask spec.

    Parameters
    ----------
    Y : NDArray
        Data matrix of shape ``(n_time, n_voxels_in_data)``.
    nuisance_mask : object
        Nuisance mask specification:
        - 1-D boolean/numeric vector
        - 3-D boolean/numeric array
        - path to a NIfTI mask file (requires nibabel)
    dataset_mask : NDArray[bool], optional
        Full dataset spatial mask (typically 3-D). When provided and
        ``nuisance_mask`` is in full-volume space, it is mapped into
        in-data voxel space via ``nuisance_mask[dataset_mask]``.

    Returns
    -------
    NDArray
        Nuisance matrix of shape ``(n_time, n_nuisance_voxels)``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2-D matrix")

    mask_obj = nuisance_mask
    if isinstance(mask_obj, str):
        try:
            import nibabel as nib  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "nuisance_mask path support requires nibabel to be installed"
            ) from exc
        mask_arr = np.asarray(nib.load(mask_obj).get_fdata(), dtype=bool)
    else:
        mask_arr = np.asarray(mask_obj)

    if mask_arr.ndim == 0:
        raise ValueError("nuisance_mask must be a vector, 3-D mask, or mask file path")
    if mask_arr.ndim not in (1, 3):
        raise ValueError("nuisance_mask must be a 1-D vector or 3-D array")

    mask_vec = mask_arr.astype(bool).ravel()

    if mask_vec.shape[0] == Y.shape[1]:
        in_data_mask = mask_vec
    else:
        if dataset_mask is None:
            raise ValueError(
                "nuisance_mask length does not match data voxels and no dataset_mask was provided"
            )
        ds_mask_vec = np.asarray(dataset_mask, dtype=bool).ravel()
        if mask_vec.shape[0] != ds_mask_vec.shape[0]:
            raise ValueError(
                f"nuisance_mask length {mask_vec.shape[0]} does not match dataset mask size {ds_mask_vec.shape[0]}"
            )
        if int(np.sum(ds_mask_vec)) != Y.shape[1]:
            raise ValueError(
                "dataset mask voxel count does not match number of data columns"
            )
        in_data_mask = mask_vec[ds_mask_vec]

    if not np.any(in_data_mask):
        raise ValueError("nuisance_mask selected zero nuisance voxels")
    return Y[:, in_data_mask]


def _select_lambda_gcv(
    U: NDArray[np.float64],
    d2: NDArray[np.float64],
    Y: NDArray[np.float64],
) -> float:
    """Select soft-projection lambda via GCV (fmrireg parity)."""
    n = float(U.shape[0])
    UY = U.T @ Y

    def gcv_score(log_lambda: float) -> float:
        lam = float(np.exp(log_lambda))
        shrink = d2 / (d2 + lam)
        df = float(np.sum(shrink))
        y_hat = U @ (shrink[:, np.newaxis] * UY)
        rss = float(np.sum((Y - y_hat) ** 2))
        denom = (1.0 - df / n) ** 2
        if denom < 1e-10:
            return np.inf
        return rss / denom

    d2_pos = d2[d2 > 0]
    if d2_pos.size == 0:
        return 0.0
    lo = float(np.log(np.min(d2_pos) / 100.0))
    hi = float(np.log(np.max(d2_pos) * 100.0))
    opt = optimize.minimize_scalar(gcv_score, bounds=(lo, hi), method="bounded")
    return float(np.exp(opt.x))


def _resolve_soft_lambda(
    d2: NDArray[np.float64],
    lam: Union[float, str],
    Y: Optional[NDArray[np.float64]],
    U: NDArray[np.float64],
) -> float:
    """Resolve soft-projection lambda from numeric/'auto'/'gcv' spec."""
    if isinstance(lam, str):
        if lam not in ("auto", "gcv"):
            raise ValueError("lambda must be a non-negative number, 'auto', or 'gcv'")
        if d2.size == 0:
            return 0.0
        if lam == "auto":
            return float(np.median(d2))
        if Y is None:
            warnings.warn("GCV requires Y; falling back to auto", RuntimeWarning)
            return float(np.median(d2))
        return _select_lambda_gcv(U, d2, Y)

    lam_val = float(lam)
    if lam_val < 0:
        raise ValueError("lambda must be non-negative")
    return lam_val


def soft_subspace_projection(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    lam: Union[float, str],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply soft subspace projection to remove nuisance signals.

    Projects both X and Y onto the space orthogonal to the nuisance
    subspace, with soft (ridge-regularised) shrinkage controlled by *lam*.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    nuisance : NDArray
        Nuisance matrix, shape ``(n, q)``.
    lam : float or {"auto", "gcv"}
        Regularisation parameter. 0 = hard projection, larger = softer.
        String options follow fmrireg semantics.

    Returns
    -------
    X_proj, Y_proj : tuple of NDArray
        Projected matrices.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    N = np.asarray(nuisance, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2 or N.ndim != 2:
        raise ValueError("X, Y, and nuisance must all be 2-D matrices")
    if X.shape[0] != Y.shape[0] or X.shape[0] != N.shape[0]:
        raise ValueError("X, Y, and nuisance must have matching row counts")
    if N.shape[1] == 0:
        return X.copy(), Y.copy()

    # SVD-based form mirrors fmrireg::soft_projection implementation.
    U, s, _ = np.linalg.svd(N, full_matrices=False)
    d2 = s ** 2
    lam_val = _resolve_soft_lambda(d2, lam, Y=Y, U=U)
    shrink = d2 / (d2 + lam_val)

    UX = U.T @ X
    UY = U.T @ Y
    X_proj = X - U @ (shrink[:, np.newaxis] * UX)
    Y_proj = Y - U @ (shrink[:, np.newaxis] * UY)
    return X_proj, Y_proj
