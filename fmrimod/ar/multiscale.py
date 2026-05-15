"""Multi-scale spatial pooling for AR estimation.

Ports ``multiscale.R`` and ``multiscale_fast.R`` into pure NumPy.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .numhelpers import (
    ar_to_pacf,
    enforce_stationary_ar,
    levinson_durbin,
    pacf_to_ar,
    segmented_acvf,
)

# ---------------------------------------------------------------------------
# Parcel-level helpers
# ---------------------------------------------------------------------------

def parcel_means(
    resid: NDArray[Any], parcels: NDArray[Any], na_rm: bool = False
) -> Dict[str, NDArray[Any]]:
    """Compute parcel-averaged time series.

    Parameters
    ----------
    resid : NDArray[Any]
        Residual matrix, shape ``(n, V)``.
    parcels : NDArray[Any]
        Parcel labels, shape ``(V,)``.
    na_rm : bool
        If ``True``, skip NaN values.

    Returns
    -------
    dict
        Mapping from parcel id (str) to mean time series (n,).
    """
    resid = np.asarray(resid, dtype=np.float64)
    parcels = np.asarray(parcels, dtype=np.intp)
    n = resid.shape[0]
    ids = np.unique(parcels)

    result = {}
    for pid in ids:
        cols = np.where(parcels == pid)[0]
        if len(cols) == 0:
            continue
        sub = resid[:, cols]
        if na_rm:
            result[str(pid)] = np.nanmean(sub, axis=1)
        else:
            result[str(pid)] = sub.mean(axis=1)

    return result


def ms_dispersion(resid: NDArray[Any], parcels: NDArray[Any]) -> Dict[str, float]:
    """Compute voxel-variance dispersion (MAD) within each parcel.

    Parameters
    ----------
    resid : NDArray[Any]
        Residual matrix, shape ``(n, V)``.
    parcels : NDArray[Any]
        Parcel labels, shape ``(V,)``.

    Returns
    -------
    dict
        Mapping from parcel id (str) to dispersion value.
    """
    parcels = np.asarray(parcels, dtype=np.intp)
    vvar = np.var(resid, axis=0)
    ids = np.unique(parcels)
    result = {}
    for pid in ids:
        cols = np.where(parcels == pid)[0]
        if len(cols) == 0:
            result[str(pid)] = 0.0
            continue
        v = vvar[cols]
        mad = float(np.median(np.abs(v - np.median(v))))
        result[str(pid)] = mad
    return result


def ms_weights(
    n_t: int,
    n_runs: int,
    sizes: NDArray[Any],
    disp: NDArray[Any],
    beta: float = 0.5,
    eps: float = 1e-8,
) -> NDArray[Any]:
    """Compute multi-scale combination weights.

    Parameters
    ----------
    n_t : int
        Number of timepoints.
    n_runs : int
        Number of runs.
    sizes : NDArray[Any]
        Parcel sizes for each scale, shape ``(n_scales,)``.
    disp : NDArray[Any]
        Dispersion values for each scale, shape ``(n_scales,)``.
    beta : float
        Size exponent.
    eps : float
        Floor for numerical stability.

    Returns
    -------
    NDArray[Any]
        Normalised weights, shape ``(n_scales,)``.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    disp = np.asarray(disp, dtype=np.float64)
    s = (n_t * n_runs) * (sizes ** beta)
    h = 1.0 / (1.0 + np.maximum(disp, 0.0))
    w = s * h
    w = np.maximum(w, eps)
    return w


def _pad(x: NDArray[Any], length: int) -> NDArray[Any]:
    """Zero-pad or truncate *x* to *length*."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) >= length:
        return x[:length]
    return np.concatenate([x, np.zeros(length - len(x))])


# ---------------------------------------------------------------------------
# Parent maps for nested parcellations
# ---------------------------------------------------------------------------

def ms_parent_maps(
    parcels_fine: NDArray[Any],
    parcels_medium: NDArray[Any],
    parcels_coarse: NDArray[Any],
) -> Dict[str, Dict[int, int]]:
    """Build parent lookup tables mapping fine parcels to medium/coarse.

    Uses majority vote within each fine parcel.

    Parameters
    ----------
    parcels_fine, parcels_medium, parcels_coarse : NDArray[Any]
        Parcel labels, each shape ``(V,)``.

    Returns
    -------
    dict
        ``{"parent_medium": {fine_id: med_id}, "parent_coarse": {fine_id: coarse_id}}``
    """
    pf = np.asarray(parcels_fine, dtype=np.intp)
    pm = np.asarray(parcels_medium, dtype=np.intp)
    pc = np.asarray(parcels_coarse, dtype=np.intp)

    fine_ids = np.unique(pf)
    parent_medium = {}
    parent_coarse = {}

    for fid in fine_ids:
        idx = np.where(pf == fid)[0]
        mids = pm[idx]
        cids = pc[idx]
        vals_m, counts_m = np.unique(mids, return_counts=True)
        vals_c, counts_c = np.unique(cids, return_counts=True)
        parent_medium[int(fid)] = int(vals_m[np.argmax(counts_m)])
        parent_coarse[int(fid)] = int(vals_c[np.argmax(counts_c)])

    return {"parent_medium": parent_medium, "parent_coarse": parent_coarse}


# ---------------------------------------------------------------------------
# Scale-level estimation
# ---------------------------------------------------------------------------

def ms_estimate_scale(
    M: Dict[str, NDArray[np.float64]],
    estimator: Callable[..., Any],
    run_starts: Optional[NDArray[Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Estimate AR at a single spatial scale.

    Parameters
    ----------
    M : dict
        Parcel means: ``{parcel_id: time_series}``.
    estimator : callable
        ``estimator(y) -> {"phi": NDArray[Any], "order": (p, q)}``.
    run_starts : NDArray[Any], optional
        0-based run start indices.

    Returns
    -------
    dict
        ``{"phi": {pid: NDArray[Any]}, "acvf": {pid: NDArray[Any]}}``
    """
    if run_starts is None:
        run_starts = np.array([0], dtype=np.intp)

    phi_by = {}
    acvf_by = {}
    for pid, y_col in M.items():
        fit = estimator(y_col)
        phi_by[pid] = np.asarray(fit["phi"], dtype=np.float64)
        lag_max = max(0, fit["order"][0] + 1) if "order" in fit else len(fit["phi"]) + 1
        acvf_by[pid] = segmented_acvf(y_col, run_starts, lag_max)

    return {"phi": phi_by, "acvf": acvf_by}


# ---------------------------------------------------------------------------
# Multi-scale combination
# ---------------------------------------------------------------------------

def ms_combine_to_fine(
    phi_by_coarse: Dict[str, NDArray[Any]],
    phi_by_medium: Dict[str, NDArray[Any]],
    phi_by_fine: Dict[str, NDArray[Any]],
    acvf_by_coarse: Optional[Dict[str, NDArray[Any]]] = None,
    acvf_by_medium: Optional[Dict[str, NDArray[Any]]] = None,
    acvf_by_fine: Optional[Dict[str, NDArray[Any]]] = None,
    parents: Optional[Dict[str, Any]] = None,
    sizes: Optional[Dict[str, Any]] = None,
    disp_list: Optional[Dict[str, Any]] = None,
    p_target: int = 6,
    mode: str = "pacf_weighted",
    kappa_clip: float = 0.99,
) -> Dict[str, NDArray[np.float64]]:
    """Combine AR estimates across spatial scales for each fine parcel.

    Parameters
    ----------
    phi_by_coarse, phi_by_medium, phi_by_fine : dict
        Per-parcel AR coefficients at each scale.
    acvf_by_coarse, acvf_by_medium, acvf_by_fine : dict, optional
        Per-parcel ACVF at each scale (required for ``"acvf_pooled"`` mode).
    parents : dict
        Parent mappings from :func:`ms_parent_maps`.
    sizes : dict
        Scale size information.
    disp_list : dict
        Per-scale dispersion values.
    p_target : int
        Target AR order.
    mode : str
        ``"pacf_weighted"`` or ``"acvf_pooled"``.
    kappa_clip : float
        PACF clipping bound.

    Returns
    -------
    dict
        Per-fine-parcel AR coefficients.
    """
    if mode not in ("pacf_weighted", "acvf_pooled"):
        raise ValueError(f"mode must be 'pacf_weighted' or 'acvf_pooled', got {mode!r}")
    if parents is None or sizes is None or disp_list is None:
        raise ValueError("parents, sizes, and disp_list are required")

    fine_ids = sorted(phi_by_fine.keys(), key=lambda x: int(x))
    out_phi: Dict[str, NDArray[np.float64]] = {}

    for fid in fine_ids:
        fid_int = int(fid)
        mid = parents["parent_medium"].get(fid_int, fid_int)
        cid = parents["parent_coarse"].get(fid_int, fid_int)
        key_f = str(fid)
        key_m = str(mid)
        key_c = str(cid)

        # Gather sizes and dispersions for the 3 scales
        size_vec = np.array([
            float(sizes["coarse"].get(key_c, 1)),
            float(sizes["medium"].get(key_m, 1)),
            float(sizes["fine"].get(key_f, 1)),
        ])
        disp_vec = np.array([
            float(disp_list["coarse"].get(key_c, 0)),
            float(disp_list["medium"].get(key_m, 0)),
            float(disp_list["fine"].get(key_f, 0)),
        ])

        w = ms_weights(
            sizes["n_t"], sizes["n_runs"], size_vec, disp_vec,
            beta=sizes.get("beta", 0.5),
        )
        w = w / w.sum()

        if mode == "pacf_weighted":
            kap_c = _pad(ar_to_pacf(phi_by_coarse.get(key_c, np.array([]))), p_target)
            kap_m = _pad(ar_to_pacf(phi_by_medium.get(key_m, np.array([]))), p_target)
            kap_f = _pad(ar_to_pacf(phi_by_fine.get(key_f, np.array([]))), p_target)

            kap = w[0] * kap_c + w[1] * kap_m + w[2] * kap_f
            kap = np.clip(kap, -kappa_clip, kappa_clip)
            out_phi[key_f] = pacf_to_ar(kap)
        else:
            if (
                acvf_by_coarse is None
                or acvf_by_medium is None
                or acvf_by_fine is None
            ):
                raise ValueError("acvf_pooled mode requires all three acvf maps")
            g_c = _pad(acvf_by_coarse.get(key_c, np.array([0.0])), p_target + 1)
            g_m = _pad(acvf_by_medium.get(key_m, np.array([0.0])), p_target + 1)
            g_f = _pad(acvf_by_fine.get(key_f, np.array([0.0])), p_target + 1)

            g = w[0] * g_c + w[1] * g_m + w[2] * g_f
            phi_est, _ = levinson_durbin(g, p_target)
            out_phi[key_f] = enforce_stationary_ar(phi_est)

    return out_phi
