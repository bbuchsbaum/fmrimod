"""Spatially-aware FDR correction.

Implements the Structure-Adaptive Weighted Benjamini-Hochberg procedure
for fMRI data, where spatial grouping of voxels into blocks allows
local estimation of the null proportion (pi0) and adaptive weighting
of p-values.

References
----------
Benjamini, Y., & Hochberg, Y. (1997). Multiple hypotheses testing with
weights. *Scandinavian Journal of Statistics*, 24(3), 407-418.

Storey, J. D. (2002). A direct approach to false discovery rates.
*JRSS-B*, 64(3), 479-498.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpatialFdrResult:
    """Result of spatially-aware FDR correction.

    Attributes
    ----------
    reject : NDArray[bool], shape ``(V,)``
        Boolean mask of rejected hypotheses.
    qvalues : NDArray[float], shape ``(V,)``
        FDR-adjusted p-values (q-values).
    weights : NDArray[float], shape ``(V,)``
        Normalised adaptive weights per voxel.
    pi0_raw : NDArray[float], shape ``(n_groups,)``
        Raw pi0 estimates per group.
    pi0_smooth : NDArray[float], shape ``(n_groups,)``
        Smoothed pi0 estimates (after spatial averaging).
    threshold : float
        BH rejection threshold on weighted p-values.
    """

    reject: NDArray[np.bool_]
    qvalues: NDArray[np.float64]
    weights: NDArray[np.float64]
    pi0_raw: NDArray[np.float64]
    pi0_smooth: NDArray[np.float64]
    threshold: float


# -- Group construction --------------------------------------------------------

def create_3d_blocks(
    mask: NDArray[np.bool_],
    block_size: int = 5,
) -> tuple[NDArray[np.intp], int]:
    """Partition a 3-D mask into spatial blocks.

    Assigns each True voxel in *mask* to a block based on integer
    division of its (i, j, k) coordinates by *block_size*.

    Parameters
    ----------
    mask : NDArray[bool], shape ``(nx, ny, nz)``
        Binary brain mask.
    block_size : int
        Edge length of each cubic block in voxels.

    Returns
    -------
    group_ids : NDArray[int], shape ``(V,)``
        Block assignment for each in-mask voxel.
    n_groups : int
        Number of distinct blocks.
    """
    ijk = np.argwhere(mask)  # (V, 3)
    block_ijk = ijk // block_size  # integer block coordinates

    # Map (bx, by, bz) tuples to consecutive IDs
    unique_blocks, group_ids = np.unique(
        block_ijk, axis=0, return_inverse=True,
    )
    return group_ids.astype(np.intp), len(unique_blocks)


def create_group_neighbors(
    mask: NDArray[np.bool_],
    group_ids: NDArray[np.intp],
    n_groups: int,
    block_size: int = 5,
) -> list[list[int]]:
    """Build adjacency list for spatial groups.

    Two groups are neighbours if their block coordinates differ by at
    most 1 along each axis (26-connectivity).

    Parameters
    ----------
    mask : NDArray[bool], shape ``(nx, ny, nz)``
    group_ids : NDArray[int], shape ``(V,)``
    n_groups : int
    block_size : int

    Returns
    -------
    list of list of int
        ``neighbors[g]`` = list of group IDs adjacent to group *g*.
    """
    ijk = np.argwhere(mask)
    block_ijk = ijk // block_size
    unique_blocks = np.zeros((n_groups, 3), dtype=np.intp)

    # Recover the canonical block coordinate for each group
    for g in range(n_groups):
        idx = np.where(group_ids == g)[0][0]
        unique_blocks[g] = block_ijk[idx]

    neighbors: list[list[int]] = [[] for _ in range(n_groups)]
    for g in range(n_groups):
        for h in range(g + 1, n_groups):
            if np.all(np.abs(unique_blocks[g] - unique_blocks[h]) <= 1):
                neighbors[g].append(h)
                neighbors[h].append(g)

    return neighbors


# -- Pi0 estimation ------------------------------------------------------------

def estimate_pi0(
    p_values: NDArray[np.float64],
    tau: float = 0.5,
    min_pi0: float = 0.05,
) -> float:
    """Storey's pi0 estimator for a vector of p-values.

    Parameters
    ----------
    p_values : NDArray, shape ``(m,)``
    tau : float
        Threshold for counting null p-values.
    min_pi0 : float
        Minimum pi0 to return (prevents zero weights).

    Returns
    -------
    float
        Estimated proportion of true nulls. Missing p-values are not
        counted, and a group with none left gives 1.
    """
    p = np.asarray(p_values, dtype=np.float64)
    p = p[np.isfinite(p)]
    m = len(p)
    if m == 0:
        return 1.0
    n_above: float = float(np.sum(p > tau))
    pi0 = n_above / (m * (1.0 - tau))
    return float(np.clip(pi0, min_pi0, 1.0))


def estimate_pi0_grouped(
    p_values: NDArray[np.float64],
    group_ids: NDArray[np.intp],
    n_groups: int,
    tau: float = 0.5,
    min_pi0: float = 0.05,
) -> NDArray[np.float64]:
    """Estimate pi0 per spatial group.

    Parameters
    ----------
    p_values : NDArray, shape ``(V,)``
    group_ids : NDArray[int], shape ``(V,)``
    n_groups : int
    tau, min_pi0 : float

    Returns
    -------
    NDArray, shape ``(n_groups,)``
    """
    pi0 = np.ones(n_groups, dtype=np.float64)
    for g in range(n_groups):
        idx = group_ids == g
        if np.any(idx):
            pi0[g] = estimate_pi0(p_values[idx], tau=tau, min_pi0=min_pi0)
    return pi0


def smooth_pi0(
    pi0_raw: NDArray[np.float64],
    neighbors: list[list[int]],
    lam: float = 0.5,
) -> NDArray[np.float64]:
    """Smooth pi0 across spatial neighbours.

    One step of ``(pi0_g + lam * mean_{N(g)} pi0) / (1 + lam)`` over the
    finite neighbours of each group (fmrigds ``sabha_pi0_smooth_cpp``).

    Parameters
    ----------
    pi0_raw : NDArray, shape ``(n_groups,)``
    neighbors : list of list of int
    lam : float
        Non-negative smoothing strength; ``lam = 1`` averages a group with
        its neighbours' mean.

    Returns
    -------
    NDArray, shape ``(n_groups,)``
    """
    if lam <= 0:
        return pi0_raw.copy()

    pi0_smooth = pi0_raw.copy()
    for g in range(len(pi0_raw)):
        nbr_vals = pi0_raw[[j for j in neighbors[g] if j != g]]
        nbr_vals = nbr_vals[np.isfinite(nbr_vals)]
        if np.isfinite(pi0_raw[g]) and nbr_vals.size:
            pi0_smooth[g] = (pi0_raw[g] + lam * nbr_vals.mean()) / (1.0 + lam)
    return pi0_smooth


# -- Weighted BH ---------------------------------------------------------------

def _normalise_weights(
    p_values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Weights scaled to sum to the number of finite p-values.

    Tests with a missing p get weight 0; non-positive or non-finite
    weights become a small positive value (fmrigds ``sabha_weighted_bh_cpp``).
    """
    finite = np.isfinite(p_values)
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-8)
    w = np.where(finite, w, 0.0)
    total = float(w.sum())
    if total > 0:
        w = w * (float(finite.sum()) / total)
    return w


def weighted_bh(
    p_values: NDArray[np.float64],
    weights: NDArray[np.float64],
    alpha: float = 0.05,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], float]:
    """Weighted Benjamini-Hochberg procedure (Genovese, Roeder & Wasserman 2006).

    Weights are normalised to sum to the number of finite p-values, which
    is what the procedure's FDR guarantee assumes. Missing p-values get
    a missing q-value, are never rejected and do not count as tests.

    Parameters
    ----------
    p_values : NDArray, shape ``(V,)``
    weights : NDArray, shape ``(V,)``
        Per-voxel weights (higher = more likely to reject).
    alpha : float
        Target FDR level.

    Returns
    -------
    reject : NDArray[bool]
    qvalues : NDArray[float]
    threshold : float
        The BH threshold on the weighted scale ``p / w``.
    """
    p = np.asarray(p_values, dtype=np.float64)
    reject = np.zeros(p.shape, dtype=bool)
    qvalues = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    n = int(finite.sum())
    if n == 0:
        return reject, qvalues, 0.0

    # Weighted p-values: p_w = p / w  (lower weight → inflate p)
    w = _normalise_weights(p, weights)
    p_weighted = p[finite] / w[finite]

    # Standard BH on weighted p-values
    sorted_idx = np.argsort(p_weighted)
    sorted_pw = p_weighted[sorted_idx]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    adjusted = sorted_pw * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_finite = np.empty(n, dtype=np.float64)
    q_finite[sorted_idx] = adjusted
    qvalues[finite] = q_finite

    below = np.nonzero(sorted_pw <= alpha * ranks / n)[0]
    threshold = float(alpha * (below.max() + 1) / n) if below.size else 0.0
    if below.size:
        reject[finite] = p_weighted <= threshold

    return reject, qvalues, threshold


# -- Main entry point ----------------------------------------------------------

def spatial_fdr(
    p_values: NDArray[np.float64],
    group_ids: Optional[NDArray[np.intp]] = None,
    mask: Optional[NDArray[np.bool_]] = None,
    alpha: float = 0.05,
    tau: float = 0.5,
    smooth_lambda: float = 1.0,
    block_size: int = 5,
    min_pi0: float = 0.05,
) -> SpatialFdrResult:
    """Structure-Adaptive Weighted BH for spatial FDR control.

    Either provide pre-computed ``group_ids`` or a 3-D ``mask`` from
    which groups will be automatically constructed.

    Parameters
    ----------
    p_values : NDArray, shape ``(V,)``
        Raw p-values.
    group_ids : NDArray[int], optional
        Pre-computed group assignments.
    mask : NDArray[bool], shape ``(nx, ny, nz)``, optional
        Brain mask for automatic block construction.
    alpha : float
        Target FDR level.
    tau : float
        Threshold for Storey's pi0 estimator.
    smooth_lambda : float
        Spatial smoothing of pi0 across neighbours, one step of
        ``(pi0 + lam * mean_neighbours) / (1 + lam)``.
    block_size : int
        Block edge length for automatic grouping.
    min_pi0 : float
        Lower bound for the smoothed pi0 of each group.

    Returns
    -------
    SpatialFdrResult
    """
    p_values = np.asarray(p_values, dtype=np.float64).ravel()
    V = len(p_values)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0, 1)")
    if not (0.0 <= min_pi0 <= 1.0):
        raise ValueError("min_pi0 must be in [0, 1]")
    if smooth_lambda < 0.0:
        warnings.warn("smooth_lambda < 0 not recommended; setting to 0", UserWarning)
        smooth_lambda = 0.0

    # Build groups if not provided
    neighbors: list[list[int]] = []
    if group_ids is None:
        if mask is not None:
            group_ids, n_groups = create_3d_blocks(mask, block_size)
            neighbors = create_group_neighbors(
                mask, group_ids, n_groups, block_size,
            )
        else:
            # Single group fallback — reduces to standard BH
            group_ids = np.zeros(V, dtype=np.intp)
            n_groups = 1
            neighbors = [[]]
    else:
        group_ids_arr = np.asarray(group_ids).ravel()
        if group_ids_arr.shape[0] != V:
            raise ValueError(
                f"group_ids length {group_ids_arr.shape[0]} must match p_values length {V}"
            )
        if group_ids_arr.dtype.kind in ("f", "c"):
            if not np.all(np.isfinite(group_ids_arr)):
                raise ValueError("group_ids must contain finite integer labels")
            if not np.allclose(group_ids_arr, np.round(group_ids_arr)):
                raise ValueError("group_ids must contain integer labels")
        group_ids_int = group_ids_arr.astype(np.int64, copy=False)
        # Match fmrireg semantics: compress arbitrary labels to dense 0..G-1 IDs.
        _, group_ids_dense = np.unique(group_ids_int, return_inverse=True)
        group_ids = group_ids_dense.astype(np.intp)
        n_groups = int(group_ids.max()) + 1
        if not neighbors:
            neighbors = [[] for _ in range(n_groups)]

    # 1. Estimate pi0 per group from its finite p-values (bounded to [0, 1])
    pi0_raw = estimate_pi0_grouped(
        p_values, group_ids, n_groups, tau=tau, min_pi0=0.0,
    )

    # 2. Smooth pi0 across neighbours, then apply the lower bound
    pi0_smooth = np.clip(smooth_pi0(pi0_raw, neighbors, lam=smooth_lambda), min_pi0, 1.0)

    # 3. Adaptive weights w = 1 / pi0 per voxel, normalised over the tested voxels
    voxel_weights = _normalise_weights(
        p_values, 1.0 / np.maximum(pi0_smooth, 1e-10)[group_ids]
    )

    # 4. Weighted BH
    reject, qvalues, threshold = weighted_bh(p_values, voxel_weights, alpha)

    return SpatialFdrResult(
        reject=reject,
        qvalues=qvalues,
        weights=voxel_weights,
        pi0_raw=pi0_raw,
        pi0_smooth=pi0_smooth,
        threshold=threshold,
    )
