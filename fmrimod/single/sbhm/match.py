"""SBHM matching: find best library HRF per voxel via cosine similarity.

Whitens and normalizes basis coefficients, then computes cosine similarity
with library coordinates to find the best-matching HRF shape per voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SbhmMatchResult:
    """Typed SBHM library-match result."""

    matched_idx: NDArray[np.int_]
    margin: NDArray[np.float64]
    alpha_coords: NDArray[np.float64]
    similarity: NDArray[np.float64]
    weights: NDArray[np.float64] | None = None


def _stein_shrinkage(
    beta: NDArray[np.float64],
    S: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply Stein shrinkage to basis coefficients.

    Parameters
    ----------
    beta : NDArray, shape ``(r, V)``
        Basis coefficients per voxel.
    S : NDArray, shape ``(r,)``
        Singular values from library SVD.

    Returns
    -------
    NDArray, shape ``(r, V)``
        Shrunk coefficients.
    """
    r, V = beta.shape
    # Simple diagonal shrinkage: shrink toward zero
    # Scale by S^2 / (S^2 + mean(beta^2))
    beta_var = np.mean(beta ** 2, axis=1, keepdims=True)  # (r, 1)
    S2 = S[:, np.newaxis] ** 2
    shrink_factor = S2 / (S2 + beta_var + 1e-8)
    return cast("NDArray[np.float64]", beta * shrink_factor)


def sbhm_match(
    beta_bar: NDArray[np.float64],
    S: NDArray[np.float64],
    A: NDArray[np.float64],
    shrink: bool = True,
    top_k: int = 1,
) -> SbhmMatchResult:
    """Match per-voxel HRF shapes to library via cosine similarity.

    Parameters
    ----------
    beta_bar : NDArray, shape ``(r, V)``
        Average basis coefficients per voxel from prepass.
    S : NDArray, shape ``(r,)``
        Singular values from library SVD.
    A : NDArray, shape ``(n_library, r)``
        Library coordinates (right singular vectors).
    shrink : bool, default=True
        Apply Stein shrinkage to coefficients before matching.
    top_k : int, default=1
        Number of top matches to return per voxel. If ``k > 1``, returns
        softmax-weighted blend of top-k library members.

    Returns
    -------
    SbhmMatchResult
        Typed result with matched indices, margin, matched coordinates,
        full similarity matrix, and optional top-k weights.

    Notes
    -----
    The matching procedure:
    1. Whiten coefficients: ``beta_w = beta_bar / S``
    2. L2-normalize both whitened coefficients and library coordinates
    3. Compute cosine similarity: ``sim = A_w_norm @ beta_w_norm``
    4. Find best match per voxel (argmax)
    5. Margin = top1_similarity - top2_similarity

    If ``top_k > 1``, returns softmax-weighted blend of top-k matches.

    Examples
    --------
    >>> import numpy as np
    >>> from fmrimod.single.sbhm import sbhm_match
    >>> beta_bar = np.random.randn(3, 500)
    >>> S = np.array([10.0, 5.0, 2.0])
    >>> A = np.random.randn(50, 3)
    >>> result = sbhm_match(beta_bar, S, A, top_k=1)
    >>> result.matched_idx.shape
    (500,)
    >>> result.margin.shape
    (500,)
    """
    beta_bar = np.asarray(beta_bar, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)

    if beta_bar.ndim == 1:
        beta_bar = beta_bar[:, np.newaxis]

    r, V = beta_bar.shape
    n_library = A.shape[0]

    if S.shape[0] != r:
        raise ValueError(f"S has length {S.shape[0]}, expected {r}")
    if A.shape[1] != r:
        raise ValueError(f"A has {A.shape[1]} columns, expected {r}")

    # Optional shrinkage
    if shrink:
        beta_shrunk = _stein_shrinkage(beta_bar, S)
    else:
        beta_shrunk = beta_bar.copy()

    # Whiten: divide by singular values
    beta_w = beta_shrunk / (S[:, np.newaxis] + 1e-12)
    A_w = A / (S[np.newaxis, :] + 1e-12)

    # L2-normalize
    beta_norms = np.sqrt(np.sum(beta_w ** 2, axis=0, keepdims=True))
    beta_norms[beta_norms == 0] = 1.0
    beta_w_norm = beta_w / beta_norms

    A_norms = np.sqrt(np.sum(A_w ** 2, axis=1, keepdims=True))
    A_norms[A_norms == 0] = 1.0
    A_w_norm = A_w / A_norms

    # Cosine similarity: (n_library, V)
    sim = A_w_norm @ beta_w_norm

    # Top-k matching
    if top_k == 1:
        matched_idx = np.argmax(sim, axis=0)  # (V,)
        top_sim = sim[matched_idx, np.arange(V)]  # (V,)

        # Margin: difference between top-1 and top-2
        sim_sorted = np.sort(sim, axis=0)
        if n_library >= 2:
            margin = sim_sorted[-1, :] - sim_sorted[-2, :]
        else:
            margin = np.ones(V, dtype=np.float64)

        # Final coordinates: from the matched library member
        alpha_coords = A[matched_idx, :].T  # (r, V)

        return SbhmMatchResult(
            matched_idx=matched_idx,
            margin=margin,
            alpha_coords=alpha_coords,
            similarity=sim,
        )
    else:
        # Top-k: return indices and softmax weights
        top_k = min(top_k, n_library)
        top_idx = np.argsort(-sim, axis=0)[:top_k, :]  # (k, V)
        top_sim = np.take_along_axis(sim, top_idx, axis=0)  # (k, V)

        # Softmax weights
        exp_sim = np.exp(top_sim - np.max(top_sim, axis=0, keepdims=True))
        weights = exp_sim / np.sum(exp_sim, axis=0, keepdims=True)  # (k, V)

        # Weighted blend of coordinates
        alpha_coords = np.zeros((r, V), dtype=np.float64)
        for i in range(top_k):
            alpha_coords += weights[i, :][np.newaxis, :] * A[top_idx[i, :], :].T

        # Margin: top1 - top2
        if top_k >= 2:
            margin = top_sim[0, :] - top_sim[1, :]
        else:
            margin = np.ones(V, dtype=np.float64)

        return SbhmMatchResult(
            matched_idx=top_idx.T,  # (V, k)
            margin=margin,
            alpha_coords=alpha_coords,
            similarity=sim,
            weights=weights.T,  # (V, k)
        )
