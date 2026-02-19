"""Nyström approximation for extending landmark solutions to full space.

Given a GLM solved at *L* landmark voxels, the Nyström extension
interpolates beta coefficients to all *V* voxels using a sparse
heat-kernel weight matrix built from k-nearest-neighbour distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


@dataclass(frozen=True)
class LandmarkWeights:
    """Sparse landmark-to-voxel interpolation weights.

    Attributes
    ----------
    indices : NDArray[int], shape ``(V, k)``
        For each voxel, the indices of its *k* nearest landmarks.
    weights : NDArray[float], shape ``(V, k)``
        Row-normalised heat-kernel weights.
    n_voxels : int
        Number of full-space voxels (*V*).
    n_landmarks : int
        Number of landmark voxels (*L*).
    """

    indices: NDArray[np.intp]
    weights: NDArray[np.float64]
    n_voxels: int
    n_landmarks: int


def select_landmarks(
    coords: NDArray[np.float64],
    n_landmarks: int,
    method: str = "kmeans",
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.intp]:
    """Select landmark voxel indices from coordinates.

    Parameters
    ----------
    coords : NDArray, shape ``(V, d)``
        Voxel coordinates (e.g. MNI or ijk).
    n_landmarks : int
        Number of landmarks to select.
    method : str
        ``"kmeans"`` (default) or ``"random"``.
    rng : Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    NDArray[int], shape ``(n_landmarks,)``
        Indices into *coords* of selected landmarks.
    """
    if rng is None:
        rng = np.random.default_rng()
    V = coords.shape[0]
    n_landmarks = min(n_landmarks, V)

    if method == "random":
        return rng.choice(V, size=n_landmarks, replace=False)

    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=n_landmarks,
            random_state=int(rng.integers(2**31)),
            batch_size=min(1024, V),
            n_init=3,
        )
        km.fit(coords)
        # Pick the voxel closest to each centroid
        tree = KDTree(coords)
        _, idx = tree.query(km.cluster_centers_)
        return np.asarray(idx, dtype=np.intp)

    raise ValueError(f"Unknown method: {method!r}")


def build_landmark_weights(
    coords: NDArray[np.float64],
    landmark_coords: NDArray[np.float64],
    k: int = 6,
    bandwidth: Optional[float] = None,
) -> LandmarkWeights:
    """Build sparse k-NN heat-kernel weights from voxels to landmarks.

    Parameters
    ----------
    coords : NDArray, shape ``(V, d)``
        Full-space voxel coordinates.
    landmark_coords : NDArray, shape ``(L, d)``
        Landmark voxel coordinates.
    k : int
        Number of nearest neighbours per voxel.
    bandwidth : float, optional
        Heat-kernel bandwidth *h*.  If ``None``, uses the median of
        the k-th neighbour distances.

    Returns
    -------
    LandmarkWeights
    """
    V = coords.shape[0]
    L = landmark_coords.shape[0]
    if L < 1:
        raise ValueError("landmark_coords must contain at least one landmark")
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, L)

    tree = KDTree(landmark_coords)
    dists, indices = tree.query(coords, k=k)
    if k == 1:
        # scipy.spatial.KDTree returns 1-D arrays for k=1, but downstream code
        # assumes a (V, k) layout.
        dists = np.asarray(dists)[:, np.newaxis]
        indices = np.asarray(indices)[:, np.newaxis]

    # Bandwidth: median of k-th neighbour distance
    if bandwidth is None:
        bandwidth = float(np.maximum(np.median(dists[:, -1]), 1e-8))

    # Heat-kernel weights: exp(-d^2 / (2 h^2))
    weights = np.exp(-dists**2 / (2.0 * bandwidth**2))

    # Row-normalise so each voxel's weights sum to 1
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    weights /= row_sums

    return LandmarkWeights(
        indices=indices.astype(np.intp),
        weights=weights,
        n_voxels=V,
        n_landmarks=L,
    )


def extend_betas(
    betas_landmark: NDArray[np.float64],
    lw: LandmarkWeights,
) -> NDArray[np.float64]:
    """Extend landmark betas to full voxel space via Nyström interpolation.

    Parameters
    ----------
    betas_landmark : NDArray, shape ``(p, L)``
        Coefficients at landmark voxels.
    lw : LandmarkWeights
        Pre-computed landmark weights.

    Returns
    -------
    NDArray, shape ``(p, V)``
        Interpolated coefficients for all voxels.
    """
    p, L = betas_landmark.shape
    V = lw.n_voxels
    assert L == lw.n_landmarks, f"Expected {lw.n_landmarks} landmarks, got {L}"

    # B_full[:, v] = sum_j w[v,j] * B_landmark[:, idx[v,j]]
    betas_full = np.zeros((p, V), dtype=np.float64)
    for j in range(lw.indices.shape[1]):
        betas_full += lw.weights[:, j][np.newaxis, :] * betas_landmark[:, lw.indices[:, j]]

    return betas_full
