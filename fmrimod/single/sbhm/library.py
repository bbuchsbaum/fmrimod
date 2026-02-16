"""SBHM library construction via SVD.

Constructs a shared low-rank basis from a library of candidate HRF shapes
via truncated SVD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


@dataclass
class SbhmLibrary:
    """Result of SBHM library SVD.

    Attributes
    ----------
    B : NDArray, shape ``(T, r)``
        Shared temporal basis (left singular vectors scaled by singular values).
    S : NDArray, shape ``(r,)``
        Singular values of the library.
    A : NDArray, shape ``(n_library, r)``
        Library coordinates (right singular vectors).
    library_H : NDArray, shape ``(T, n_library)``
        Original HRF library matrix.
    """

    B: NDArray[np.float64]
    S: NDArray[np.float64]
    A: NDArray[np.float64]
    library_H: NDArray[np.float64]


def build_sbhm_library(
    library_H: NDArray[np.float64],
    r: int,
    normalize: bool = True,
) -> SbhmLibrary:
    """Construct SBHM shared basis from HRF library via truncated SVD.

    Parameters
    ----------
    library_H : NDArray, shape ``(T, n_library)``
        Matrix of candidate HRF shapes evaluated on a TR grid.
        Each column is one candidate HRF.
    r : int
        Number of SVD components to retain.
    normalize : bool, default=True
        If ``True``, L2-normalize each library column before SVD.

    Returns
    -------
    SbhmLibrary
        Dataclass containing:
        - ``B``: shared temporal basis (T, r)
        - ``S``: singular values (r,)
        - ``A``: library coordinates (n_library, r)
        - ``library_H``: original library (T, n_library)

    Notes
    -----
    The shared basis ``B`` is constructed as ``U[:, :r] * S[:r]``, so that
    ``library_H ≈ B @ A.T``.  The coordinates ``A`` can be used to match
    per-voxel HRF shapes to the library.

    Examples
    --------
    >>> import numpy as np
    >>> from fmrimod.single.sbhm import build_sbhm_library
    >>> # 20 time points, 50 candidate HRFs
    >>> library_H = np.random.randn(20, 50)
    >>> lib = build_sbhm_library(library_H, r=3)
    >>> lib.B.shape
    (20, 3)
    >>> lib.A.shape
    (50, 3)
    """
    library_H = np.asarray(library_H, dtype=np.float64)
    if library_H.ndim != 2:
        raise ValueError("library_H must be 2D (T, n_library)")

    T, n_library = library_H.shape
    if r > min(T, n_library):
        raise ValueError(
            f"r={r} exceeds min(T={T}, n_library={n_library})"
        )

    H = library_H.copy()

    # Optional L2 normalization
    if normalize:
        col_norms = np.sqrt(np.sum(H ** 2, axis=0))
        col_norms[col_norms == 0] = 1.0
        H = H / col_norms[np.newaxis, :]

    # Truncated SVD: H ≈ U @ diag(S) @ Vt
    U, S_vals, Vt = linalg.svd(H, full_matrices=False)

    # Keep first r components
    U_r = U[:, :r]
    S_r = S_vals[:r]
    Vt_r = Vt[:r, :]

    # B = U_r @ diag(S_r)  (scaled basis)
    B = U_r * S_r[np.newaxis, :]

    # A = Vt_r.T  (library coordinates)
    A = Vt_r.T

    return SbhmLibrary(
        B=B,
        S=S_r.copy(),
        A=A,
        library_H=library_H,
    )
