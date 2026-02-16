"""Sketch matrices for randomised linear algebra.

Provides Gaussian, SRHT (subsampled randomised Hadamard transform),
and CountSketch projections used by the low-rank GLM solver.

All sketches implement the same interface::

    S = make_sketch(n, k, kind="gaussian", rng=rng)
    Y_sketch = S @ Y          # (k, V)
    X_sketch = S @ X          # (k, p)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import hadamard


class SketchKind(str, Enum):
    GAUSSIAN = "gaussian"
    SRHT = "srht"
    COUNTSKETCH = "countsketch"


def make_sketch(
    n: int,
    k: int,
    kind: Union[str, SketchKind] = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Create a ``(k, n)`` sketch matrix.

    Parameters
    ----------
    n : int
        Original dimension (rows to compress).
    k : int
        Target sketch dimension (``k << n``).
    kind : str or SketchKind
        One of ``"gaussian"``, ``"srht"``, ``"countsketch"``.
    rng : Generator, optional
        Numpy random generator.  Defaults to ``default_rng()``.

    Returns
    -------
    NDArray, shape ``(k, n)``
        The sketch matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    kind = SketchKind(kind)

    if kind is SketchKind.GAUSSIAN:
        return _gaussian_sketch(n, k, rng)
    if kind is SketchKind.SRHT:
        return _srht_sketch(n, k, rng)
    if kind is SketchKind.COUNTSKETCH:
        return _countsketch_matrix(n, k, rng)
    raise ValueError(f"Unknown sketch kind: {kind}")  # pragma: no cover


# -- Gaussian -----------------------------------------------------------------

def _gaussian_sketch(n: int, k: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Dense Gaussian sketch: ``S ~ N(0, 1/k)``."""
    return rng.standard_normal((k, n)) / np.sqrt(k)


# -- SRHT ---------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _srht_sketch(n: int, k: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Subsampled Randomised Hadamard Transform.

    Pads *n* to the next power of two, applies a diagonal sign flip,
    the Walsh-Hadamard transform, then subsamples *k* rows.
    """
    n_pad = _next_power_of_two(n)

    # Diagonal sign flips
    signs = rng.choice([-1.0, 1.0], size=n_pad)
    D = np.diag(signs)

    # Hadamard matrix (normalised)
    H = hadamard(n_pad).astype(np.float64) / np.sqrt(n_pad)

    # Random row subset
    rows = rng.choice(n_pad, size=k, replace=False)
    S_full = H[rows] @ D  # (k, n_pad)

    # Trim padding columns
    return S_full[:, :n] * np.sqrt(n_pad / k)


# -- CountSketch --------------------------------------------------------------

def _countsketch_matrix(
    n: int, k: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sparse CountSketch matrix.

    Each column of the identity gets mapped to a random bucket with a
    random sign.  This gives a sparse ``(k, n)`` matrix.
    """
    buckets = rng.integers(0, k, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)

    S = np.zeros((k, n), dtype=np.float64)
    S[buckets, np.arange(n)] = signs
    return S


# -- Utility: apply sketch to (n, ...) data ----------------------------------

def sketch_data(
    S: NDArray[np.float64],
    *arrays: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Apply a sketch ``S`` to one or more ``(n, ...)`` arrays.

    Parameters
    ----------
    S : NDArray, shape ``(k, n)``
        Sketch matrix.
    *arrays : NDArray
        Arrays whose first axis has length *n*.

    Returns
    -------
    tuple of NDArray
        Sketched versions of each input array.
    """
    return tuple(S @ a for a in arrays)
