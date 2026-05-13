"""Penalty matrices for regularizing HRF basis coefficients.

Dispatch is keyed on the *type* of the HRF, not on its ``name``. That
matters because decorator chains (``lag_hrf``, ``block_hrf``,
``normalize``, ``bind_basis``) mutate the name string but preserve the
underlying kind via ``.base`` / ``.components``. The earlier
substring-on-name implementation silently fell through to ridge
whenever the HRF had been decorated; this version delegates through
the wrapper subclasses so the penalty is the one that matches the
*structure*. See bead ``bd-01KRGCZ1VMBRS9BXWVM1DTDE4M``.
"""

from __future__ import annotations

from functools import singledispatch

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag

from .core import HRF, BoundBasisHRF
from .decorators import BlockedHRF, LaggedHRF, _PeakNormalizedHRF
from .library import (
    BSplineHRF,
    DaguerreHRF,
    FIRHRF,
    FourierHRF,
)
from .normalization import _NormalizedHRF
from .spm_hrf import SPMG2_HRF, SPMG3_HRF


@singledispatch
def penalty_matrix(hrf: HRF, order: int = 2, **kwargs) -> NDArray[np.float64]:
    """Generate penalty matrix for regularizing HRF basis coefficients.

    Default (no specialised handler registered): ridge / identity penalty.

    Args:
        hrf: HRF object.
        order: Order of the penalty (default: 2).
        **kwargs: Forwarded to the handler for ``type(hrf)``.

    Returns:
        Symmetric positive-semidefinite penalty matrix of shape
        ``(hrf.nbasis, hrf.nbasis)``.
    """
    return np.eye(hrf.nbasis)


# --- Decorator delegation --------------------------------------------------
# Decorators preserve the structure of their base; the penalty is whatever
# the base's penalty would be. ``_PeakNormalizedHRF`` and ``_NormalizedHRF``
# only rescale, so they delegate too.


@penalty_matrix.register
def _penalty_lagged(
    hrf: LaggedHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    assert hrf.base is not None
    return penalty_matrix(hrf.base, order, **kwargs)


@penalty_matrix.register
def _penalty_blocked(
    hrf: BlockedHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    assert hrf.base is not None
    return penalty_matrix(hrf.base, order, **kwargs)


@penalty_matrix.register
def _penalty_peak_normalized(
    hrf: _PeakNormalizedHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    assert hrf.base is not None
    return penalty_matrix(hrf.base, order, **kwargs)


@penalty_matrix.register
def _penalty_normalized(
    hrf: _NormalizedHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    assert hrf.base is not None
    return penalty_matrix(hrf.base, order, **kwargs)


@penalty_matrix.register
def _penalty_bound_basis(
    hrf: BoundBasisHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    """Block-diagonal composition: each component carries its own penalty."""
    blocks = [penalty_matrix(c, order, **kwargs) for c in hrf.components]
    return block_diag(*blocks)


# --- Per-kind penalties ----------------------------------------------------


@penalty_matrix.register
def _penalty_spmg2(
    hrf: SPMG2_HRF, order: int = 2, *, shrink_deriv: float = 2.0, **kwargs
) -> NDArray[np.float64]:
    """Canonical term unpenalized; temporal derivative shrunk."""
    nb = hrf.nbasis
    R = np.eye(nb)
    if nb >= 1:
        R[0, 0] = 0.0
    if nb >= 2:
        R[1, 1] = shrink_deriv
    return R


@penalty_matrix.register
def _penalty_spmg3(
    hrf: SPMG3_HRF, order: int = 2, *, shrink_deriv: float = 2.0, **kwargs
) -> NDArray[np.float64]:
    """Canonical term unpenalized; temporal + dispersion derivatives shrunk."""
    nb = hrf.nbasis
    R = np.eye(nb)
    if nb >= 1:
        R[0, 0] = 0.0
    if nb >= 2:
        R[1, 1] = shrink_deriv
    if nb >= 3:
        R[2, 2] = shrink_deriv
    return R


@penalty_matrix.register
def _penalty_bspline(
    hrf: BSplineHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    """Roughness penalty via discrete derivatives."""
    return _roughness_penalty(hrf.nbasis, order)


@penalty_matrix.register
def _penalty_fir(
    hrf: FIRHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    """Roughness penalty via discrete derivatives."""
    return _roughness_penalty(hrf.nbasis, order)


@penalty_matrix.register
def _penalty_fourier(
    hrf: FourierHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    """Higher frequencies are penalized more."""
    nb = hrf.nbasis
    freqs = np.ceil(np.arange(1, nb + 1) / 2)
    return np.diag(freqs ** order)


@penalty_matrix.register
def _penalty_daguerre(
    hrf: DaguerreHRF, order: int = 2, **kwargs
) -> NDArray[np.float64]:
    """Higher-order Daguerre terms are penalized more."""
    nb = hrf.nbasis
    return np.diag(np.arange(nb) ** 2)


# --- Helpers ---------------------------------------------------------------


def _roughness_penalty(nbasis: int, order: int) -> NDArray[np.float64]:
    """Discrete-derivative roughness penalty for FIR / B-spline / tent bases."""
    if nbasis <= 1 or nbasis <= order:
        return np.eye(nbasis)
    D = np.diff(np.eye(nbasis), n=order, axis=0)
    return D.T @ D
