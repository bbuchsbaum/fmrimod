"""HRF normalization modes.

Wraps an :class:`HRF` so that every evaluation is divided by a pre-computed
normalization factor. The factor is computed once from a reference grid so
the resulting HRF behaves as a fixed-scale function of time.

Modes
-----
``"spm"``
    Match Nilearn's ``hrf_model="spm"`` scale convention: divide by the
    *sum* of the HRF evaluated on Nilearn's reference grid
    (``time_length=32``, ``oversampling=50``). This makes
    ``hrf(reference_grid).sum() == 1`` and reproduces the normalization scale
    used by :func:`nilearn.glm.first_level.spm_hrf`.

``"unit_peak"``
    Divide by the absolute peak on the reference grid; the resulting HRF
    has unit peak amplitude. Equivalent to the R idiom
    ``normalise_hrf(hrf)`` from ``fmrihrf``.

``"unit_integral"``
    Divide by the trapezoidal integral on the HRF span so the continuous
    integral of the HRF equals 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF

NormMode = Literal["spm", "unit_peak", "unit_integral"]
VALID_NORM_MODES: tuple[str, ...] = ("spm", "unit_peak", "unit_integral")

__all__ = ["NormMode", "VALID_NORM_MODES", "normalize"]

# Reference grid used to compute the normalization factor. Matches
# Nilearn's ``spm_hrf`` defaults (``time_length=32``, ``oversampling=50``).
_REF_DT = 1.0 / 50.0
_SPM_TIME_LENGTH = 32.0


def _reference_grid(hrf: HRF, mode: NormMode) -> NDArray[np.float64]:
    """Return the grid used to compute ``mode``'s fixed divisor."""
    if mode == "spm":
        n = max(int(round(_SPM_TIME_LENGTH / _REF_DT)), 2)
        return np.linspace(0.0, _SPM_TIME_LENGTH, n, dtype=np.float64)

    span = float(getattr(hrf, "span", 32.0)) or 32.0
    n = max(int(round(span / _REF_DT)) + 1, 2)
    return np.linspace(0.0, span, n, dtype=np.float64)


def _normalization_factor(hrf: HRF, mode: NormMode) -> float:
    """Compute the divisor for ``mode`` from a dense reference grid."""
    t = _reference_grid(hrf, mode)
    values = np.asarray(hrf(t), dtype=np.float64)
    # For multi-basis HRFs, normalize by the canonical column (column 0)
    # so derivative columns scale by the same factor.
    if values.ndim == 2:
        values = values[:, 0]

    if mode == "spm":
        z = float(values.sum())
    elif mode == "unit_peak":
        z = float(np.max(np.abs(values)))
    elif mode == "unit_integral":
        z = float(np.trapz(values, t))
    else:
        raise ValueError(
            f"Unknown HRF normalization mode {mode!r}. "
            "Expected one of: 'spm', 'unit_peak', 'unit_integral'."
        )
    if not np.isfinite(z) or abs(z) < np.finfo(np.float64).tiny:
        raise ValueError(
            f"Normalization factor for {hrf.name!r} (mode={mode!r}) "
            f"is {z!r}; refusing to divide."
        )
    return z


@dataclass
class _NormalizedHRF(HRF):
    """An HRF whose every evaluation is divided by a fixed scalar.

    Built by :func:`normalize`. Use the factory rather than constructing
    this class directly.
    """

    base: HRF | None = None
    norm_mode: NormMode | None = None
    norm_factor: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.base is None or self.norm_mode is None:
            raise ValueError("_NormalizedHRF requires `base` and `norm_mode`")

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        if self.base is None:
            raise ValueError("_NormalizedHRF requires `base` before evaluation")
        values = np.asarray(self.base(t), dtype=np.float64)
        return values / self.norm_factor


def normalize(hrf: HRF, mode: NormMode) -> HRF:
    """Return a new HRF whose calls are divided by a fixed normalization factor.

    Parameters
    ----------
    hrf
        Source HRF (any subclass of :class:`HRF`).
    mode
        ``"spm"``, ``"unit_peak"``, or ``"unit_integral"``. See module
        docstring for the exact convention.

    Returns
    -------
    HRF
        A new HRF with the same ``nbasis``/``span``/params, but every
        evaluation rescaled by the precomputed factor.

    Examples
    --------
    >>> from fmrimod.hrf import HRF_SPMG1
    >>> from fmrimod.hrf.normalization import normalize
    >>> hrf = normalize(HRF_SPMG1, "spm")
    >>> # hrf(t) now matches Nilearn's spm_hrf scale on its reference grid.
    """
    factor = _normalization_factor(hrf, mode)
    label = f"{hrf.name}[norm={mode}]"
    params = dict(hrf.params)
    params["hrf_norm"] = mode
    params["hrf_norm_factor"] = factor
    return _NormalizedHRF(
        name=label,
        nbasis=hrf.nbasis,
        span=hrf.span,
        params=params,
        param_names=hrf.param_names,
        base=hrf,
        norm_mode=mode,
        norm_factor=factor,
    )
