"""HRF normalization modes.

A single typed entry point — :func:`normalize` — wraps an :class:`HRF`
in :class:`_NormalizedHRF` so every evaluation is rescaled by a fixed
divisor computed once from a reference grid. The result behaves as a
fixed-scale function of time and composes cleanly with the rest of the
decorator chain.

Modes
-----
``"spm"``
    Match Nilearn's ``hrf_model="spm"`` scale convention: divide by the
    *sum* of the HRF evaluated on Nilearn's reference grid
    (``time_length=32``, ``oversampling=50``). Makes
    ``hrf(reference_grid).sum() == 1`` and reproduces the normalization
    scale used by :func:`nilearn.glm.first_level.spm_hrf`.

``"unit_peak"``
    Divide all columns by a single scalar — the absolute peak of the
    canonical (column 0 for multi-basis) on the reference grid.
    R-aligned: matches ``normalise_hrf`` from ``fmrihrf``.

``"unit_peak_per_basis"``
    Multi-basis only mode: each column is divided by its own absolute
    peak so every column has unit peak. This is the semantic the legacy
    ``decorators.normalize_hrf`` shipped before bead
    ``bd-01KRGCZ6QJME1JD8FD5D4PGC04`` collapsed the three normalization
    APIs into this one entry point. It exists because callers that
    normalize a heterogeneously-scaled basis set (e.g. an ``as_hrf``
    wrapper around a hand-built scaling) rely on independent column
    rescaling; ``unit_peak`` would leave non-canonical columns at
    arbitrary scale. For SPMG2 / SPMG3 prefer ``unit_peak``.

``"unit_integral"``
    Divide by the trapezoidal integral on the HRF span so the
    continuous integral equals 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF

NormMode = Literal["spm", "unit_peak", "unit_peak_per_basis", "unit_integral"]
VALID_NORM_MODES: tuple[str, ...] = (
    "spm", "unit_peak", "unit_peak_per_basis", "unit_integral",
)

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


def _normalization_factor(
    hrf: HRF, mode: NormMode
) -> Union[float, NDArray[np.float64]]:
    """Compute the divisor for ``mode`` from a dense reference grid.

    Returns a scalar for ``spm`` / ``unit_peak`` / ``unit_integral`` and
    a length-``nbasis`` vector for ``unit_peak_per_basis``.
    """
    t = _reference_grid(hrf, mode)
    values = np.asarray(hrf(t), dtype=np.float64)

    if mode == "unit_peak_per_basis":
        if values.ndim == 1:
            z_scalar = float(np.max(np.abs(values)))
            return _guard_factor(hrf, mode, z_scalar)
        peaks = np.array(
            [float(np.max(np.abs(values[:, i]))) for i in range(values.shape[1])],
            dtype=np.float64,
        )
        # Replace zero peaks (constant-zero column) with 1.0 to avoid
        # division by zero; the column comes through unchanged.
        peaks = np.where(peaks > 0, peaks, 1.0)
        if not np.all(np.isfinite(peaks)):
            raise ValueError(
                f"Per-basis peaks for {hrf.name!r} include non-finite "
                f"values ({peaks!r}); refusing to divide."
            )
        return peaks

    # All other modes collapse to a single scalar derived from the canonical
    # column (column 0 for multi-basis).
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
            f"Expected one of: {sorted(VALID_NORM_MODES)}."
        )
    return _guard_factor(hrf, mode, z)


def _guard_factor(hrf: HRF, mode: NormMode, z: float) -> float:
    if not np.isfinite(z) or abs(z) < np.finfo(np.float64).tiny:
        raise ValueError(
            f"Normalization factor for {hrf.name!r} (mode={mode!r}) "
            f"is {z!r}; refusing to divide."
        )
    return z


@dataclass
class _NormalizedHRF(HRF):
    """An HRF whose every evaluation is divided by a fixed factor.

    For ``unit_peak_per_basis`` the factor is a length-``nbasis``
    vector and division is per-column; for all other modes the factor
    is a scalar applied to every column.

    Built by :func:`normalize`. Use the factory rather than
    constructing this class directly.
    """

    name: str = ""
    nbasis: int = 1
    span: float = 24.0
    base: HRF | None = None
    norm_mode: NormMode | None = None
    norm_factor: Union[float, NDArray[np.float64]] = 1.0

    def __post_init__(self) -> None:
        if self.base is None or self.norm_mode is None:
            raise ValueError("_NormalizedHRF requires `base` and `norm_mode`")
        # Derive identity from the base; the factory sets a label on top.
        if not self.name:
            self.name = f"{self.base.name}[norm={self.norm_mode}]"
        self.nbasis = self.base.nbasis
        self.span = self.base.span
        new_params = dict(self.base.params)
        new_params["hrf_norm"] = self.norm_mode
        # The factor is stored on the instance; we keep a JSON-safe
        # representation in params for back-compat readers.
        if np.ndim(self.norm_factor) == 0:
            new_params["hrf_norm_factor"] = float(self.norm_factor)
        else:
            new_params["hrf_norm_factor"] = (
                np.asarray(self.norm_factor).tolist()
            )
        new_params["_normalized"] = True
        self.params = new_params
        self.param_names = self.base.param_names

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
        assert self.base is not None
        values = np.asarray(self.base(t), dtype=np.float64)
        if np.ndim(self.norm_factor) == 0:
            return values / float(self.norm_factor)
        # Vector factor: per-column division. Reshape 1-D output of a
        # multi-basis evaluation at a scalar point into the matching 2-D
        # form before broadcasting.
        factor_arr = np.asarray(self.norm_factor, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return values / factor_arr[np.newaxis, :]


def normalize(hrf: HRF, mode: NormMode) -> HRF:
    """Return a new HRF whose calls are divided by a fixed normalization factor.

    See module docstring for the four supported modes.

    Parameters
    ----------
    hrf
        Source HRF (any subclass of :class:`HRF`).
    mode
        ``"spm"``, ``"unit_peak"``, ``"unit_peak_per_basis"``, or
        ``"unit_integral"``.

    Returns
    -------
    HRF
        A new HRF whose evaluations are rescaled by the precomputed
        factor.

    Examples
    --------
    >>> from fmrimod.hrf import HRF_SPMG1
    >>> from fmrimod.hrf.normalization import normalize
    >>> hrf = normalize(HRF_SPMG1, "spm")
    >>> # hrf(t) now matches Nilearn's spm_hrf scale on its reference grid.
    """
    factor = _normalization_factor(hrf, mode)
    label = f"{hrf.name}[norm={mode}]"
    return _NormalizedHRF(
        name=label,
        base=hrf,
        norm_mode=mode,
        norm_factor=factor,
    )
