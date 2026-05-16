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

Multi-basis broadcasting rule
-----------------------------
For multi-basis HRFs (SPMG2 / SPMG3 / FIR / B-spline), ``"spm"``,
``"unit_peak"``, and ``"unit_integral"`` all collapse to a **single
scalar divisor** computed from the canonical column (column 0) on the
reference grid; that scalar is then applied uniformly to every basis
column. This preserves the physical interpretation of derivative and
dispersion columns as latency / shape perturbations of a
fixed-amplitude canonical — ``β`` units stay commensurate across
columns.

``"unit_peak_per_basis"`` is the only mode that rescales each basis
column independently. Use it when the basis columns are not SPM-style
shape derivatives — e.g. an arbitrary multi-channel kernel wrapped via
``as_hrf``.
"""

from __future__ import annotations

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
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        z = float(trapezoid(values, t))
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


class _NormalizedHRF(HRF):
    """An HRF whose every evaluation is divided by a fixed factor.

    For ``unit_peak_per_basis`` the factor is a length-``nbasis``
    vector and division is per-column; for all other modes the factor
    is a scalar applied to every column.

    Built by :func:`normalize`. Use the factory rather than
    constructing this class directly.

    Not a ``@dataclass`` (the :class:`FunctionHRF` precedent): ``base``
    and ``norm_mode`` are genuinely required constructor arguments, so
    an invalid instance is *unconstructible* rather than an
    ``Optional[...] = None`` field validated after the fact in
    ``__post_init__``. (A ``@dataclass`` cannot express a required
    field after the base's defaulted ``nbasis``/``span`` on Python 3.9
    without ``kw_only`` (3.10+); the typed custom ``__init__`` sidesteps
    that while keeping both mypy gates green.)
    """

    base: HRF
    norm_mode: NormMode
    norm_factor: Union[float, NDArray[np.float64]]

    def __init__(
        self,
        base: HRF,
        norm_mode: NormMode,
        norm_factor: Union[float, NDArray[np.float64]] = 1.0,
        name: str = "",
    ) -> None:
        # Derive identity from the base; the factory sets a label on top.
        super().__init__(
            name=name or f"{base.name}[norm={norm_mode}]",
            nbasis=base.nbasis,
            span=base.span,
        )
        self.base = base
        self.norm_mode = norm_mode
        self.norm_factor = norm_factor

    def __call__(self, t: ArrayLike) -> NDArray[np.float64]:
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
