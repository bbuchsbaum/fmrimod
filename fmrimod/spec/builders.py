"""Top-level builder functions for :mod:`fmrimod.spec` terms.

These are the user-facing entry points::

    fm.hrf("trial_type", basis="spm")
    fm.drift("cosine", cutoff=128)
    fm.intercept(per="run")
    fm.confounds("trans_x", "rot_z", source=confounds_df)
    fm.trialwise(basis="spm")

They return frozen :class:`Term` dataclasses; compose with ``+``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional, Sequence, cast

import pandas as pd

from ..hrf.core import HRF
from ..hrf.normalization import VALID_NORM_MODES, NormMode
from .terms import (
    Confounds,
    CovariateTerm,
    Drift,
    HrfTerm,
    Intercept,
    Predicate,
)

if TYPE_CHECKING:
    from ..contrast.contrast_spec import ContrastSpec


def _validate_norm(norm: str | None) -> NormMode | None:
    if norm is None:
        return None
    if norm not in VALID_NORM_MODES:
        expected = "', '".join(VALID_NORM_MODES)
        raise ValueError(
            f"Unknown HRF normalization mode {norm!r}. "
            f"Expected one of: '{expected}'."
        )
    return cast(NormMode, norm)


def hrf(
    *variables: str,
    basis: HRF | str = "spm",
    hrf_fun: Optional[Callable[..., object]] = None,
    nbasis: Optional[int] = None,
    contrasts: Sequence[ContrastSpec] = (),
    modulators: Sequence[str] = (),
    durations: str | float | None = None,
    lag: float = 0.0,
    subset: Optional[Predicate] = None,
    prefix: Optional[str] = None,
    id: Optional[str] = None,
    norm: Optional[NormMode] = None,
    normalize: bool = False,
    summate: bool = True,
) -> HrfTerm:
    """Build an :class:`HrfTerm`.

    Parameters
    ----------
    norm
        HRF normalization mode. Leave as ``None`` for the raw canonical
        scale.

        - ``"spm"`` — divide by the *sum* of the HRF evaluated on
          Nilearn's reference grid (``time_length=32``,
          ``oversampling=50``). Matches Nilearn's ``hrf_model="spm"``
          scale.
        - ``"unit_peak"`` — divide by the absolute peak of the HRF on
          its span. Mirrors R's ``normalise_hrf``.
        - ``"unit_integral"`` — divide by the trapezoidal integral so
          the continuous integral equals 1.
        - ``"unit_peak_per_basis"`` — multi-basis only: each column is
          divided by *its own* absolute peak so every column has unit
          peak. Use when the basis columns are heterogeneously scaled
          (e.g. an ``as_hrf`` wrapper around a hand-built basis).

        **Multi-basis behaviour (SPMG2 / SPMG3 / FIR / B-spline).**
        ``"spm"``, ``"unit_peak"``, and ``"unit_integral"`` all derive a
        *single* scalar from the canonical column (column 0) on the
        reference grid and apply that same scalar uniformly to every
        basis column. This preserves the physical interpretation of the
        derivative / dispersion columns as latency / shape *perturbations*
        of a fixed-amplitude canonical — ``β_derivative`` units stay
        commensurate with ``β_canonical`` units. ``"unit_peak_per_basis"``
        rescales each column independently, which breaks that
        interpretation but is useful when the columns are not basis
        functions in the SPM sense (e.g. an arbitrary multi-channel
        kernel). For ``basis="spmg2"`` / ``"spmg3"``, prefer ``"spm"``
        (Nilearn-compatible) or ``"unit_peak"`` (R-aligned) over
        ``"unit_peak_per_basis"``.

    Notes
    -----
    Informed basis sets (``basis="spmg2"`` / ``"spmg3"``) use **closed-form**
    temporal derivatives, not the finite-difference derivatives that
    Nilearn (``hrf_model="spm + derivative"``) and SPM12 produce. This is
    a deliberate divergence — see :class:`fmrimod.hrf.SPMG2_HRF` for the
    full rationale and the downstream latency-calibration caveat
    (``β_derivative / β_canonical`` ratios do **not** transfer from
    SPM12-published constants without re-derivation).

    Examples
    --------
    >>> hrf("trial_type")
    >>> hrf("trial_type", basis="spmg3")
    >>> hrf("trial_type", basis="fir", nbasis=8)
    >>> hrf("trial_type", norm="spm")  # Nilearn-compatible scale
    >>> hrf("trial_type", "block", subset={"task": "memory"})
    """
    if not variables:
        raise ValueError("hrf() requires at least one variable name")
    if nbasis is not None and int(nbasis) < 1:
        raise ValueError("hrf(..., nbasis=) must be a positive integer")
    return HrfTerm(
        variables=tuple(variables),
        hrf=basis,
        hrf_fun=hrf_fun,
        nbasis=None if nbasis is None else int(nbasis),
        contrasts=tuple(contrasts),
        modulators=tuple(modulators),
        durations=durations,
        lag=lag,
        subset=subset,
        prefix=prefix,
        id=id,
        norm=_validate_norm(norm),
        normalize=bool(normalize),
        summate=bool(summate),
    )


def covariate(
    *variables: str,
    source: Optional[pd.DataFrame] = None,
    prefix: Optional[str] = None,
    id: Optional[str] = None,
) -> CovariateTerm:
    """Build an unconvolved sampled-covariate event term.

    Mirrors R ``fmridesign::covariate()``: the variables are treated as
    sampled time courses whose HRF is the identity/no-op function. Use this
    for seed signals, motion traces, physiological traces, or any regressor
    that already lives on the BOLD sampling grid and should remain in the
    event-model design rather than the baseline block.
    """
    if not variables:
        raise ValueError("covariate() requires at least one variable name")
    if source is not None:
        missing = [name for name in variables if name not in source.columns]
        if missing:
            raise ValueError(f"Covariate columns not found in source: {missing}")
    return CovariateTerm(
        variables=tuple(variables),
        hrf="identity",
        source=source,
        prefix=prefix,
        id=id,
        normalize=False,
        summate=False,
    )


def drift(
    basis: Literal["constant", "poly", "bs", "ns", "cosine"] = "cosine",
    *,
    cutoff: Optional[float] = None,
    degree: int = 1,
) -> Drift:
    """Build a :class:`Drift` baseline term.

    Examples
    --------
    >>> drift("cosine", cutoff=128)
    >>> drift("poly", degree=3)
    """
    return Drift(basis=basis, degree=degree, cutoff=cutoff)


def intercept(per: Literal["run", "global", "none"] = "run") -> Intercept:
    """Build an :class:`Intercept` baseline term."""
    return Intercept(per=per)


def confounds(
    *columns: str,
    source: Optional[pd.DataFrame] = None,
) -> Confounds:
    """Build a :class:`Confounds` baseline term.

    Parameters
    ----------
    *columns
        Names of confound columns to include.
    source
        Optional DataFrame providing the confound values. If omitted, the
        names are interpreted against the event-table's confound columns at
        compile time.
    """
    if not columns:
        raise ValueError("confounds() requires at least one column name")
    return Confounds(columns=tuple(columns), source=source)


def trialwise(
    basis: HRF | str = "spm",
    *,
    hrf_fun: Optional[Callable[..., object]] = None,
    nbasis: Optional[int] = None,
    durations: str | float | None = None,
    lag: float = 0.0,
    subset: Optional[Predicate] = None,
    prefix: Optional[str] = None,
    id: Optional[str] = "trial",
    norm: Optional[NormMode] = None,
    normalize: bool = False,
    summate: bool = True,
) -> HrfTerm:
    """Build a per-trial beta-series term (LSS-friendly).

    Mirrors R's :func:`trialwise()` — produces one regressor per event.
    Realisation happens in :mod:`fmrimod.spec._compile`. See
    :func:`hrf` for the ``norm`` argument.
    """
    if nbasis is not None and int(nbasis) < 1:
        raise ValueError("trialwise(..., nbasis=) must be a positive integer")
    return HrfTerm(
        variables=("__trial__",),  # sentinel resolved at compile time
        hrf=basis,
        hrf_fun=hrf_fun,
        nbasis=None if nbasis is None else int(nbasis),
        durations=durations,
        lag=lag,
        subset=subset,
        prefix=prefix,
        id=id,
        norm=_validate_norm(norm),
        normalize=bool(normalize),
        summate=bool(summate),
    )
