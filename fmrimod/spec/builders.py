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

from typing import TYPE_CHECKING, Callable, Literal, Optional, Sequence, Union, cast

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
    span: Optional[float] = None,
    contrasts: Sequence[ContrastSpec] = (),
    modulators: Sequence[str] = (),
    center_modulators: bool = True,
    pool_modulators: bool = False,
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
    *variables
        Events-table column(s) whose levels (or values) become regressors,
        e.g. ``"trial_type"``.
    basis
        HRF name (``"spm"``, ``"spmg2"``, ``"spmg3"``, ``"fir"``,
        ``"bspline"``, ``"tent"``, ``"fourier"``, ...) or an :class:`HRF`.
    nbasis
        Number of basis functions for flexible bases (FIR bins, spline
        knots, ...).
    span
        Flexible bases only: the window in seconds the ``nbasis`` functions
        cover (default 24 s). For an FIR with one bin per scan, as Nilearn's
        ``fir_delays=range(n)``, pass ``nbasis=n, span=n * tr``.
    modulators
        Names of parametric-modulator columns in the events DataFrame.
        Each modulator expands into one additional regressor: the
        unmodulated boxcar scaled by the modulator's per-event value.
        Listing order is irrelevant — fmrimod does not orthogonalize
        modulators against each other.
    pool_modulators
        If True, each modulator is one regressor across all levels of the
        term (centered over all its events), contrasted as ``"rt"``; if
        False (default), one slope per level (``"trial_type:rt.face"``).
        Pooling is the model Nilearn builds from a ``modulation`` column.
    center_modulators
        Mean-center each modulator at the input-variable level (on
        the raw per-event scalar that becomes the boxcar amplitude),
        before convolution. Default ``True``, matching
        :class:`~fmrimod.events.EventVariable`'s ``center=True``
        default and the post-Mumford modern-correct workflow. Set to
        ``False`` for exact R ``fmridesign`` parity or when the
        modulator's absolute scale carries the analytic meaning. See
        the *parametric modulators* tutorial for the full discussion.
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
    Informed basis sets (``basis="spmg2"`` / ``"spmg3"``) follow R
    fmrihrf: the second column is the analytic time derivative
    ``dh/dt`` and the third (SPMG3) is SPM's dispersion difference.

    Examples
    --------
    >>> hrf("trial_type")
    >>> hrf("trial_type", basis="spmg3")
    >>> hrf("trial_type", basis="fir", nbasis=8)
    >>> hrf("trial_type", basis="fir", nbasis=8, span=16.0)  # 8 x 2 s bins
    >>> hrf("trial_type", norm="spm")  # Nilearn-compatible scale
    >>> hrf("trial_type", "block", subset={"task": "memory"})
    >>> hrf("trial_type", modulators=("rt", "accuracy"))  # auto-centered
    >>> hrf("trial_type", modulators=("rt",), center_modulators=False)
    >>> hrf("trial_type", modulators=("rt",), pool_modulators=True)  # one rt column
    """
    if not variables:
        raise ValueError("hrf() requires at least one variable name")
    if nbasis is not None and int(nbasis) < 1:
        raise ValueError("hrf(..., nbasis=) must be a positive integer")
    if span is not None:
        basis = _basis_with_span(basis, nbasis, float(span))
    return HrfTerm(
        variables=tuple(variables),
        hrf=basis,
        hrf_fun=hrf_fun,
        nbasis=None if nbasis is None else int(nbasis),
        contrasts=tuple(contrasts),
        modulators=tuple(modulators),
        center_modulators=bool(center_modulators),
        pool_modulators=bool(pool_modulators),
        durations=durations,
        lag=lag,
        subset=subset,
        prefix=prefix,
        id=id,
        norm=_validate_norm(norm),
        normalize=bool(normalize),
        summate=bool(summate),
    )


_SPAN_GENERATORS = {
    "fir": "hrf_fir_generator",
    "bspline": "hrf_bspline_generator",
    "bs": "hrf_bspline_generator",
    "tent": "hrf_tent_generator",
    "fourier": "hrf_fourier_generator",
    "daguerre": "hrf_daguerre_generator",
}


def _basis_with_span(basis: HRF | str, nbasis: Optional[int], span: float) -> HRF:
    """Build a flexible-basis HRF with an explicit window length."""
    if span <= 0:
        raise ValueError(f"hrf(..., span=) must be positive, got {span!r}")
    if not isinstance(basis, str) or basis.lower() not in _SPAN_GENERATORS:
        raise ValueError(
            f"hrf(..., span=) applies to flexible bases "
            f"{sorted(_SPAN_GENERATORS)}; got basis={basis!r}"
        )
    from .. import hrf as hrf_module

    generator = getattr(hrf_module, _SPAN_GENERATORS[basis.lower()])
    kwargs: dict[str, object] = {"span": span}
    if nbasis is not None:
        kwargs["nbasis"] = int(nbasis)
    return cast("HRF", generator(**kwargs))


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
    """Build a :class:`Drift` baseline term (slow-signal / high-pass model).

    Drift is modelled only when the spec includes this term; ``fmri_lm``
    otherwise fits just the per-run intercepts.

    Parameters
    ----------
    basis
        ``"cosine"`` (discrete cosine set, SPM/Nilearn high-pass),
        ``"poly"`` (Legendre-style polynomials), ``"bs"`` / ``"ns"``
        (B-/natural splines) or ``"constant"``. Each basis is built per run.
    cutoff
        Cosine only: high-pass period in seconds; frequencies slower than
        ``1 / cutoff`` Hz are modelled. ``None`` means 128 s (SPM's
        default). Nilearn's ``FirstLevelModel`` defaults to 100 s.
    degree
        Polynomial / spline degree (ignored for ``"cosine"``).

    Examples
    --------
    >>> spec = hrf("trial_type") + drift()   # cosine high-pass, 128 s cutoff
    >>> nilearn_like = drift("cosine", cutoff=100)
    >>> drift("poly", degree=3).basis
    'poly'
    """
    return Drift(basis=basis, degree=degree, cutoff=cutoff)


def intercept(per: Literal["run", "global", "none"] = "run") -> Intercept:
    """Build an :class:`Intercept` baseline term."""
    return Intercept(per=per)


def confounds(
    *columns: str,
    source: Optional[Union[pd.DataFrame, Sequence[pd.DataFrame]]] = None,
) -> Confounds:
    """Build a :class:`Confounds` baseline term (motion, physio, ...).

    Parameters
    ----------
    *columns
        Names of confound columns to include. Omit to use every column of
        ``source`` (e.g. a pre-selected fMRIPrep confounds table).
    source
        The confound values: one DataFrame covering all runs back to back,
        or a sequence of per-run DataFrames (as Nilearn's
        ``confounds=[...]``), each with the same columns. If omitted, the
        names are interpreted against the event-table's confound columns at
        compile time and ``columns`` are required.

    Examples
    --------
    >>> import pandas as pd
    >>> run1 = pd.DataFrame({"trans_x": [0.0, 0.1], "rot_z": [0.0, 0.01]})
    >>> run2 = pd.DataFrame({"trans_x": [0.2, 0.1], "rot_z": [0.02, 0.0]})
    >>> confounds(source=[run1, run2]).columns
    ('trans_x', 'rot_z')
    >>> confounds("trans_x", source=[run1, run2]).columns
    ('trans_x',)
    """
    if not columns:
        if source is None:
            raise ValueError(
                "confounds() needs column names or a `source` DataFrame "
                "(whose columns are then all used)"
            )
        first = source if isinstance(source, pd.DataFrame) else next(iter(source))
        columns = tuple(str(c) for c in first.columns)
    return Confounds(columns=tuple(columns), source=source)


def trialwise(
    basis: HRF | str = "spm",
    *,
    hrf_fun: Optional[Callable[..., object]] = None,
    nbasis: Optional[int] = None,
    condition: Optional[str] = None,
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

    Parameters
    ----------
    condition
        Optional name of an events-table column carrying the
        experimental-condition label for each trial (e.g.
        ``"trial_type"``). When set, each per-trial realised column
        carries the corresponding condition value on
        ``DesignColumn.condition`` / ``DesignColumn.level``, so MVPA
        pipelines can group per-trial betas by condition via typed
        lookup (``cols.where(role="task", condition="A")``) instead
        of parsing trial indices out of column names. When omitted,
        ``condition`` defaults to the zero-padded trial index
        (``"trial.01"``, ``"trial.02"``, ...).

    Examples
    --------
    LSS beta series, one map per trial, labelled by condition::

        betas = fm.estimate_single_trial_from_dataset(
            ds, trialwise(condition="trial_type"), method="lss",
            include_intercept=True,
        )

    Or as a beta-series GLM (LSA) through ``fmri_lm``::

        fit = fm.fmri_lm(trialwise() + drift(), ds)
    """
    if nbasis is not None and int(nbasis) < 1:
        raise ValueError("trialwise(..., nbasis=) must be a positive integer")
    term = HrfTerm(
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
    if condition is not None:
        # Stash on the typed kwargs bag — the compile path plumbs it
        # through to the EventModel term so trialwise condition labels
        # come from the events DataFrame rather than the
        # ``"trial_NN"`` zero-padded indices.
        object.__setattr__(
            term, "_trialwise_condition_col", str(condition)
        )
    return term
