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

from typing import Any, Literal, Optional, Sequence

import pandas as pd

from ..hrf.core import HRF
from .terms import (
    Confounds,
    Drift,
    HrfTerm,
    Intercept,
    Predicate,
)


def hrf(
    *variables: str,
    basis: HRF | str = "spm",
    contrasts: Sequence[Any] = (),
    modulators: Sequence[Any] = (),
    durations: str | float | None = None,
    lag: float = 0.0,
    subset: Optional[Predicate] = None,
    prefix: Optional[str] = None,
    id: Optional[str] = None,
) -> HrfTerm:
    """Build an :class:`HrfTerm`.

    Examples
    --------
    >>> hrf("trial_type")
    >>> hrf("trial_type", basis="spmg3")
    >>> hrf("trial_type", "block", subset={"task": "memory"})
    """
    if not variables:
        raise ValueError("hrf() requires at least one variable name")
    return HrfTerm(
        variables=tuple(variables),
        hrf=basis,
        contrasts=tuple(contrasts),
        modulators=tuple(modulators),
        durations=durations,
        lag=lag,
        subset=subset,
        prefix=prefix,
        id=id,
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
    durations: str | float | None = None,
    lag: float = 0.0,
    subset: Optional[Predicate] = None,
    prefix: Optional[str] = None,
    id: Optional[str] = "trial",
) -> HrfTerm:
    """Build a per-trial beta-series term (LSS-friendly).

    Mirrors R's :func:`trialwise()` — produces one regressor per event.
    Realisation happens in :mod:`fmrimod.spec._compile`.
    """
    return HrfTerm(
        variables=("__trial__",),  # sentinel resolved at compile time
        hrf=basis,
        durations=durations,
        lag=lag,
        subset=subset,
        prefix=prefix,
        id=id,
    )
