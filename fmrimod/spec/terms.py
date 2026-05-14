"""Typed Term and Spec dataclasses.

This is the canonical, IDE-introspectable representation of an fMRI design.
The string formula parser (``fmrimod.event_model("hrf(trial_type)")``) is
treated as sugar that compiles down to the same Spec tree.

Composition with ``+``::

    spec = (
        hrf("trial_type", basis="spm")
        + drift("cosine", cutoff=128)
        + intercept(per="run")
    )

Terms are frozen so they can be hashed, pickled, and diffed; the actual
design-matrix realisation happens via :func:`fmrimod.spec._compile.compile`.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

from ..hrf.core import HRF
from ..hrf.normalization import NormMode

if TYPE_CHECKING:
    from ..contrast.contrast_spec import ContrastSpec
    from .diff import SpecDiff

Predicate = Union[str, Mapping[str, Any], Callable[[pd.DataFrame], Any]]


# -- Base classes ------------------------------------------------------------


class Term(ABC):
    """Marker base class for spec terms.

    A ``Term`` describes one or more design columns. It is intentionally
    decoupled from realisation: a Term knows *what* it represents, not *how*
    to evaluate it.  Lowering happens in :func:`fmrimod.spec._compile.compile`.
    """

    # All concrete Terms are frozen dataclasses; the ABC only marks the type.

    def __add__(self, other: "Term | Spec") -> "Spec":
        """Compose terms into a :class:`Spec`."""
        return Spec(events=()).__add__(self).__add__(other)


@dataclass(frozen=True)
class HrfTerm(Term):
    """Event-related HRF term: variables convolved with an HRF basis.

    Parameters
    ----------
    variables
        One or more column names from the event table. Multiple names indicate
        an interaction term (Cartesian over factor levels).
    hrf
        HRF basis. Either an :class:`~fmrimod.hrf.HRF` instance or a registry
        key string (e.g. ``"spm"``, ``"spmg3"``, ``"gamma"``).
    hrf_fun
        Optional callable HRF implementation. When supplied, realization wraps
        it as a typed HRF and uses ``nbasis`` to declare its basis width.
    nbasis
        Optional basis count for generator-backed HRFs such as ``"fir"`` or
        custom ``hrf_fun`` callables.
    contrasts
        Optional :class:`~fmrimod.contrast.ContrastSpec` objects attached to
        this term (filled in by bd-01KRFMD3F66TENJMP6BQYE32HC).
    modulators
        Names of parametric-modulator columns from the event table. Each
        modulator expands into one additional event-model term during
        compilation, mirroring the legacy ``trial_type:hrf(...) +
        trial_type:rt:hrf(...)`` formula path. The main term keeps any
        declared ``contrasts``; the parametric terms inherit the same HRF,
        ``durations``, ``lag``, ``subset``, ``prefix``, ``normalize`` and
        ``summate``.
    durations
        Per-event duration; either a column name in the event table or a
        scalar.  ``None`` defers to the spec-level default.
    lag
        Temporal offset in seconds applied to the HRF before convolution.
    subset
        Optional ``where`` predicate restricting the events used for this
        term. Accepts a dict (``{"block": 1}``), a string predicate
        (``"block <= 3"``), or a callable.
    prefix
        Optional column-name prefix.
    id
        Optional explicit term identifier (otherwise derived from variables).
    norm
        Optional fixed HRF normalization mode. ``"spm"`` uses the Nilearn
        SPM reference-grid scale; ``"unit_peak"`` and ``"unit_integral"``
        provide explicit amplitude/integral normalization.
    normalize
        If True, peak-normalize realised regressors after convolution.
    summate
        Whether overlapping HRF responses are summed during convolution.
    """

    variables: Tuple[str, ...]
    hrf: Union[HRF, str] = "spm"
    hrf_fun: Optional[Callable[..., object]] = None
    nbasis: Optional[int] = None
    contrasts: Tuple[ContrastSpec, ...] = ()
    modulators: Tuple[str, ...] = ()
    durations: Union[str, float, None] = None
    lag: float = 0.0
    subset: Optional[Predicate] = None
    prefix: Optional[str] = None
    id: Optional[str] = None
    norm: Optional[NormMode] = None
    normalize: bool = False
    summate: bool = True


@dataclass(frozen=True)
class Drift(Term):
    """Polynomial / spline / cosine drift basis.

    Compiled into the baseline block of :func:`baseline_model`.
    """

    basis: Literal["constant", "poly", "bs", "ns", "cosine"] = "constant"
    degree: int = 1
    cutoff: Optional[float] = None  # high-pass cutoff for cosine drift


@dataclass(frozen=True)
class Intercept(Term):
    """Block / global / suppressed intercept."""

    per: Literal["run", "global", "none"] = "run"


@dataclass(frozen=True)
class Confounds(Term):
    """Nuisance regressors (motion, physio, etc.)."""

    columns: Tuple[str, ...]
    source: Optional[pd.DataFrame] = None


# -- Composition container ---------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """An ordered collection of event and baseline terms.

    Use the ``+`` operator to compose:

    >>> spec = hrf("trial_type") + drift("cosine", cutoff=128) + intercept()

    The container distinguishes *event* terms (HrfTerm, parametric variants)
    from *baseline* terms (Drift, Intercept, Confounds). Composition routes
    each term to the correct bucket automatically.
    """

    events: Tuple[Term, ...] = ()
    baseline: Tuple[Term, ...] = ()

    @staticmethod
    def _is_event_term(term: Term) -> bool:
        return isinstance(term, HrfTerm)

    @staticmethod
    def _is_baseline_term(term: Term) -> bool:
        return isinstance(term, (Drift, Intercept, Confounds))

    def __add__(self, other: "Term | Spec") -> "Spec":
        if isinstance(other, Spec):
            return Spec(
                events=self.events + other.events,
                baseline=self.baseline + other.baseline,
            )
        if isinstance(other, Term):
            if Spec._is_event_term(other):
                return Spec(events=self.events + (other,), baseline=self.baseline)
            if Spec._is_baseline_term(other):
                return Spec(events=self.events, baseline=self.baseline + (other,))
            raise TypeError(f"Unrecognised Term subtype: {type(other).__name__}")
        return NotImplemented

    def __radd__(self, other: "Term") -> "Spec":
        if isinstance(other, Term):
            return Spec().__add__(other).__add__(self)
        return NotImplemented

    def __iter__(self) -> Iterator[Term]:
        yield from self.events
        yield from self.baseline

    def __len__(self) -> int:
        return len(self.events) + len(self.baseline)

    @property
    def terms(self) -> Tuple[Term, ...]:
        """All terms, events then baseline, in declaration order."""
        return self.events + self.baseline

    def diff(self, other: "Spec") -> "SpecDiff":
        """Return a structural :class:`SpecDiff` against ``other``.

        Pure equality (``self == other``) only answers yes/no; this
        helper reports *what* changed: added/removed/changed terms and
        the specific fields that diverge on each changed term. See
        :func:`fmrimod.spec.diff.spec_diff` for the matching rules.
        """
        from .diff import spec_diff

        return spec_diff(self, other)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this :class:`Spec` to a JSON-safe dict.

        The inverse of :meth:`from_dict`. See
        :mod:`fmrimod.spec.serialize` for the supported value set and
        the explicit-error behavior on shapes that don't round-trip
        (e.g. inline callables, DataFrame payloads).
        """
        from .serialize import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Spec":
        """Reconstruct a :class:`Spec` from :meth:`to_dict` output."""
        from .serialize import from_dict

        return from_dict(payload)


# -- Utility -----------------------------------------------------------------


def is_spec(obj: Any) -> bool:
    """Return True iff *obj* is a :class:`Spec` or a single :class:`Term`."""
    return isinstance(obj, (Spec, Term))


def as_spec(obj: Spec | Term | Sequence[Term]) -> Spec:
    """Coerce a Term, list of Terms, or Spec into a canonical :class:`Spec`."""
    if isinstance(obj, Spec):
        return obj
    if isinstance(obj, Term):
        return Spec() + obj
    if isinstance(obj, (list, tuple)):
        out = Spec()
        for t in obj:
            if not isinstance(t, Term):
                raise TypeError(
                    f"as_spec: expected Term in sequence, got {type(t).__name__}"
                )
            out = out + t
        return out
    raise TypeError(f"as_spec: cannot coerce {type(obj).__name__} to Spec")
