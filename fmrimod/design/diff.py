"""Typed structural diff between two :class:`~fmrimod.model.FmriModel` values.

Operationalises VISION.md's commitment that "designs are first-class
objects that can be inspected, diffed, serialized, and re-fit". The
returned :class:`DesignDiff` is a sum type of frozen dataclasses - never
a ``dict[str, Any]`` - so callers can branch on what changed and
respond at the smallest structural unit (a single HRF parameter tweak
returns :class:`HRFDiff`, not a wholesale :class:`EventDiff`).

Bead: ``bd-01KRK97QMGJMH62H7NEXD14QGY``. The BIDS Stats Model translator
(``bd-01KRFKZ0J0Z0GJ9VK35ABCAGJW``) is the first consumer: the proof-row
red check there will assert ``design_diff(translated, refit) == NoDiff``
for a non-trivial JSON node, turning this primitive into a receipt
rather than ornamental decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model.fmri_model import FmriModel


@dataclass(frozen=True)
class NoDiff:
    """Two designs are structurally equivalent."""


@dataclass(frozen=True)
class TermAdded:
    """A term is in ``b`` but not in ``a``."""

    name: str


@dataclass(frozen=True)
class TermRemoved:
    """A term is in ``a`` but not in ``b``."""

    name: str


@dataclass(frozen=True)
class TermFieldChange:
    """A single typed field on a shared term has changed.

    ``field`` names the attribute (``"hrf"``, ``"basis"``, ``"events"``,
    ``"normalize"``, ``"summate"``); ``a_value`` / ``b_value`` carry the
    typed values from each side. The carrier preserves the values
    verbatim - no ``str(repr(...))`` coercion - so consumers can compare
    structurally if they need to.
    """

    field: str
    a_value: object
    b_value: object


@dataclass(frozen=True)
class TermChanged:
    """A term with the same ``name`` differs in one or more fields."""

    name: str
    changes: tuple[TermFieldChange, ...]


@dataclass(frozen=True)
class EventDiff:
    """Event-model differences at term granularity."""

    added: tuple[TermAdded, ...] = ()
    removed: tuple[TermRemoved, ...] = ()
    changed: tuple[TermChanged, ...] = ()


@dataclass(frozen=True)
class HRFParameterChange:
    """An HRF parameter on a shared term changed value."""

    term_name: str
    parameter: str
    a_value: object
    b_value: object


@dataclass(frozen=True)
class HRFKindChange:
    """The HRF *kind* (the dataclass class) on a shared term changed."""

    term_name: str
    a_kind: str
    b_kind: str


@dataclass(frozen=True)
class HRFDiff:
    """HRF-only changes carried at the smallest structural unit.

    ``kind_changes`` lists terms where the HRF class differs;
    ``parameter_changes`` lists terms where the kind matches but
    parameters differ. The two lists are disjoint by construction.
    """

    kind_changes: tuple[HRFKindChange, ...] = ()
    parameter_changes: tuple[HRFParameterChange, ...] = ()


@dataclass(frozen=True)
class BaselineDiff:
    """Baseline-model identity change.

    Baseline is treated as a single typed unit: either the two
    :class:`~fmrimod.baseline.BaselineModel` values compare ``==`` or
    they do not. Finer-grained baseline diffing is intentionally
    deferred to a follow-up that the BIDS receipt can drive.
    """

    a_repr: str
    b_repr: str


@dataclass(frozen=True)
class ColumnsDiff:
    """Column-name set differs between the two realized designs."""

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    reordered: bool = False


@dataclass(frozen=True)
class SamplingDiff:
    """Sampling frame (timepoints / runs) differs."""

    a_n_timepoints: tuple[int, ...]
    b_n_timepoints: tuple[int, ...]


@dataclass(frozen=True)
class Composite:
    """Multiple disjoint diffs collapsed into a single value."""

    parts: tuple["DesignDiffPart", ...]


DesignDiffPart = EventDiff | HRFDiff | BaselineDiff | ColumnsDiff | SamplingDiff
"""Non-empty diff variants - anything except ``NoDiff`` and ``Composite``."""

DesignDiff = NoDiff | DesignDiffPart | Composite
"""The closed sum type returned by :func:`design_diff`."""


# ---------------------------------------------------------------------------
# Component diffs
# ---------------------------------------------------------------------------


_TERM_FIELDS: tuple[str, ...] = (
    "events",
    "basis",
    "name",
    "normalize",
    "summate",
)
"""Term fields compared *outside* of HRF; HRF is routed through its own
diff variant so a parameter tweak doesn't masquerade as a wholesale
event change."""


def _term_name(term: object) -> str:
    name = getattr(term, "name", None)
    if name is not None:
        return str(name)
    events = getattr(term, "events", None)
    if isinstance(events, (list, tuple)):
        return ":".join(str(e) for e in events)
    return repr(term)


def _hrf_kind(hrf: object) -> str | None:
    if hrf is None:
        return None
    if isinstance(hrf, str):
        return f"name:{hrf}"
    return type(hrf).__name__


def _hrf_parameters(hrf: object) -> dict[str, object]:
    """Best-effort typed parameter mapping for an HRF value.

    Reads ``dataclasses.fields`` when the HRF is a dataclass; falls
    back to the legacy ``params`` mirror otherwise so designs that
    pre-date the typed HRF lift still diff cleanly.
    """

    if hrf is None or isinstance(hrf, str):
        return {}
    import dataclasses as _dc

    if _dc.is_dataclass(hrf):
        # Skip the back-compat ``params`` / ``param_names`` mirror that
        # the typed-HRF lift left on each subclass - the typed fields
        # already encode the same information, and reading both would
        # double-count every parameter change.
        return {
            f.name: getattr(hrf, f.name)
            for f in _dc.fields(hrf)
            if f.name not in {"name", "nbasis", "span", "params", "param_names"}
        }
    params = getattr(hrf, "params", None)
    if isinstance(params, dict):
        return dict(params)
    return {}


def _diff_hrfs(
    terms_by_name: dict[str, tuple[object, object]],
) -> HRFDiff:
    kind_changes: list[HRFKindChange] = []
    parameter_changes: list[HRFParameterChange] = []
    for tname, (ta, tb) in terms_by_name.items():
        ha = getattr(ta, "hrf", None)
        hb = getattr(tb, "hrf", None)
        ka, kb = _hrf_kind(ha), _hrf_kind(hb)
        if ka != kb:
            kind_changes.append(HRFKindChange(
                term_name=tname,
                a_kind=ka or "None",
                b_kind=kb or "None",
            ))
            continue
        pa = _hrf_parameters(ha)
        pb = _hrf_parameters(hb)
        for key in sorted(pa.keys() | pb.keys()):
            va = pa.get(key)
            vb = pb.get(key)
            if va != vb:
                parameter_changes.append(HRFParameterChange(
                    term_name=tname,
                    parameter=key,
                    a_value=va,
                    b_value=vb,
                ))
    return HRFDiff(
        kind_changes=tuple(kind_changes),
        parameter_changes=tuple(parameter_changes),
    )


def _diff_terms(a_terms: list, b_terms: list) -> tuple[EventDiff, HRFDiff]:
    """Split event-model term differences into EventDiff + HRFDiff parts."""

    a_by_name = {_term_name(t): t for t in a_terms}
    b_by_name = {_term_name(t): t for t in b_terms}

    added = tuple(
        TermAdded(name=n) for n in sorted(b_by_name.keys() - a_by_name.keys())
    )
    removed = tuple(
        TermRemoved(name=n) for n in sorted(a_by_name.keys() - b_by_name.keys())
    )

    shared_names = sorted(a_by_name.keys() & b_by_name.keys())
    shared_terms = {n: (a_by_name[n], b_by_name[n]) for n in shared_names}

    changed: list[TermChanged] = []
    for n, (ta, tb) in shared_terms.items():
        field_changes: list[TermFieldChange] = []
        for f in _TERM_FIELDS:
            va = getattr(ta, f, None)
            vb = getattr(tb, f, None)
            if va != vb:
                field_changes.append(TermFieldChange(field=f, a_value=va, b_value=vb))
        if field_changes:
            changed.append(TermChanged(name=n, changes=tuple(field_changes)))

    event = EventDiff(added=added, removed=removed, changed=tuple(changed))
    hrf = _diff_hrfs(shared_terms)
    return event, hrf


def _diff_baseline(a_bl: object, b_bl: object) -> BaselineDiff | None:
    equal = _baseline_signature(a_bl) == _baseline_signature(b_bl)
    if equal:
        return None
    return BaselineDiff(a_repr=repr(a_bl), b_repr=repr(b_bl))


def _baseline_signature(model: object) -> tuple[object, ...]:
    matrix = getattr(model, "design_matrix", None)
    columns = getattr(model, "column_names", None)
    if matrix is not None:
        arr = np.asarray(matrix, dtype=np.float64)
        return (
            "matrix",
            tuple(str(name) for name in (columns or ())),
            tuple(arr.shape),
            arr.tobytes(),
        )
    return ("repr", repr(model))


def _diff_columns(a_cols: list[str], b_cols: list[str]) -> ColumnsDiff | None:
    a_set = set(a_cols)
    b_set = set(b_cols)
    added = tuple(sorted(b_set - a_set))
    removed = tuple(sorted(a_set - b_set))
    if added or removed:
        return ColumnsDiff(added=added, removed=removed, reordered=False)
    if a_cols != b_cols:
        return ColumnsDiff(added=(), removed=(), reordered=True)
    return None


def _diff_sampling(a: FmriModel, b: FmriModel) -> SamplingDiff | None:
    a_tp = tuple(int(v) for v in a.n_timepoints)
    b_tp = tuple(int(v) for v in b.n_timepoints)
    if a_tp == b_tp:
        return None
    return SamplingDiff(a_n_timepoints=a_tp, b_n_timepoints=b_tp)


def _model_column_names(model: FmriModel) -> list[str]:
    design_matrix = getattr(model, "design_matrix", None)
    if callable(design_matrix):
        try:
            frame = design_matrix()
        except Exception:  # pragma: no cover - defensive legacy path
            frame = None
        columns = getattr(frame, "columns", None)
        if columns is not None:
            return [str(column) for column in columns]

    event_model = getattr(model, "event_model", None)
    return [str(column) for column in (getattr(event_model, "column_names", []) or [])]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _event_diff_is_empty(diff: EventDiff) -> bool:
    return not (diff.added or diff.removed or diff.changed)


def _hrf_diff_is_empty(diff: HRFDiff) -> bool:
    return not (diff.kind_changes or diff.parameter_changes)


def design_diff(a: FmriModel, b: FmriModel) -> DesignDiff:
    """Return the structural difference between two :class:`FmriModel` values.

    The result is :class:`NoDiff` when the two designs are structurally
    equivalent. Otherwise the smallest non-empty diff variant is
    returned; when several disjoint dimensions differ they are
    collapsed into a single :class:`Composite`.

    The comparison reads:

    * event-model term names, fields (``events``, ``basis``, ``name``,
      ``normalize``, ``summate``);
    * HRF kind + typed parameters per shared term, routed into
      :class:`HRFDiff`;
    * baseline model identity (``==``);
    * realized column names (added / removed / reordered);
    * sampling-frame timepoints.

    The dataset itself is *not* in the comparison: two models that
    differ only in the data they consume are equivalent designs.
    """

    a_em = a.event_model
    b_em = b.event_model
    a_terms = list(getattr(a_em, "terms", ()) or ())
    b_terms = list(getattr(b_em, "terms", ()) or ())
    event_diff, hrf_diff = _diff_terms(a_terms, b_terms)
    baseline_diff = _diff_baseline(a.baseline_model, b.baseline_model)
    a_cols = _model_column_names(a)
    b_cols = _model_column_names(b)
    columns_diff = _diff_columns(a_cols, b_cols)
    sampling_diff = _diff_sampling(a, b)

    parts: list[DesignDiffPart] = []
    if not _event_diff_is_empty(event_diff):
        parts.append(event_diff)
    if not _hrf_diff_is_empty(hrf_diff):
        parts.append(hrf_diff)
    if baseline_diff is not None:
        parts.append(baseline_diff)
    if columns_diff is not None:
        parts.append(columns_diff)
    if sampling_diff is not None:
        parts.append(sampling_diff)

    if not parts:
        return NoDiff()
    if len(parts) == 1:
        return parts[0]
    return Composite(parts=tuple(parts))


__all__ = [
    "BaselineDiff",
    "ColumnsDiff",
    "Composite",
    "DesignDiff",
    "DesignDiffPart",
    "EventDiff",
    "HRFDiff",
    "HRFKindChange",
    "HRFParameterChange",
    "NoDiff",
    "SamplingDiff",
    "TermAdded",
    "TermChanged",
    "TermFieldChange",
    "TermRemoved",
    "design_diff",
]
