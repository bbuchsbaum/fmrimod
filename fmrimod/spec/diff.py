"""Structural diff between two :class:`Spec` trees.

The terms.py module docstring promises that Spec/Term values are "frozen
so they can be hashed, pickled, and diffed". Equality and hashing come
for free from ``@dataclass(frozen=True)``; this module supplies the
*diff* half of that contract: a structured report of what changed
between two designs, suitable for inspection in tests and for human
review of analysis-spec edits.

The output is itself a typed value -- no formatted-string-only API --
so test code can assert on individual fields and downstream tooling
can serialize the diff alongside the fits it explains.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Callable, Hashable, List, Mapping, Tuple, Union, cast

import pandas as pd

from .terms import Confounds, Drift, HrfTerm, Intercept, Spec, Term, as_spec


@dataclass(frozen=True)
class FieldDiff:
    """One field's left/right values where they differ.

    ``left`` is the value from the spec on the left-hand side of the
    diff; ``right`` is the corresponding value from the other spec.
    Equal values are omitted from :class:`TermDiff.fields` rather than
    represented as a no-op ``FieldDiff``.
    """

    left: object
    right: object


@dataclass(frozen=True)
class TermDiff:
    """Field-level diff between two matched terms.

    The two terms are paired up by identity (see :func:`spec_diff` for
    the matching rules). ``fields`` contains only the fields whose
    values actually differ; a term-pair with no diverging fields shows
    up here with ``fields == {}`` only if the matcher emitted it for
    completeness -- ordinarily callers can read :attr:`is_empty`."""

    left: Term
    right: Term
    fields: Mapping[str, FieldDiff]

    @property
    def is_empty(self) -> bool:
        """True iff no fields differ between :attr:`left` and :attr:`right`."""
        return not self.fields


@dataclass(frozen=True)
class SpecDiff:
    """Structural diff between two :class:`Spec` trees.

    Events and baseline terms are diffed separately because they live
    in distinct slots on :class:`Spec`. For each slot the diff
    partitions terms three ways:

    - ``added_*``: present only on the right.
    - ``removed_*``: present only on the left.
    - ``changed_*``: matched on both sides with at least one field
      diverging.

    Matched-but-identical terms are omitted from ``changed_*``; they
    contribute no information to the diff and would only inflate the
    report.
    """

    added_events: Tuple[HrfTerm, ...] = ()
    removed_events: Tuple[HrfTerm, ...] = ()
    changed_events: Tuple[TermDiff, ...] = ()
    added_baseline: Tuple[Term, ...] = ()
    removed_baseline: Tuple[Term, ...] = ()
    changed_baseline: Tuple[TermDiff, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True iff the two specs are structurally indistinguishable."""
        return not (
            self.added_events
            or self.removed_events
            or self.changed_events
            or self.added_baseline
            or self.removed_baseline
            or self.changed_baseline
        )

    def __bool__(self) -> bool:
        return not self.is_empty

    def summary(self) -> str:
        """Return a multi-line, human-readable rendering of the diff.

        Intended for inspection in REPLs, logs, and test failure
        messages. Stable enough for tests to assert on key substrings,
        but the exact formatting is not part of the public contract.
        """
        if self.is_empty:
            return "(no differences)"
        lines: list[str] = []
        if self.removed_events:
            lines.append("events removed:")
            for t in self.removed_events:
                lines.append(f"  - {_term_summary(t)}")
        if self.added_events:
            lines.append("events added:")
            for t in self.added_events:
                lines.append(f"  + {_term_summary(t)}")
        for td in self.changed_events:
            lines.append(f"event changed: {_term_summary(td.left)}")
            for fname, fd in td.fields.items():
                lines.append(f"    {fname}: {fd.left!r} -> {fd.right!r}")
        if self.removed_baseline:
            lines.append("baseline removed:")
            for bt in self.removed_baseline:
                lines.append(f"  - {_term_summary(bt)}")
        if self.added_baseline:
            lines.append("baseline added:")
            for bt in self.added_baseline:
                lines.append(f"  + {_term_summary(bt)}")
        for td in self.changed_baseline:
            lines.append(f"baseline changed: {_term_summary(td.left)}")
            for fname, fd in td.fields.items():
                lines.append(f"    {fname}: {fd.left!r} -> {fd.right!r}")
        return "\n".join(lines)


def spec_diff(left: Union[Spec, Term], right: Union[Spec, Term]) -> SpecDiff:
    """Compute the structural diff between two :class:`Spec` trees.

    Both arguments are coerced through :func:`as_spec`, so a bare
    :class:`Term` (e.g. ``hrf("trial_type")`` on its own) is accepted
    on either side.

    Matching rules:

    - **Event terms (HrfTerm)** are matched first by :attr:`HrfTerm.id`
      when both sides set it, then by ``(variables, hrf-identity)``.
      Identical-key duplicates are paired in declaration order; any
      surplus on one side becomes ``added`` or ``removed``.
    - **Baseline terms** are matched by concrete type
      (``Drift`` / ``Intercept`` / ``Confounds``). Multiple terms of
      the same type pair by declaration order within type.

    Field-level comparison uses regular equality plus
    :meth:`pandas.DataFrame.equals` for any ``pd.DataFrame`` payload
    (e.g. ``Confounds.source``). Callable-valued predicates are
    compared by identity only, since arbitrary callables don't admit
    structural equality.
    """
    left_spec = as_spec(left)
    right_spec = as_spec(right)
    added_e, removed_e, changed_e = _diff_slot(
        left_spec.events, right_spec.events, _event_key
    )
    added_b, removed_b, changed_b = _diff_slot(
        left_spec.baseline, right_spec.baseline, _baseline_key
    )
    return SpecDiff(
        added_events=cast("Tuple[HrfTerm, ...]", added_e),
        removed_events=cast("Tuple[HrfTerm, ...]", removed_e),
        changed_events=changed_e,
        added_baseline=added_b,
        removed_baseline=removed_b,
        changed_baseline=changed_b,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _diff_slot(
    left_terms: Tuple[Term, ...],
    right_terms: Tuple[Term, ...],
    key_fn: Callable[[Term], Hashable],
) -> tuple[Tuple[Term, ...], Tuple[Term, ...], Tuple["TermDiff", ...]]:
    """Pair terms by key, diff matched pairs, return the three buckets."""
    left_buckets: dict[Hashable, List[Term]] = defaultdict(list)
    for t in left_terms:
        left_buckets[key_fn(t)].append(t)
    right_buckets: dict[Hashable, List[Term]] = defaultdict(list)
    for t in right_terms:
        right_buckets[key_fn(t)].append(t)

    # Preserve a stable visit order: keys in declaration order from
    # ``left`` first, then keys that appear only on ``right``.
    seen: set[Hashable] = set()
    ordered_keys: list[Hashable] = []
    for t in left_terms:
        k = key_fn(t)
        if k not in seen:
            ordered_keys.append(k)
            seen.add(k)
    for t in right_terms:
        k = key_fn(t)
        if k not in seen:
            ordered_keys.append(k)
            seen.add(k)

    added: list[Term] = []
    removed: list[Term] = []
    changed: list[TermDiff] = []
    for key in ordered_keys:
        ls = left_buckets.get(key, [])
        rs = right_buckets.get(key, [])
        paired = min(len(ls), len(rs))
        for i in range(paired):
            td = _diff_term(ls[i], rs[i])
            if not td.is_empty:
                changed.append(td)
        removed.extend(ls[paired:])
        added.extend(rs[paired:])

    return tuple(added), tuple(removed), tuple(changed)


def _diff_term(left: Term, right: Term) -> TermDiff:
    differences: dict[str, FieldDiff] = {}
    if type(left) is not type(right):
        # Should not happen given the matching keys, but stay safe.
        for f in fields(cast(Any, left)):
            differences[f.name] = FieldDiff(
                left=getattr(left, f.name), right=None
            )
        return TermDiff(left=left, right=right, fields=differences)
    for f in fields(cast(Any, left)):
        left_val = getattr(left, f.name)
        right_val = getattr(right, f.name)
        if not _values_equal(left_val, right_val):
            differences[f.name] = FieldDiff(left=left_val, right=right_val)
    return TermDiff(left=left, right=right, fields=differences)


def _values_equal(a: object, b: object) -> bool:
    if a is b:
        return True
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            return a.equals(b)
        return False
    if callable(a) or callable(b):
        # Arbitrary callables don't admit structural equality. ``is``
        # was already checked above, so two distinct callables are
        # always "different" as far as the diff is concerned.
        return a is b
    try:
        return bool(a == b)
    except (ValueError, TypeError):
        return False


def _event_key(term: Term) -> Hashable:
    """Identity key for matching HrfTerm-style event terms.

    Explicit ``id`` wins when set. Otherwise the ``variables`` tuple
    alone is the identity: an HRF-basis change on the same variables
    is the more useful read as a field-level diff, not as remove+add.
    Callers who need finer-grained matching should set ``id=``.
    """
    if not isinstance(term, HrfTerm):
        return ("type", type(term).__name__)
    if term.id is not None:
        return ("id", term.id)
    return ("vars", term.variables)


def _baseline_key(term: Term) -> Hashable:
    """Identity key for matching baseline terms.

    Two baseline terms of the same concrete type are considered the
    same logical slot. Multiple terms of the same type pair by
    declaration order within that type (handled by
    :func:`_diff_slot`).
    """
    return ("type", type(term).__name__)


def _term_summary(term: Term) -> str:
    if isinstance(term, HrfTerm):
        ident = term.id or "+".join(term.variables)
        return f"hrf({ident!r}, basis={term.hrf!r})"
    if isinstance(term, Drift):
        return f"drift({term.basis!r}, degree={term.degree})"
    if isinstance(term, Intercept):
        return f"intercept(per={term.per!r})"
    if isinstance(term, Confounds):
        return f"confounds(columns={term.columns!r})"
    return repr(term)
