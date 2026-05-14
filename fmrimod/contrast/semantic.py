"""Authored semantic t-contrast intent over declared design provenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..design.columns import DesignColumn, DesignColumns
from .errors import DesignProvenanceError


@dataclass(frozen=True)
class ConditionRef:
    """Reference to an authored condition level in a realized design."""

    level: str
    term: str | None = None
    basis_ix: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.basis_ix, tuple):
            object.__setattr__(self, "basis_ix", tuple(self.basis_ix))

    def __sub__(self, other: "ConditionRef") -> "SemanticContrast":
        if not isinstance(other, ConditionRef):
            return NotImplemented
        name = f"{self.display_name}_minus_{other.display_name}"
        return SemanticContrast(positive=self, negative=other, name=name)

    @property
    def display_name(self) -> str:
        return self.level if self.term is None else f"{self.term}.{self.level}"


@dataclass(frozen=True)
class SemanticContrast:
    """A t-contrast authored over condition levels, not rendered columns."""

    positive: ConditionRef
    negative: ConditionRef
    name: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or (
            f"{self.positive.display_name}_minus_{self.negative.display_name}"
        )

    def resolve(self, columns: DesignColumns) -> NDArray[np.float64]:
        """Return a weight vector resolved against declared design provenance."""
        if not isinstance(columns, DesignColumns):
            raise TypeError(
                "SemanticContrast.resolve requires DesignColumns; got "
                f"{type(columns).__name__}. Pass fit.design_columns()."
            )

        positive = _select_condition(columns, self.positive, side="positive")
        negative = _select_condition(columns, self.negative, side="negative")
        _ensure_disjoint(positive, negative)

        weights = np.zeros(len(columns), dtype=np.float64)
        for column in positive:
            weights[column.index] = 1.0 / len(positive)
        for column in negative:
            weights[column.index] = -1.0 / len(negative)
        return weights

    def intent(self, *, rows: int = 1) -> dict[str, object]:
        """Return JSON-ready intent metadata for ``ContrastResult.explain``."""
        return {
            "kind": "semantic_contrast",
            "name": self.display_name,
            "positive": _ref_payload(self.positive),
            "negative": _ref_payload(self.negative),
            "term": (
                self.positive.term
                if self.positive.term == self.negative.term
                else None
            ),
            "levels": [self.positive.level, self.negative.level],
            "rows": rows,
        }


def condition(
    level: str,
    *,
    term: str | None = None,
    basis_ix: int | tuple[int, ...] | None = None,
) -> ConditionRef:
    """Author a condition-level reference for semantic contrasts.

    Examples
    --------
    ``condition("gain") - condition("loss")`` creates a contrast that resolves
    against declared design-column provenance rather than rendered column order.
    """
    basis: tuple[int, ...]
    if basis_ix is None:
        basis = ()
    elif isinstance(basis_ix, tuple):
        basis = basis_ix
    else:
        basis = (int(basis_ix),)
    return ConditionRef(level=level, term=term, basis_ix=basis)


def _select_condition(
    columns: DesignColumns,
    ref: ConditionRef,
    *,
    side: str,
) -> tuple[DesignColumn, ...]:
    candidates = tuple(column for column in columns if column.level == ref.level)
    if ref.term is not None:
        candidates = tuple(column for column in candidates if column.term == ref.term)
    if ref.basis_ix:
        wanted = set(ref.basis_ix)
        candidates = tuple(column for column in candidates if column.basis_ix in wanted)
    elif len(candidates) > 1:
        basis = sorted(
            {
                column.basis_ix
                for column in candidates
                if column.basis_ix is not None
            }
        )
        raise DesignProvenanceError(
            f"SemanticContrast {side} condition {ref.display_name!r} is "
            f"ambiguous across basis columns={basis!r}",
            weak_fields=("basis_ix",),
            repair_path=(
                "pass basis_ix=... to condition(...) or use an F/omnibus "
                "contrast for multi-basis condition effects."
            ),
        )

    if not candidates:
        available = sorted(
            {
                column.level
                for column in columns
                if column.level is not None
                and (ref.term is None or column.term == ref.term)
            }
        )
        raise DesignProvenanceError(
            f"SemanticContrast {side} condition {ref.display_name!r} "
            f"matched no declared design columns; available levels={available!r}",
            weak_fields=("level",),
            repair_path=(
                "author the contrast against a level emitted by the typed design "
                "compiler, or carry declared level provenance into DesignColumns."
            ),
        )

    if ref.term is None:
        terms = {column.term for column in candidates}
        if len(terms) != 1:
            raise DesignProvenanceError(
                f"SemanticContrast {side} condition {ref.level!r} is ambiguous "
                f"across terms={sorted(str(term) for term in terms)!r}",
                weak_fields=("term",),
                repair_path=(
                    "pass term=... to condition(...) so the authored hypothesis "
                    "does not depend on rendered column names."
                ),
            )

    weak = _weak_provenance(candidates, require_term=ref.term is not None)
    if weak:
        field_names = tuple(sorted({field for field, _, _ in weak}))
        details = ", ".join(
            f"{column}.{field}={grade}" for field, column, grade in weak
        )
        raise DesignProvenanceError(
            f"SemanticContrast {side} condition {ref.display_name!r} cannot "
            f"resolve: required provenance is weaker than declared. "
            f"weak_fields={field_names}; details={details}",
            weak_fields=field_names,
            repair_path=(
                "propagate authored term/level facts into DesignColumns; do not "
                "fall back to regex over rendered column names."
            ),
        )
    return candidates


def _weak_provenance(
    columns: tuple[DesignColumn, ...],
    *,
    require_term: bool,
) -> list[tuple[str, str, str]]:
    weak: list[tuple[str, str, str]] = []
    for column in columns:
        level_grade = column.provenance_for("level")
        if level_grade != "declared":
            weak.append(("level", column.name, level_grade))
        if require_term:
            term_grade = column.provenance_for("term")
            if term_grade != "declared":
                weak.append(("term", column.name, term_grade))
    return weak


def _ensure_disjoint(
    positive: tuple[DesignColumn, ...],
    negative: tuple[DesignColumn, ...],
) -> None:
    overlap = {column.index for column in positive} & {
        column.index for column in negative
    }
    if overlap:
        raise DesignProvenanceError(
            f"SemanticContrast positive and negative conditions overlap at "
            f"indices={sorted(overlap)!r}",
            weak_fields=("level",),
            repair_path="author disjoint positive and negative condition references.",
        )


def _ref_payload(ref: ConditionRef) -> dict[str, object]:
    return {
        "level": ref.level,
        "term": ref.term,
        "basis_ix": list(ref.basis_ix),
    }


__all__ = ["ConditionRef", "SemanticContrast", "condition"]
