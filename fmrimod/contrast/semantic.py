"""Authored semantic t-contrast intent over declared design provenance."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from numbers import Real

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

    def __add__(
        self,
        other: "ConditionRef | SemanticContrast | LinearSemanticContrast",
    ) -> "LinearSemanticContrast":
        return _as_linear(self).__add__(other)

    def __mul__(self, scalar: float) -> "LinearSemanticContrast":
        return _as_linear(self).__mul__(scalar)

    def __rmul__(self, scalar: float) -> "LinearSemanticContrast":
        return self.__mul__(scalar)

    def __neg__(self) -> "LinearSemanticContrast":
        return -_as_linear(self)

    @property
    def display_name(self) -> str:
        return self.level if self.term is None else f"{self.term}.{self.level}"


@dataclass(frozen=True)
class SemanticContrast:
    """A t-contrast authored over condition levels, not rendered columns."""

    positive: ConditionRef
    negative: ConditionRef | None = None
    name: str | None = None

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        if self.negative is None:
            return self.positive.display_name
        return f"{self.positive.display_name}_minus_{self.negative.display_name}"

    def resolve(self, columns: DesignColumns) -> NDArray[np.float64]:
        """Return a weight vector resolved against declared design provenance."""
        if not isinstance(columns, DesignColumns):
            raise TypeError(
                "SemanticContrast.resolve requires DesignColumns; got "
                f"{type(columns).__name__}. Pass fit.design_columns()."
            )

        positive = _select_condition(columns, self.positive, side="positive")
        negative: tuple[DesignColumn, ...] = ()
        if self.negative is not None:
            negative = _select_condition(columns, self.negative, side="negative")
            _ensure_disjoint(positive, negative)

        weights = np.zeros(len(columns), dtype=np.float64)
        for column in positive:
            weights[column.index] = 1.0 / len(positive)
        for column in negative:
            weights[column.index] = -1.0 / len(negative)
        return weights

    def __add__(
        self,
        other: "ConditionRef | SemanticContrast | LinearSemanticContrast",
    ) -> "LinearSemanticContrast":
        return _as_linear(self).__add__(other)

    def __sub__(
        self,
        other: "ConditionRef | SemanticContrast | LinearSemanticContrast",
    ) -> "LinearSemanticContrast":
        return _as_linear(self).__sub__(other)

    def __mul__(self, scalar: float) -> "LinearSemanticContrast":
        return _as_linear(self).__mul__(scalar)

    def __rmul__(self, scalar: float) -> "LinearSemanticContrast":
        return self.__mul__(scalar)

    def __neg__(self) -> "LinearSemanticContrast":
        return -_as_linear(self)

    def intent(self, *, rows: int = 1) -> dict[str, object]:
        """Return JSON-ready intent metadata for ``ContrastResult.explain``."""
        return {
            "kind": "semantic_contrast",
            "name": self.display_name,
            "positive": _ref_payload(self.positive),
            "negative": (
                None if self.negative is None else _ref_payload(self.negative)
            ),
            "term": (
                self.positive.term
                if self.negative is None
                or self.positive.term == self.negative.term
                else None
            ),
            "levels": (
                [self.positive.level]
                if self.negative is None
                else [self.positive.level, self.negative.level]
            ),
            "rows": rows,
        }


@dataclass(frozen=True)
class LinearSemanticContrast:
    """A linear combination of authored condition-level references.

    This is the small algebraic layer needed for factorial hypotheses such as
    ``0.5 * (A1 + A2 - B1 - B2)``. Resolution still goes through
    :class:`DesignColumns` and the same declared-provenance checks as
    :class:`SemanticContrast`; no rendered-name regex or positional lookup is
    introduced.
    """

    terms: tuple[tuple[ConditionRef, float], ...]
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terms",
            tuple((ref, float(weight)) for ref, weight in self.terms),
        )

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        pieces: list[str] = []
        for ref, weight in self.terms:
            sign = "+" if weight >= 0 else "-"
            magnitude = abs(weight)
            label = ref.display_name
            if magnitude == 1.0:
                piece = label
            else:
                piece = f"{magnitude:g}*{label}"
            if not pieces:
                pieces.append(piece if sign == "+" else f"-{piece}")
            else:
                pieces.append(f" {sign} {piece}")
        return "".join(pieces) if pieces else "semantic_linear_contrast"

    def named(self, name: str) -> "LinearSemanticContrast":
        """Return this contrast with a display/result name attached."""
        return LinearSemanticContrast(self.terms, name=name)

    def resolve(self, columns: DesignColumns) -> NDArray[np.float64]:
        """Return a weight vector resolved against declared design provenance."""
        if not isinstance(columns, DesignColumns):
            raise TypeError(
                "LinearSemanticContrast.resolve requires DesignColumns; got "
                f"{type(columns).__name__}. Pass fit.design_columns()."
            )
        weights = np.zeros(len(columns), dtype=np.float64)
        for ref, coefficient in self.terms:
            selected = _select_condition(columns, ref, side="linear")
            for column in selected:
                weights[column.index] += coefficient / len(selected)
        return weights

    def intent(self, *, rows: int = 1) -> dict[str, object]:
        """Return JSON-ready intent metadata for ``ContrastResult.explain``."""
        terms = [
            {**_ref_payload(ref), "weight": float(weight)}
            for ref, weight in self.terms
        ]
        levels = tuple(dict.fromkeys(ref.level for ref, _ in self.terms))
        term_names = {ref.term for ref, _ in self.terms}
        return {
            "kind": "semantic_linear_contrast",
            "name": self.display_name,
            "term": next(iter(term_names)) if len(term_names) == 1 else None,
            "levels": list(levels),
            "terms": terms,
            "rows": rows,
        }

    def __add__(
        self,
        other: "ConditionRef | SemanticContrast | LinearSemanticContrast",
    ) -> "LinearSemanticContrast":
        other_linear = _as_linear(other)
        return LinearSemanticContrast(
            _combine_terms((*self.terms, *other_linear.terms)),
            name=self.name,
        )

    def __sub__(
        self,
        other: "ConditionRef | SemanticContrast | LinearSemanticContrast",
    ) -> "LinearSemanticContrast":
        other_linear = -_as_linear(other)
        return LinearSemanticContrast(
            _combine_terms((*self.terms, *other_linear.terms)),
            name=self.name,
        )

    def __mul__(self, scalar: float) -> "LinearSemanticContrast":
        if not isinstance(scalar, Real):
            return NotImplemented
        return LinearSemanticContrast(
            tuple((ref, float(scalar) * weight) for ref, weight in self.terms),
            name=self.name,
        )

    def __rmul__(self, scalar: float) -> "LinearSemanticContrast":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "LinearSemanticContrast":
        if not isinstance(scalar, Real):
            return NotImplemented
        if float(scalar) == 0.0:
            raise ZeroDivisionError("cannot divide a semantic contrast by zero")
        return self.__mul__(1.0 / float(scalar))

    def __neg__(self) -> "LinearSemanticContrast":
        return LinearSemanticContrast(
            tuple((ref, -weight) for ref, weight in self.terms),
            name=self.name,
        )


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


def cell(
    term: str,
    *values: str,
    basis_ix: int | tuple[int, ...] | None = None,
    **levels: str,
) -> ConditionRef:
    """Author a factorial cell reference from factor-level assignments.

    For a two-factor term such as ``"task:valence"``, this returns the
    corresponding declared level label, e.g. ``cell("task:valence",
    task="recall", valence="emotional")`` resolves to
    ``"task.recall_valence.emotional"``. Single-factor terms route to
    :func:`condition`, so ``cell("trial_type", "gain")`` matches the same
    design provenance as ``condition("gain", term="trial_type")``.
    """
    if not isinstance(term, str) or not term:
        raise ValueError("cell() requires a non-empty term name")
    factors = tuple(part for part in term.split(":") if part)
    if not factors:
        raise ValueError("cell() requires at least one factor in term")
    if values and levels:
        raise ValueError("cell() accepts positional values or keyword levels, not both")
    if values:
        if len(values) != len(factors):
            raise ValueError(
                f"cell({term!r}) expected {len(factors)} values, got {len(values)}"
            )
        assignments = dict(zip(factors, values))
    else:
        missing = [factor for factor in factors if factor not in levels]
        extra = sorted(set(levels) - set(factors))
        if missing or extra:
            raise ValueError(
                f"cell({term!r}) level mismatch: missing={missing}, extra={extra}"
            )
        assignments = {factor: levels[factor] for factor in factors}

    if len(factors) == 1:
        return condition(str(assignments[factors[0]]), term=term, basis_ix=basis_ix)

    level = "_".join(f"{factor}.{assignments[factor]}" for factor in factors)
    return condition(level, term=term, basis_ix=basis_ix)


@dataclass(frozen=True)
class ModulatorRef:
    """Reference to a parametric modulator before it is scoped to a term."""

    name: str

    def within(self, term: str) -> "ScopedModulatorRef":
        """Scope this modulator to a factor/term that owns condition levels."""
        if not isinstance(term, str) or not term:
            raise ValueError("modulator(...).within(...) requires a non-empty term")
        return ScopedModulatorRef(name=self.name, term=term)


@dataclass(frozen=True)
class ScopedModulatorRef:
    """Parametric modulator scoped to one design term."""

    name: str
    term: str

    @property
    def parametric_term(self) -> str:
        return f"{self.term}:{self.name}"

    def slope(self, level: str) -> ConditionRef:
        """Reference the modulator slope column for one condition level."""
        if not isinstance(level, str) or not level:
            raise ValueError("slope(...) requires a non-empty level")
        return condition(level, term=self.parametric_term)

    def slopes(self, *levels: str) -> None:
        """Reserve the v2 F-contrast spelling recorded in the contract."""
        raise NotImplementedError(
            "Parametric F-contrast sugar is deferred to v2; see "
            "docs/contracts/parametric_contrast_sugar_v1.md"
        )

    def omnibus(self, *levels: str) -> None:
        """Reserve the v2 omnibus spelling recorded in the contract."""
        raise NotImplementedError(
            "Parametric F-contrast sugar is deferred to v2; see "
            "docs/contracts/parametric_contrast_sugar_v1.md"
        )


def modulator(name: str) -> ModulatorRef:
    """Author a parametric-modulator reference for semantic slope contrasts."""
    if not isinstance(name, str) or not name:
        raise ValueError("modulator(...) requires a non-empty modulator name")
    return ModulatorRef(name=name)


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
        _raise_missing_condition(columns, ref, side=side)

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


def _raise_missing_condition(
    columns: DesignColumns,
    ref: ConditionRef,
    *,
    side: str,
) -> None:
    if ref.term is not None and ":" in ref.term:
        factor, modulator_name = ref.term.split(":", 1)
        available_modulators = sorted(
            {
                str(column.term).split(":", 1)[1]
                for column in columns
                if column.term is not None
                and ":" in str(column.term)
                and str(column.term).split(":", 1)[0] == factor
            }
        )
        available_levels = sorted(
            {
                str(column.level)
                for column in columns
                if column.level is not None and column.term == ref.term
            }
        )
        suggestions = get_close_matches(modulator_name, available_modulators, n=3)
        suggestion_text = (
            f"; did you mean {suggestions[0]!r}?"
            if len(suggestions) == 1
            else (f"; did you mean one of {suggestions!r}?" if suggestions else "")
        )
        raise DesignProvenanceError(
            f"Parametric modulator {modulator_name!r} within term {factor!r} "
            f"could not resolve level {ref.level!r}{suggestion_text}. "
            f"Available modulators: {available_modulators!r}; "
            f"available levels for requested modulator: {available_levels!r}.",
            weak_fields=("term", "level"),
            repair_path=(
                "Use modulator(<available>).within(<term>).slope(<level>) "
                "or the lower-level condition(level, term='term:modulator') "
                "escape hatch."
            ),
        )

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


def _as_linear(
    value: ConditionRef | SemanticContrast | LinearSemanticContrast,
) -> LinearSemanticContrast:
    if isinstance(value, LinearSemanticContrast):
        return value
    if isinstance(value, ConditionRef):
        return LinearSemanticContrast(((value, 1.0),))
    if isinstance(value, SemanticContrast):
        terms: list[tuple[ConditionRef, float]] = [(value.positive, 1.0)]
        if value.negative is not None:
            terms.append((value.negative, -1.0))
        return LinearSemanticContrast(tuple(terms), name=value.name)
    return NotImplemented


def _combine_terms(
    terms: tuple[tuple[ConditionRef, float], ...],
) -> tuple[tuple[ConditionRef, float], ...]:
    combined: dict[ConditionRef, float] = {}
    order: list[ConditionRef] = []
    for ref, weight in terms:
        if ref not in combined:
            order.append(ref)
            combined[ref] = 0.0
        combined[ref] += float(weight)
    return tuple(
        (ref, combined[ref]) for ref in order if combined[ref] != 0.0
    )


__all__ = [
    "ConditionRef",
    "LinearSemanticContrast",
    "ModulatorRef",
    "SemanticContrast",
    "ScopedModulatorRef",
    "cell",
    "condition",
    "modulator",
]
