"""Typed realized-design column provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, Mapping, cast

import pandas as pd

from ..design_colmap import design_colmap

ProvenanceGrade = Literal["declared", "derived", "inferred", "missing"]


@dataclass(frozen=True)
class DesignColumn:
    """One realized design column plus field-level provenance."""

    name: str
    index: int
    role: str
    model_source: str
    term: str | None = None
    term_tag: str | None = None
    term_index: int | None = None
    condition: str | None = None
    level: str | None = None
    basis_ix: int | None = None
    basis_name: str | None = None
    basis_total: int | None = None
    provenance: Mapping[str, ProvenanceGrade] | None = None

    def provenance_for(self, field: str) -> ProvenanceGrade:
        """Return the provenance grade for a field."""
        if self.provenance is None:
            return "missing"
        return self.provenance.get(field, "missing")


@dataclass(frozen=True)
class DesignColumns:
    """Typed realized-design provenance for a combined design matrix."""

    columns: tuple[DesignColumn, ...]

    def __iter__(self) -> Iterator[DesignColumn]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def __getitem__(self, index: int) -> DesignColumn:
        return self.columns[index]

    @property
    def names(self) -> tuple[str, ...]:
        """Column names in realized design order."""
        return tuple(column.name for column in self.columns)

    def terms(self) -> tuple[str, ...]:
        """Unique task-term names in design order.

        Baseline/intercept columns legitimately carry ``term=None``;
        crossed-term task columns carry a string term. Mixing the two
        makes the otherwise-natural ``sorted(c.term for c in columns)``
        raise ``TypeError: '<' not supported between 'NoneType' and
        'str'`` exactly where a caller introspects term names. This
        accessor is the None-safe way to read the task terms a design
        realized: ``None`` terms are excluded and order follows the
        design, so ``sorted(dc.terms())`` is always safe.
        """
        ordered: dict[str, None] = {}
        for column in self.columns:
            if column.term is not None:
                ordered.setdefault(column.term, None)
        return tuple(ordered)

    def where(
        self,
        *,
        term: str | None = None,
        condition: str | None = None,
        level: str | None = None,
        role: str | None = None,
        model_source: str | None = None,
        basis_ix: int | None = None,
    ) -> "DesignColumns":
        """Filter columns by semantic fields."""
        columns = self.columns
        if term is not None:
            columns = tuple(column for column in columns if column.term == term)
        if condition is not None:
            columns = tuple(
                column for column in columns if column.condition == condition
            )
        if level is not None:
            columns = tuple(column for column in columns if column.level == level)
        if role is not None:
            columns = tuple(column for column in columns if column.role == role)
        if model_source is not None:
            columns = tuple(
                column for column in columns if column.model_source == model_source
            )
        if basis_ix is not None:
            columns = tuple(
                column for column in columns if column.basis_ix == basis_ix
            )
        return DesignColumns(columns)

    def one(self) -> DesignColumn:
        """Return the only selected column."""
        if len(self.columns) != 1:
            raise ValueError(
                f"expected exactly one design column, got {len(self.columns)}"
            )
        return self.columns[0]

    @classmethod
    def from_model(cls, model: object) -> "DesignColumns":
        """Construct provenance for an object exposing event/baseline models."""
        event_model = getattr(model, "event_model", None)
        baseline_model = getattr(model, "baseline_model", None)
        if event_model is None or baseline_model is None:
            raise TypeError(
                "DesignColumns.from_model requires an FmriModel-like object"
            )

        event_columns = _event_columns(event_model)
        baseline_columns = _baseline_columns(baseline_model, offset=len(event_columns))
        columns = event_columns + baseline_columns

        design_matrix = getattr(model, "design_matrix", None)
        if callable(design_matrix):
            names = list(design_matrix(run=0).columns)
            columns = tuple(
                _replace_name(column, names[column.index])
                for column in columns
                if column.index < len(names)
            )
        return cls(tuple(columns))


def _replace_name(column: DesignColumn, name: str) -> DesignColumn:
    return DesignColumn(
        name=name,
        index=column.index,
        role=column.role,
        model_source=column.model_source,
        term=column.term,
        term_tag=column.term_tag,
        term_index=column.term_index,
        condition=column.condition,
        level=column.level,
        basis_ix=column.basis_ix,
        basis_name=column.basis_name,
        basis_total=column.basis_total,
        provenance=column.provenance,
    )


def _event_columns(event_model: object) -> tuple[DesignColumn, ...]:
    facts = getattr(event_model, "column_facts", None)
    if facts is not None:
        return tuple(_column_from_fact(fact, offset=0) for fact in facts)

    return tuple(_columns_from_colmap(design_colmap(event_model), offset=0))


def _baseline_columns(
    baseline_model: object,
    *,
    offset: int,
) -> tuple[DesignColumn, ...]:
    try:
        return tuple(_columns_from_colmap(design_colmap(baseline_model), offset=offset))
    except TypeError:
        design_matrix = getattr(baseline_model, "design_matrix", None)
        dm = design_matrix() if callable(design_matrix) else design_matrix
        if dm is None:
            return ()
        frame = dm if isinstance(dm, pd.DataFrame) else pd.DataFrame(dm)
        names = getattr(baseline_model, "column_names", None)
        if names is not None and len(list(names)) == frame.shape[1]:
            frame = frame.copy()
            frame.columns = list(names)
        return tuple(
            DesignColumn(
                name=str(name),
                index=offset + local_index,
                role="baseline",
                model_source="baseline",
                provenance={
                    "role": "inferred",
                    "term": "missing",
                    "condition": "missing",
                    "level": "missing",
                    "basis_ix": "missing",
                    "basis_name": "missing",
                    "basis_total": "missing",
                },
            )
            for local_index, name in enumerate(frame.columns)
        )


def _column_from_fact(fact: Mapping[str, object], *, offset: int) -> DesignColumn:
    index = int(cast(Any, fact["index"])) + offset
    provenance = fact.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    return DesignColumn(
        name=str(fact["name"]),
        index=index,
        role=str(fact.get("role") or "task"),
        model_source=str(fact.get("model_source") or "event"),
        term=_str_or_none(fact.get("term")),
        term_tag=_str_or_none(fact.get("term_tag")),
        term_index=_int_or_none(fact.get("term_index")),
        condition=_str_or_none(fact.get("condition")),
        level=_str_or_none(fact.get("level")),
        basis_ix=_int_or_none(fact.get("basis_ix")),
        basis_name=_str_or_none(fact.get("basis_name")),
        basis_total=_int_or_none(fact.get("basis_total")),
        provenance=dict(provenance),
    )


def _columns_from_colmap(
    colmap: pd.DataFrame,
    *,
    offset: int,
) -> Iterable[DesignColumn]:
    for row in colmap.to_dict("records"):
        source = row.get("model_source")
        inferred = {
            "term": _grade(row.get("term_tag"), "inferred"),
            "condition": _grade(row.get("condition"), "inferred"),
            "level": "missing",
            "basis_ix": _grade(row.get("basis_ix"), "inferred"),
            "basis_name": _grade(row.get("basis_name"), "derived"),
            "basis_total": _grade(row.get("basis_total"), "derived"),
            "role": _grade(row.get("role"), "inferred"),
        }
        yield DesignColumn(
            name=str(row["name"]),
            index=int(row["col"]) - 1 + offset,
            role=str(row.get("role") or "baseline"),
            model_source=str(source or "baseline"),
            term=_str_or_none(row.get("term_tag")),
            term_tag=_str_or_none(row.get("term_tag")),
            term_index=_int_or_none(row.get("term_index")),
            condition=_str_or_none(row.get("condition")),
            level=None,
            basis_ix=_int_or_none(row.get("basis_ix")),
            basis_name=_str_or_none(row.get("basis_name")),
            basis_total=_int_or_none(row.get("basis_total")),
            provenance=inferred,
        )


def _grade(value: object, present: ProvenanceGrade) -> ProvenanceGrade:
    if _str_or_none(value) is not None or _int_or_none(value) is not None:
        return present
    return "missing"


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(cast(Any, value)):
        return None
    return str(value)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if pd.isna(cast(Any, value)):
        return None
    return int(cast(Any, value))
