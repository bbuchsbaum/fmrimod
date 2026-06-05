"""Typed realized-design column provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Mapping

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

    def cell(
        self,
        *,
        term: str | None = None,
        **factor_values: str,
    ) -> DesignColumn:
        """Look up a single factorial cell by factor-value keyword arguments.

        For a factorial term like ``"trial_type:difficulty:context"`` the
        realised level string follows the convention
        ``"trial_type.A_difficulty.easy_context.off"`` (factor names
        and values may contain underscores). This helper parses each
        column's level string against the known factor list (from the
        ``term`` field) and returns the column whose factor values
        match every supplied keyword.

        Parameters
        ----------
        term
            Optional explicit term name. When omitted, the term is
            taken from the unique task term whose factors match
            the supplied keyword names (works when the design has
            exactly one factorial term covering those factors).
        **factor_values
            One keyword per factor (e.g. ``trial_type="A"``,
            ``difficulty="easy"``, ``context="off"``). All must match.

        Returns
        -------
        DesignColumn
            The unique cell matching the supplied factor values.

        Raises
        ------
        KeyError
            No cell matched.
        ValueError
            More than one cell matched, or the term couldn't be
            inferred when ``term=None``.
        """
        if not factor_values:
            raise ValueError(
                "DesignColumns.cell requires at least one factor=value keyword"
            )
        # Resolve the term if not given: find the unique task term whose
        # ``factor1:factor2:...`` decomposition covers every supplied
        # keyword name.
        resolved_term = term
        if resolved_term is None:
            wanted = set(factor_values.keys())
            candidates = {
                col.term for col in self.columns
                if col.term is not None and ":" in (col.term or "")
            }
            matches = [
                t for t in candidates
                if wanted.issubset(set(t.split(":")))
            ]
            if not matches:
                raise ValueError(
                    f"DesignColumns.cell: no factorial term covers factors "
                    f"{sorted(wanted)!r}; available terms: "
                    f"{sorted(candidates)!r}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"DesignColumns.cell: factors {sorted(wanted)!r} are "
                    f"covered by multiple terms {sorted(matches)!r}; pass "
                    f"``term=`` explicitly"
                )
            resolved_term = matches[0]

        factor_order = resolved_term.split(":")
        unknown = [k for k in factor_values if k not in factor_order]
        if unknown:
            raise ValueError(
                f"DesignColumns.cell: factor(s) {unknown!r} not in term "
                f"{resolved_term!r} (available factors: {factor_order!r})"
            )

        # Filter to columns from the resolved term, then match by parsed
        # factor-value mapping.
        candidates = [c for c in self.columns if c.term == resolved_term]
        matches: list[DesignColumn] = []
        for c in candidates:
            parsed = _parse_factorial_level(c.level, factor_order)
            if parsed is None:
                continue
            if all(parsed.get(k) == v for k, v in factor_values.items()):
                matches.append(c)
        if not matches:
            raise KeyError(
                f"DesignColumns.cell: no column matched factor values "
                f"{factor_values!r} on term {resolved_term!r}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"DesignColumns.cell: {len(matches)} columns matched "
                f"{factor_values!r} on term {resolved_term!r}; expected 1"
            )
        return matches[0]

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


def _parse_factorial_level(
    level: str | None, factor_order: list[str]
) -> dict[str, str] | None:
    """Parse a factorial level string into a ``{factor: value}`` mapping.

    The level-string convention from the typed factorial expansion is
    ``"<f1>.<v1>_<f2>.<v2>_<f3>.<v3>"`` where factor names and
    values may themselves contain underscores. Anchoring on the
    known factor sequence avoids the ambiguity.
    """
    if level is None or not factor_order:
        return None
    text = level
    prefix = factor_order[0] + "."
    if not text.startswith(prefix):
        return None
    text = text[len(prefix):]
    parsed: dict[str, str] = {}
    for idx, factor in enumerate(factor_order[:-1]):
        next_marker = "_" + factor_order[idx + 1] + "."
        split_at = text.find(next_marker)
        if split_at < 0:
            return None
        parsed[factor] = text[:split_at]
        text = text[split_at + len(next_marker):]
    # The remaining ``text`` is the last factor's value.
    parsed[factor_order[-1]] = text
    return parsed


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
    index = int(fact["index"]) + offset
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
    if pd.isna(value):
        return None
    return str(value)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return int(value)
