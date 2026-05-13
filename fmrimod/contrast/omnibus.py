"""Typed F-contrast intent that resolves against typed design provenance.

`OmnibusContrast` is the first typed contrast-intent value: it names a modeled
term and (optionally) a subset of its factor levels, and resolves into a
realized F-contrast weight matrix only when the realized design columns carry
*declared* term/level provenance. If the provenance is only inferred or
missing, resolution fails with :class:`DesignProvenanceError` instead of
silently falling back to regex or raw column positions.

This deliberately is *additive*: existing array/string/dict/Formula contrast
paths in :meth:`fmrimod.glm.fmri_lm.FmriLm.contrast` remain valid. Only the
typed OmnibusContrast path is held to the declared-provenance contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..design.columns import DesignColumn, DesignColumns
from .errors import DesignProvenanceError


@dataclass(frozen=True)
class OmnibusContrast:
    """F-contrast over a typed term and an optional subset of its levels.

    Parameters
    ----------
    term
        Name of the modeled term, matching ``DesignColumn.term`` produced at
        event-model construction time.
    levels
        Optional tuple of factor-level labels to include, matching
        ``DesignColumn.level``. When empty, every column carrying the named
        term contributes one row to the resulting F matrix.
    basis_ix
        Optional tuple of 1-indexed basis-function positions to include,
        matching ``DesignColumn.basis_ix``. When empty, every basis function
        present for the selected levels contributes a row. Useful for
        multi-basis HRFs (e.g. FIR, spmg3) when only a subset of lags is of
        interest.
    name
        Optional contrast name; defaults to ``"<term>_omnibus"``.
    """

    term: str
    levels: tuple[str, ...] = ()
    basis_ix: tuple[int, ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.levels, tuple):
            object.__setattr__(self, "levels", tuple(self.levels))
        if not isinstance(self.basis_ix, tuple):
            object.__setattr__(self, "basis_ix", tuple(self.basis_ix))

    @property
    def display_name(self) -> str:
        return self.name or f"{self.term}_omnibus"

    def resolve(self, columns: DesignColumns) -> NDArray[np.float64]:
        """Return the F-contrast weight matrix over named design columns.

        Each retained column contributes one row whose only nonzero entry is a
        ``1.0`` at that column's realized index. Multiple basis functions per
        condition therefore produce one row per ``(condition, basis)`` pair.

        Raises
        ------
        TypeError
            If ``columns`` is not a :class:`DesignColumns` value.
        DesignProvenanceError
            If declared term/level provenance is required to interpret the
            hypothesis and is absent or only inferred.
        """
        if not isinstance(columns, DesignColumns):
            raise TypeError(
                "OmnibusContrast.resolve requires DesignColumns; got "
                f"{type(columns).__name__}. Pass fit.design_columns()."
            )

        term_columns = columns.where(term=self.term)
        if len(term_columns) == 0:
            raise DesignProvenanceError(
                f"OmnibusContrast: no design columns carry term={self.term!r}",
                weak_fields=("term",),
                repair_path=(
                    "build the spec with a typed hrf(...) term whose name "
                    "matches, or pass a different term."
                ),
            )

        weak = _weak_provenance(term_columns, require_levels=bool(self.levels))
        if weak:
            field_names = tuple(sorted({field for field, _, _ in weak}))
            details = ", ".join(
                f"{column}.{field}={grade}" for field, column, grade in weak
            )
            raise DesignProvenanceError(
                f"OmnibusContrast(term={self.term!r}) cannot resolve: required "
                f"design-column provenance is weaker than declared. "
                f"weak_fields={field_names}; details={details}",
                weak_fields=field_names,
                repair_path=(
                    "propagate construction-time term/level facts from the "
                    "event-model compiler into DesignColumns; do not lower to "
                    "regex parsing of rendered column names or raw column "
                    "positions."
                ),
            )

        selected = _select_columns(term_columns, self.levels)
        if self.levels:
            present = {col.level for col in selected if col.level is not None}
            missing = set(self.levels) - present
            if missing:
                available = sorted({
                    col.level for col in term_columns if col.level is not None
                })
                raise DesignProvenanceError(
                    f"OmnibusContrast(term={self.term!r}): no design columns "
                    f"match levels={sorted(missing)!r}; "
                    f"available levels={available!r}",
                    weak_fields=("level",),
                    repair_path=(
                        "check requested level names against the event-table "
                        "values, or remove the levels= constraint to include "
                        "every column carrying the term."
                    ),
                )

        if self.basis_ix:
            wanted_basis = set(self.basis_ix)
            available_basis = sorted({
                col.basis_ix for col in selected if col.basis_ix is not None
            })
            selected = [col for col in selected if col.basis_ix in wanted_basis]
            present_basis = {
                col.basis_ix for col in selected if col.basis_ix is not None
            }
            missing_basis = wanted_basis - present_basis
            if missing_basis:
                raise DesignProvenanceError(
                    f"OmnibusContrast(term={self.term!r}): no design columns "
                    f"match basis_ix={sorted(missing_basis)!r}; "
                    f"available basis_ix={available_basis!r}",
                    weak_fields=("basis_ix",),
                    repair_path=(
                        "check requested basis_ix against the HRF's nbasis, "
                        "or remove the basis_ix= constraint to include every "
                        "basis function carrying the term."
                    ),
                )

        n_total = len(columns)
        weights = np.zeros((len(selected), n_total), dtype=np.float64)
        for row_idx, column in enumerate(selected):
            weights[row_idx, column.index] = 1.0
        return weights


def _weak_provenance(
    columns: Iterable[DesignColumn],
    *,
    require_levels: bool,
) -> list[tuple[str, str, str]]:
    """Return (field, column_name, grade) entries for non-declared provenance."""
    weak: list[tuple[str, str, str]] = []
    for column in columns:
        term_grade = column.provenance_for("term")
        if term_grade != "declared":
            weak.append(("term", column.name, term_grade))
        if require_levels:
            level_grade = column.provenance_for("level")
            if level_grade != "declared":
                weak.append(("level", column.name, level_grade))
    return weak


def _select_columns(
    term_columns: DesignColumns,
    levels: tuple[str, ...],
) -> list[DesignColumn]:
    """Filter term columns to those matching the requested level set."""
    if not levels:
        return list(term_columns)
    wanted = set(levels)
    return [column for column in term_columns if column.level in wanted]


__all__ = ["OmnibusContrast"]
