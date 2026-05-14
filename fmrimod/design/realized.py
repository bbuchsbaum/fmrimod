"""Typed pre-built design matrices for the public GLM seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .columns import DesignColumn, DesignColumns, ProvenanceGrade

ColumnKind = Literal["condition", "baseline", "confound"]
DesignSource = Literal["fmrimod", "nilearn", "bids", "fitlins", "user"]

_VALID_COLUMN_KINDS: frozenset[str] = frozenset(("condition", "baseline", "confound"))
_VALID_DESIGN_SOURCES: frozenset[str] = frozenset(
    ("fmrimod", "nilearn", "bids", "fitlins", "user")
)


@dataclass(frozen=True)
class RealizedDesign:
    """A named, already-realized design matrix.

    ``RealizedDesign`` is the typed value for workflows where an external
    system has already constructed the design matrix. It composes the existing
    :class:`DesignColumns` substrate, rather than carrying a second parallel
    metadata shape, so callers can route through :func:`fmrimod.glm.fmri_lm`
    without falling back to raw ``(ndarray, columns)`` pairs or private matrix
    helpers.
    """

    matrix: NDArray[np.float64]
    columns: DesignColumns
    source: DesignSource = "user"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("RealizedDesign.matrix must be a 2-D matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("RealizedDesign.matrix must contain only finite values")

        if not isinstance(self.columns, DesignColumns):
            raise TypeError("RealizedDesign.columns must be a DesignColumns value")

        column_names = self.columns.names
        if len(column_names) != matrix.shape[1]:
            raise ValueError(
                "RealizedDesign.columns length must match matrix columns"
            )
        if len(set(column_names)) != len(column_names):
            raise ValueError("RealizedDesign.columns names must be unique")
        column_indexes = tuple(column.index for column in self.columns)
        expected_indexes = tuple(range(matrix.shape[1]))
        if column_indexes != expected_indexes:
            raise ValueError(
                "RealizedDesign.columns indexes must be contiguous and matrix-ordered"
            )

        source = str(self.source)
        if source not in _VALID_DESIGN_SOURCES:
            raise ValueError(
                "RealizedDesign.source must be one of "
                f"{sorted(_VALID_DESIGN_SOURCES)}"
            )

        matrix = np.array(matrix, dtype=np.float64, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "source", cast(DesignSource, source))

    @classmethod
    def from_array(
        cls,
        matrix: ArrayLike,
        columns: Sequence[str],
        kinds: Sequence[ColumnKind] | None = None,
        source: DesignSource = "user",
    ) -> "RealizedDesign":
        """Create a realized design from an array and column metadata."""
        return cls(
            matrix=np.asarray(matrix, dtype=np.float64),
            columns=_design_columns_from_names(columns, kinds, source),
            source=source,
        )

    @property
    def column_names(self) -> tuple[str, ...]:
        """Column names in realized matrix order."""
        return self.columns.names

    @property
    def column_kinds(self) -> tuple[ColumnKind, ...]:
        """Coarse column roles derived from the typed column records."""
        return tuple(_kind_from_column(column) for column in self.columns)

    @property
    def n_timepoints(self) -> int:
        """Number of rows in the realized design."""
        return int(self.matrix.shape[0])

    @property
    def n_columns(self) -> int:
        """Number of design columns."""
        return int(self.matrix.shape[1])

    def as_dataframe(self) -> pd.DataFrame:
        """Return the full design matrix with stable column names."""
        return pd.DataFrame(np.asarray(self.matrix), columns=list(self.column_names))

    def design_columns(self) -> DesignColumns:
        """Return typed column provenance for contrast resolution."""
        return self.columns


def _design_columns_from_names(
    names: Sequence[str],
    kinds: Sequence[ColumnKind] | None,
    source: DesignSource,
) -> DesignColumns:
    column_names = tuple(str(name) for name in names)
    if kinds is None:
        column_kinds = tuple(_infer_column_kind(name) for name in column_names)
        provenance_grade: ProvenanceGrade = "inferred"
    else:
        column_kinds = tuple(kinds)
        provenance_grade = "declared"
    if len(column_kinds) != len(column_names):
        raise ValueError("RealizedDesign kinds length must match columns")
    invalid_kinds = sorted(set(column_kinds) - _VALID_COLUMN_KINDS)
    if invalid_kinds:
        raise ValueError(
            "RealizedDesign kinds entries must be one of "
            f"{sorted(_VALID_COLUMN_KINDS)}; got {invalid_kinds}"
        )
    return DesignColumns(
        tuple(
            DesignColumn(
                name=name,
                index=index,
                role=_role_from_kind(kind),
                model_source=source,
                term=name if kind == "condition" else None,
                condition=name if kind == "condition" else None,
                provenance=_provenance_for_kind(kind, provenance_grade),
            )
            for index, (name, kind) in enumerate(zip(column_names, column_kinds))
        )
    )


def _role_from_kind(kind: ColumnKind) -> str:
    if kind == "condition":
        return "task"
    return kind


def _provenance_for_kind(
    kind: ColumnKind,
    grade: ProvenanceGrade,
) -> dict[str, ProvenanceGrade]:
    if kind == "condition":
        return {
            "role": grade,
            "term": grade,
            "condition": grade,
            "level": "missing",
            "basis_ix": "missing",
            "basis_name": "missing",
            "basis_total": "missing",
        }
    return {
        "role": grade,
        "term": "missing",
        "condition": "missing",
        "level": "missing",
        "basis_ix": "missing",
        "basis_name": "missing",
        "basis_total": "missing",
    }


def _kind_from_column(column: DesignColumn) -> ColumnKind:
    role = column.role
    if role in {"task", "condition"}:
        return "condition"
    if role == "baseline":
        return "baseline"
    if role == "confound":
        return "confound"
    return _infer_column_kind(column.name)


def _infer_column_kind(name: str) -> ColumnKind:
    lowered = name.lower()
    if lowered in {"intercept", "constant", "const"}:
        return "baseline"
    if lowered.startswith(("drift", "cosine", "poly", "dct")):
        return "baseline"
    confound_markers = (
        "confound",
        "nuisance",
        "motion",
        "trans_",
        "rot_",
        "csf",
        "white_matter",
        "global_signal",
    )
    if any(marker in lowered for marker in confound_markers):
        return "confound"
    return "condition"


__all__ = ["ColumnKind", "DesignSource", "RealizedDesign"]
