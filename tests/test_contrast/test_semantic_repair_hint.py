"""Tests for the basis-ambiguous repair hint on ``SemanticContrast.resolve``.

Owns the red check from bd-01KRMQ47HB20TF523VYV1HVE19: when a semantic
contrast hits a multi-basis HRF without ``basis_ix=`` set, the user-visible
error message must name ``basis_ix=N`` together with the role each ``N``
plays (``canonical`` / ``derivative`` / ``dispersion`` for SPM informed
basis sets, ``lag N`` for FIR, generic phrasing otherwise).
"""

from __future__ import annotations

import pytest

from fmrimod.contrast import DesignProvenanceError, SemanticContrast, condition
from fmrimod.design import DesignColumn, DesignColumns


def _spmg_column(
    name: str,
    index: int,
    *,
    basis_ix: int,
    basis_total: int,
    basis_name: str,
) -> DesignColumn:
    return DesignColumn(
        name=name,
        index=index,
        role="task",
        model_source="event",
        term="trial_type",
        term_tag="trial_type",
        term_index=1,
        condition="trial_type.listening",
        level="listening",
        basis_ix=basis_ix,
        basis_name=basis_name,
        basis_total=basis_total,
        provenance={
            "term": "declared",
            "condition": "declared",
            "level": "declared",
            "basis_ix": "declared",
            "basis_name": "declared",
            "basis_total": "declared",
            "role": "declared",
        },
    )


def test_repair_hint_names_spmg2_canonical_and_derivative() -> None:
    """SPMG2 ambiguity surfaces basis_ix=1 / basis_ix=2 with role labels."""
    columns = DesignColumns(
        (
            _spmg_column(
                "trial_type.listening_b1",
                0,
                basis_ix=1,
                basis_total=2,
                basis_name="SPMG2",
            ),
            _spmg_column(
                "trial_type.listening_b2",
                1,
                basis_ix=2,
                basis_total=2,
                basis_name="SPMG2",
            ),
        )
    )

    contrast = SemanticContrast(positive=condition("listening", term="trial_type"))

    with pytest.raises(DesignProvenanceError) as excinfo:
        contrast.resolve(columns)

    message = str(excinfo.value)
    assert "basis_ix=1 for canonical" in message
    assert "basis_ix=2 for derivative" in message
    assert excinfo.value.weak_fields == ("basis_ix",)


def test_repair_hint_names_spmg3_dispersion() -> None:
    """SPMG3 ambiguity surfaces dispersion alongside canonical / derivative."""
    columns = DesignColumns(
        tuple(
            _spmg_column(
                f"trial_type.listening_b{ix}",
                ix - 1,
                basis_ix=ix,
                basis_total=3,
                basis_name="SPMG3",
            )
            for ix in (1, 2, 3)
        )
    )

    contrast = SemanticContrast(positive=condition("listening", term="trial_type"))

    with pytest.raises(DesignProvenanceError) as excinfo:
        contrast.resolve(columns)

    message = str(excinfo.value)
    assert "basis_ix=1 for canonical" in message
    assert "basis_ix=2 for derivative" in message
    assert "basis_ix=3 for dispersion" in message


def test_repair_hint_labels_fir_lags() -> None:
    """FIR ambiguity surfaces 'lag N' labels rather than informed-basis roles."""
    columns = DesignColumns(
        tuple(
            _spmg_column(
                f"trial_type.listening_b{ix}",
                ix - 1,
                basis_ix=ix,
                basis_total=4,
                basis_name="FIR",
            )
            for ix in (1, 2, 3, 4)
        )
    )

    contrast = SemanticContrast(positive=condition("listening", term="trial_type"))

    with pytest.raises(DesignProvenanceError) as excinfo:
        contrast.resolve(columns)

    message = str(excinfo.value)
    assert "basis_ix=1 for lag 1" in message
    assert "basis_ix=4 for lag 4" in message
    assert "canonical" not in message


def test_repair_hint_falls_back_for_unknown_basis_name() -> None:
    """Unknown basis families fall back to 'basis column N' phrasing."""
    columns = DesignColumns(
        (
            _spmg_column(
                "trial_type.listening_b1",
                0,
                basis_ix=1,
                basis_total=2,
                basis_name="BSPLINE",
            ),
            _spmg_column(
                "trial_type.listening_b2",
                1,
                basis_ix=2,
                basis_total=2,
                basis_name="BSPLINE",
            ),
        )
    )

    contrast = SemanticContrast(positive=condition("listening", term="trial_type"))

    with pytest.raises(DesignProvenanceError) as excinfo:
        contrast.resolve(columns)

    message = str(excinfo.value)
    assert "basis_ix=1 for basis column 1" in message
    assert "basis_ix=2 for basis column 2" in message
    assert "canonical" not in message
    assert "lag" not in message
