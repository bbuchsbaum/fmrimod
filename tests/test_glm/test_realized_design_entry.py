"""Public-seam tests for typed pre-built design matrices."""

from __future__ import annotations

import numpy as np
import pytest

import fmrimod as fm
from fmrimod.design import DesignColumns, RealizedDesign


def test_fmri_lm_accepts_realized_design_and_records_source() -> None:
    rng = np.random.default_rng(41)
    n_timepoints = 60
    n_voxels = 4
    task = np.linspace(-1.0, 1.0, n_timepoints)
    matrix = np.column_stack([np.ones(n_timepoints), task])
    beta = np.vstack(
        [
            np.full(n_voxels, 1.25),
            np.linspace(0.5, 1.1, n_voxels),
        ]
    )
    data = matrix @ beta + rng.normal(scale=0.001, size=(n_timepoints, n_voxels))
    dataset = fm.matrix_dataset(data, tr=2.0, mask=np.ones(n_voxels, dtype=bool))
    design = RealizedDesign.from_array(
        matrix,
        columns=("intercept", "task"),
        kinds=("baseline", "condition"),
        source="nilearn",
    )

    assert isinstance(design.columns, DesignColumns)
    assert design.column_names == ("intercept", "task")
    assert design.column_kinds == ("baseline", "condition")

    fit = fm.fmri_lm(design, dataset)
    contrast = fit.contrast(fm.column_contrast("^task$", name="task_effect"))

    assert fit.design_columns().names == ("intercept", "task")
    assert fit.provenance is not None
    assert fit.provenance.design_source == "nilearn"
    assert fit.provenance.design_source_status == "carried"
    assert contrast.touched_columns == ("task",)
    assert contrast.intent is not None
    assert contrast.intent.kind == "contrast_spec"
    np.testing.assert_allclose(fit.coef()[1], beta[1], atol=0.01)


def test_realized_design_splits_rows_by_dataset_runs() -> None:
    rng = np.random.default_rng(42)
    run_lengths = [8, 7]
    n_timepoints = sum(run_lengths)
    matrix = np.column_stack(
        [
            np.ones(n_timepoints),
            rng.normal(size=n_timepoints),
        ]
    )
    data = matrix @ np.array([[1.0, -1.0], [0.25, 0.5]])
    dataset = fm.matrix_dataset(
        data,
        tr=1.5,
        run_length=run_lengths,
        mask=np.ones(2, dtype=bool),
    )
    design = RealizedDesign.from_array(
        matrix,
        columns=("intercept", "stimulus"),
        kinds=("baseline", "condition"),
        source="user",
    )

    fit = fm.fmri_lm(design, dataset)

    assert fit.model.n_runs == 2
    assert fit.model.design_matrix_array(0).shape == (8, 2)
    assert fit.model.design_matrix_array(1).shape == (7, 2)
    assert len(fit.run_results or []) == 2


def test_realized_design_rejects_dataset_length_mismatch() -> None:
    dataset = fm.matrix_dataset(
        np.ones((6, 2), dtype=np.float64),
        tr=2.0,
        mask=np.ones(2, dtype=bool),
    )
    design = RealizedDesign.from_array(
        np.ones((5, 2), dtype=np.float64),
        columns=("intercept", "task"),
    )

    try:
        fm.fmri_lm(design, dataset)
    except ValueError as exc:
        assert "row count must match dataset timepoints" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("expected row-count mismatch to fail")


def test_realized_design_validates_column_metadata() -> None:
    matrix = np.ones((5, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="columns length"):
        RealizedDesign.from_array(matrix, columns=("intercept",))
    with pytest.raises(ValueError, match="columns names must be unique"):
        RealizedDesign.from_array(matrix, columns=("task", "task"))
    with pytest.raises(ValueError, match="kinds entries"):
        RealizedDesign.from_array(
            matrix,
            columns=("intercept", "task"),
            kinds=("baseline", "bad"),  # type: ignore[list-item]
        )
