"""Tests for the typed F-contrast intent (`OmnibusContrast`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.contrast import DesignProvenanceError, OmnibusContrast
from fmrimod.design import DesignColumn, DesignColumns
from fmrimod.design.event_model import EventModel
from fmrimod.events import EventFactor
from fmrimod.formula.base import Term
from fmrimod.model.fmri_model import FmriModel
from fmrimod.sampling import SamplingFrame

# ---------------------------------------------------------------------------
# Synthetic DesignColumns helpers (for refusal-path tests that should not
# depend on the event-model compiler emitting weak provenance).
# ---------------------------------------------------------------------------


def _declared(
    name: str,
    index: int,
    *,
    term: str,
    level: str | None,
    basis_ix: int | None = 1,
) -> DesignColumn:
    return DesignColumn(
        name=name,
        index=index,
        role="task",
        model_source="event",
        term=term,
        term_tag=term,
        term_index=1,
        condition=f"{term}.{level}" if level is not None else None,
        level=level,
        basis_ix=basis_ix,
        basis_name="SPM_CANONICAL",
        basis_total=1,
        provenance={
            "term": "declared",
            "condition": "declared",
            "level": "declared",
            "basis_ix": "declared",
            "basis_name": "derived",
            "basis_total": "derived",
            "role": "declared",
        },
    )


def _inferred(
    name: str,
    index: int,
    *,
    term: str,
    level: str | None,
) -> DesignColumn:
    return DesignColumn(
        name=name,
        index=index,
        role="task",
        model_source="event",
        term=term,
        term_tag=term,
        term_index=1,
        condition=f"{term}.{level}" if level is not None else None,
        level=level,
        basis_ix=1,
        basis_name=None,
        basis_total=1,
        provenance={
            "term": "inferred",
            "condition": "inferred",
            "level": "inferred",
            "basis_ix": "inferred",
            "basis_name": "missing",
            "basis_total": "missing",
            "role": "inferred",
        },
    )


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


def test_omnibus_resolves_to_one_row_per_named_level() -> None:
    columns = DesignColumns(
        (
            _declared(
                "trial_type.condition_a",
                0,
                term="trial_type",
                level="condition_a",
            ),
            _declared(
                "trial_type.condition_b",
                1,
                term="trial_type",
                level="condition_b",
            ),
            _declared(
                "trial_type.condition_c",
                2,
                term="trial_type",
                level="condition_c",
            ),
        )
    )

    omnibus = OmnibusContrast("trial_type", levels=("condition_a", "condition_b"))
    weights = omnibus.resolve(columns)

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert weights.shape == (2, 3)
    np.testing.assert_array_equal(weights, expected)


def test_omnibus_with_no_levels_includes_every_term_column() -> None:
    columns = DesignColumns(
        (
            _declared(
                "trial_type.condition_a",
                0,
                term="trial_type",
                level="condition_a",
            ),
            _declared(
                "trial_type.condition_b",
                1,
                term="trial_type",
                level="condition_b",
            ),
        )
    )

    omnibus = OmnibusContrast("trial_type")
    weights = omnibus.resolve(columns)

    assert weights.shape == (2, 2)
    np.testing.assert_array_equal(weights, np.eye(2, dtype=np.float64))


def test_omnibus_emits_one_row_per_basis_when_term_has_multiple_bases() -> None:
    columns = DesignColumns(
        (
            _declared(
                "trial_type.condition_a_b1",
                0,
                term="trial_type",
                level="condition_a",
                basis_ix=1,
            ),
            _declared(
                "trial_type.condition_a_b2",
                1,
                term="trial_type",
                level="condition_a",
                basis_ix=2,
            ),
            _declared(
                "trial_type.condition_b_b1",
                2,
                term="trial_type",
                level="condition_b",
                basis_ix=1,
            ),
            _declared(
                "trial_type.condition_b_b2",
                3,
                term="trial_type",
                level="condition_b",
                basis_ix=2,
            ),
        )
    )

    omnibus = OmnibusContrast("trial_type", levels=("condition_a", "condition_b"))
    weights = omnibus.resolve(columns)

    assert weights.shape == (4, 4)


def test_display_name_defaults_to_term_plus_suffix() -> None:
    assert OmnibusContrast("trial_type").display_name == "trial_type_omnibus"
    assert (
        OmnibusContrast("trial_type", name="conditions").display_name == "conditions"
    )


# ---------------------------------------------------------------------------
# Refusal paths — declared-provenance contract
# ---------------------------------------------------------------------------


def test_unknown_term_raises_design_provenance_error_naming_term() -> None:
    columns = DesignColumns(
        (
            _declared(
                "trial_type.condition_a",
                0,
                term="trial_type",
                level="condition_a",
            ),
        )
    )

    with pytest.raises(DesignProvenanceError) as info:
        OmnibusContrast("missing_term").resolve(columns)

    assert "term" in info.value.weak_fields
    assert info.value.repair_path is not None


def test_inferred_term_provenance_is_refused_with_named_weak_fields() -> None:
    columns = DesignColumns(
        (
            _inferred(
                "trial_type.condition_a",
                0,
                term="trial_type",
                level="condition_a",
            ),
            _inferred(
                "trial_type.condition_b",
                1,
                term="trial_type",
                level="condition_b",
            ),
        )
    )

    omnibus = OmnibusContrast("trial_type", levels=("condition_a", "condition_b"))
    with pytest.raises(DesignProvenanceError) as info:
        omnibus.resolve(columns)

    # Both term and level provenance are inferred — both must surface.
    assert "term" in info.value.weak_fields
    assert "level" in info.value.weak_fields
    # Repair path must point at the compiler, not a column-level workaround.
    assert "compiler" in (info.value.repair_path or "")


def test_unknown_level_raises_design_provenance_error_naming_level() -> None:
    columns = DesignColumns(
        (
            _declared(
                "trial_type.condition_a",
                0,
                term="trial_type",
                level="condition_a",
            ),
            _declared(
                "trial_type.condition_b",
                1,
                term="trial_type",
                level="condition_b",
            ),
        )
    )

    with pytest.raises(DesignProvenanceError) as info:
        OmnibusContrast("trial_type", levels=("condition_c",)).resolve(columns)

    assert "level" in info.value.weak_fields


def test_resolve_rejects_non_design_columns_input() -> None:
    with pytest.raises(TypeError):
        OmnibusContrast("trial_type").resolve(  # type: ignore[arg-type]
            [{"name": "x", "index": 0}]
        )


# ---------------------------------------------------------------------------
# End-to-end through FmriLm.contrast()
# ---------------------------------------------------------------------------


class _Baseline:
    def __init__(self, n_scans: int):
        self.design_matrix = pd.DataFrame({"intercept": np.ones(n_scans)})
        self.column_names = ["intercept"]


class _Dataset:
    def __init__(self, n_scans: int):
        self.n_timepoints = n_scans
        self.n_runs = 1

    def get_sampling_frame(self):
        return SamplingFrame(tr=2.0, n_scans=self.n_timepoints)


def _two_condition_event_model(n_scans: int = 30) -> EventModel:
    events = {
        "trial_type": EventFactor(
            name="trial_type",
            onsets=np.array([2.0, 8.0, 14.0, 20.0]),
            values=np.array(
                ["condition_a", "condition_b", "condition_a", "condition_b"]
            ),
            durations=1.0,
        )
    }
    return EventModel(
        terms=[Term("trial_type", hrf="spm")],
        events=events,
        sampling_info=SamplingFrame(tr=2.0, n_scans=n_scans),
    )


def _build_fit_with_synthetic_data(
    n_voxels: int,
    *,
    seed: int,
    n_scans: int = 30,
):
    """Build an FmriLm with a real fitted GLM on synthetic data."""
    from fmrimod.glm.fmri_lm import FmriLm
    from fmrimod.model.config import FmriLmConfig

    event_model = _two_condition_event_model(n_scans=n_scans)
    baseline = _Baseline(n_scans=n_scans)
    dataset = _Dataset(n_scans=n_scans)
    model = FmriModel(event_model, baseline, dataset)

    rng = np.random.default_rng(seed)
    design = np.asarray(model.design_matrix(run=0), dtype=np.float64)
    betas = rng.normal(size=(design.shape[1], n_voxels))
    Y = design @ betas + rng.normal(scale=0.1, size=(n_scans, n_voxels))
    XtX = design.T @ design
    XtXinv = np.linalg.inv(XtX)
    beta_hat = XtXinv @ design.T @ Y
    resid = Y - design @ beta_hat
    residual_df = float(n_scans - design.shape[1])
    sigma = np.sqrt((resid * resid).sum(axis=0) / max(residual_df, 1.0))

    fit = FmriLm(
        betas=beta_hat,
        sigma=sigma,
        residual_df=residual_df,
        XtXinv=XtXinv,
        model=model,
        config=FmriLmConfig(),
    )
    return fit, design, beta_hat, sigma, XtXinv, residual_df


def test_fit_contrast_accepts_omnibus_and_matches_manual_weights() -> None:
    from fmrimod.glm.contrasts import contrast_f_vectorized

    fit, _design, beta_hat, sigma, XtXinv, residual_df = (
        _build_fit_with_synthetic_data(n_voxels=4, seed=20260513)
    )

    columns = fit.design_columns()
    a_index = columns.where(term="trial_type", level="condition_a").one().index
    b_index = columns.where(term="trial_type", level="condition_b").one().index
    n_total = len(columns)

    manual = np.zeros((2, n_total), dtype=np.float64)
    manual[0, a_index] = 1.0
    manual[1, b_index] = 1.0

    omnibus = OmnibusContrast(
        "trial_type",
        levels=("condition_a", "condition_b"),
        name="conditions_omnibus",
    )
    typed_result = fit.contrast(omnibus)
    manual_result = contrast_f_vectorized(
        manual, beta_hat, XtXinv, sigma, residual_df, name="conditions_omnibus"
    )

    assert typed_result.name == "conditions_omnibus"
    assert typed_result.intent is not None
    assert typed_result.intent.kind == "omnibus"
    assert typed_result.intent.term == "trial_type"
    assert typed_result.intent.levels == ("condition_a", "condition_b")
    assert typed_result.intent.rows == 2
    assert typed_result.touched_columns == (
        columns[a_index].name,
        columns[b_index].name,
    )
    np.testing.assert_allclose(typed_result.stat, manual_result.stat)
    np.testing.assert_allclose(typed_result.estimate, manual_result.estimate)


def test_fit_contrast_explain_returns_structured_omnibus_fields() -> None:
    fit, _design, *_ = _build_fit_with_synthetic_data(
        n_voxels=3,
        seed=20260514,
    )

    result = fit.contrast(
        OmnibusContrast(
            "trial_type",
            levels=("condition_a", "condition_b"),
            name="conditions_omnibus",
        )
    )

    explanation = result.explain()
    summary = result.summary()

    assert explanation.intent["kind"] == "omnibus"
    assert explanation.intent["term"] == "trial_type"
    assert explanation.intent["levels"] == ["condition_a", "condition_b"]
    assert explanation.statistic["family"] == "F"
    assert explanation.statistic["df_num"] == 2.0
    assert explanation.statistic["df_den"] == result.df[1]
    assert explanation.caveats == ()
    assert any("condition_a" in name for name in explanation.touched_columns)
    assert any("condition_b" in name for name in explanation.touched_columns)
    assert [column["level"] for column in explanation.design_columns] == [
        "condition_a",
        "condition_b",
    ]
    assert all(
        column["provenance"]["level"] == "declared"
        for column in explanation.design_columns
    )
    assert summary["statistic"]["family"] == "F"
    assert summary["design_columns"][0]["provenance"]["term"] == "declared"
    assert summary["caveats"] == []


def test_array_contrast_path_still_works_alongside_omnibus() -> None:
    """Compatibility: existing NDArray contrast path is not regressed."""
    fit, design, *_ = _build_fit_with_synthetic_data(n_voxels=2, seed=7)
    t_weights = np.zeros(design.shape[1], dtype=np.float64)
    t_weights[0] = 1.0
    t_weights[1] = -1.0
    result = fit.contrast(t_weights, name="a_minus_b")
    assert result.name == "a_minus_b"
    assert result.intent is not None
    assert result.intent.kind == "array"
    assert result.summary()["statistic"]["family"] == "t"
    assert result.summary()["statistic"]["df_resid"] == result.df
    assert len(result.summary()["touched_columns"]) == 2
    assert result.stat.shape == (2,)


# ---------------------------------------------------------------------------
# Public import boundary (acceptance from bd-01KRGN5E73ZNR0DK5R843NV372)
# ---------------------------------------------------------------------------


def test_public_imports_for_typed_contrast_and_design_provenance_error() -> None:
    from fmrimod.contrast import (  # noqa: F401  -- verify public surface
        DesignProvenanceError as PublicError,
    )
    from fmrimod.contrast import (
        OmnibusContrast as PublicOmnibus,
    )

    assert PublicError is DesignProvenanceError
    assert PublicOmnibus is OmnibusContrast


def test_omnibus_module_does_not_redefine_predicate() -> None:
    """Acceptance from bead notes: do not introduce a third Predicate alias."""
    from fmrimod.contrast import omnibus as omnibus_module

    assert not hasattr(omnibus_module, "Predicate"), (
        "OmnibusContrast must not introduce a third Predicate type alias; "
        "the dedup between fmrimod.spec.terms and fmrimod.contrast.contrast_spec "
        "is a separate follow-up under the typed-contrast sprint."
    )
