"""Contract tests for the Tier C second-level parity workflow."""

from __future__ import annotations

import inspect

import pytest

from benchmarks.parity.tier_c_second_level import workflow

pytest.importorskip("nilearn")


def test_tier_c_age_model_uses_native_ols_voxelwise() -> None:
    source = inspect.getsource(workflow.fmrimod_pipeline)
    age_branch = source.split("gd_reg = ", maxsplit=1)[1]

    assert "ols_voxelwise" in age_branch
    assert "GroupFitRequest" not in age_branch
    assert 'model="meta"' not in age_branch
    assert 'weights="equal"' not in age_branch


def test_tier_c_age_outputs_are_mapped_from_ols_assays() -> None:
    source = inspect.getsource(workflow.fmrimod_pipeline)

    assert 'assay("coef:age")' in source
    assert 'assay("t_coef:age")' in source
    assert 'assay("p_coef:age")' not in source
    assert "age_p_signed_one_sided" in source


def test_tier_c_second_level_has_no_caveats() -> None:
    case = workflow.make_case()

    assert case.declared_caveats == ()
    assert case.tolerances["age_t"].check_allclose
    assert case.tolerances["age_p_signed_one_sided"].check_allclose


def test_tier_c_age_p_parity_is_named_signed_quantity() -> None:
    outputs = workflow.fmrimod_pipeline(workflow.make_inputs()).arrays

    assert "age_p_signed_one_sided" in outputs
    assert outputs["age_p_signed_one_sided"].ndim == 1
