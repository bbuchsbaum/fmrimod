"""Canary test for the reusable external-reference parity harness."""

from __future__ import annotations

import json

import pytest

from cross_testing.fitlins_parity import fit_fitlins_reference_ols, fit_fmrimod_ols
from cross_testing.harness import ParityCase, ParityTolerance, PipelineOutput, render, run
from cross_testing.harness.fixtures import SyntheticGlmInputs, synthetic_ols_inputs


pytestmark = pytest.mark.parity

nilearn = pytest.importorskip("nilearn")


def _fmrimod_pipeline(inputs: SyntheticGlmInputs) -> PipelineOutput:
    return PipelineOutput(
        arrays=fit_fmrimod_ols(inputs.X, inputs.Y, inputs.contrast),
    )


def _nilearn_pipeline(inputs: SyntheticGlmInputs) -> PipelineOutput:
    return PipelineOutput(
        arrays=fit_fitlins_reference_ols(inputs.X, inputs.Y, inputs.contrast),
    )


def test_synthetic_ols_canary_passes_and_renders(tmp_path):
    case = ParityCase(
        name="synthetic_ols_nilearn_canary",
        fmrimod_pipeline=_fmrimod_pipeline,
        reference_pipeline=_nilearn_pipeline,
        inputs=synthetic_ols_inputs(),
        tolerances={
            "betas": ParityTolerance(rtol=1e-6, atol=1e-8),
            "sigma2": ParityTolerance(rtol=1e-6, atol=1e-8),
            "t": ParityTolerance(rtol=1e-6, atol=1e-8),
            "p": ParityTolerance(rtol=1e-6, atol=1e-8),
        },
    )

    result = run(case)
    assert result.status == "pass"
    assert all(delta.passes for delta in result.deltas.values())

    json_path, md_path = render(result, tmp_path)
    payload = json.loads(json_path.read_text())

    assert payload["name"] == "synthetic_ols_nilearn_canary"
    assert payload["status"] == "pass"
    assert md_path.read_text().startswith("# Parity Report")
