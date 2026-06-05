"""Tier A Nilearn first-level parity workflow tests."""

from __future__ import annotations

import pytest

from cross_testing.harness import render, run

pytestmark = pytest.mark.parity

pytest.importorskip("nilearn")
pytest.importorskip("nibabel")


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.parity.tier_a_f_confound_drift.workflow",
        "benchmarks.parity.tier_a_f_confound_drift.public_workflow",
        "benchmarks.parity.tier_a_fiac.workflow",
        "benchmarks.parity.tier_a_localizer_fixed_effects.workflow",
        "benchmarks.parity.tier_a_parametric_modulation.workflow",
        "benchmarks.parity.tier_a_fir_unconstrained_hrf.workflow",
        "benchmarks.parity.tier_a_factorial_2x2.workflow",
        "benchmarks.parity.tier_a_multicollinear_baseline.workflow",
        "benchmarks.parity.tier_a_factorial_3x3_parametric.workflow",
        "benchmarks.parity.tier_a_multirun_concat.workflow",
        "benchmarks.parity.tier_a_hrf_basis_set.workflow",
        "benchmarks.parity.tier_a_block_epoch_durations.workflow",
        "benchmarks.parity.tier_a_censored_concat.workflow",
        "benchmarks.parity.tier_a_fir_basis.workflow",
        "benchmarks.parity.tier_a_parametric_modulators.workflow",
        "benchmarks.parity.tier_a_mixed_tr_multirun.workflow",
        "benchmarks.parity.tier_a_realistic_confounds.workflow",
        "benchmarks.parity.tier_a_single_trial_lss.workflow",
        "benchmarks.parity.tier_a_ar1_prewhitening.workflow",
    ],
)
def test_tier_a_nilearn_case_passes_and_renders(module_name, tmp_path):
    try:
        module = __import__(module_name, fromlist=["make_case"])
    except Exception as exc:  # pragma: no cover - import environment guard
        pytest.skip(f"Tier A parity dependencies unavailable for {module_name}: {exc}")

    try:
        result = run(module.make_case(max_voxels=2048))
    except Exception as exc:
        pytest.skip(f"Tier A parity dataset unavailable for {module_name}: {exc}")

    assert result.status in {"pass", "pass_with_caveats"}
    assert all(delta.passes for delta in result.deltas.values())

    json_path, md_path = render(result, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
