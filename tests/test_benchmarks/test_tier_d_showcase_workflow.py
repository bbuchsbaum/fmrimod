"""Contract tests for the Tier D showcase workflow."""

from __future__ import annotations

import inspect

from benchmarks.parity.tier_d_showcase import workflow


def test_tier_d_includes_public_seam_single_trial_row() -> None:
    rows = workflow.run_showcases()
    by_id = {row.case_id: row for row in rows}

    assert "tier_d_lss_public_seam" in by_id
    row = by_id["tier_d_lss_public_seam"]
    assert row.status == "pass"
    assert row.metric == "max_abs_recovery_error"
    assert row.value < row.threshold
    assert row.threshold == 0.5
    assert row.details["wrapper_trial_labels_present"] is True
    assert row.details["wrapper_n_trial_labels"] == row.details["n_trials"]
    assert row.details["noise_scale"] == 0.03
    assert row.details["beta_scale"] == 0.7


def test_tier_d_includes_independent_generative_lss_row() -> None:
    rows = workflow.run_showcases()
    by_id = {row.case_id: row for row in rows}

    source = inspect.getsource(workflow.run_lss_public_seam_independent_generative)
    assert "fmrimod.design.event_model" not in source
    assert "_build_event_model" not in source
    assert "event_model(" not in source
    assert "fmri_dataset" in source
    assert "estimate_single_trial_from_dataset" in source

    row = by_id["tier_d_lss_public_seam_independent_generative"]
    assert row.status == "pass"
    assert row.metric == "max_abs_recovery_error"
    assert row.value < row.threshold
    assert row.threshold == 0.05
    assert row.details["generator_uses_event_model"] is False
    assert row.details["wrapper_trial_labels_present"] is True
    assert row.details["wrapper_n_trial_labels"] == row.details["n_trials"]
    assert row.details["noise_scale"] == 0.001
    assert row.details["beta_scale"] == 0.7
    assert row.details["hrf_basis"] == "gamma"
    assert row.details["generative_design"] == "direct_gamma_hrf"
