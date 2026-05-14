"""Contract tests for the Tier D showcase workflow."""

from __future__ import annotations

import inspect
import json

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


def test_tier_d_proof_scorecard_names_public_typed_seam() -> None:
    rows = workflow.run_showcases()
    scorecard = workflow.build_proof_scorecard(rows)

    assert scorecard.public_seam is True
    assert scorecard.fmrimod_path == (
        "fmri_dataset -> fmri_lm -> OmnibusContrast -> "
        "ContrastResult.explain -> GroupDataset -> ols_voxelwise"
    )
    assert "fmrimod.glm.ContrastResult" in scorecard.typed_objects
    assert "fmrimod.group.GroupDataset" in scorecard.typed_objects
    assert "tier_d_lss_public_seam" in scorecard.public_rows
    assert "tier_d_lss_public_seam_independent_generative" in scorecard.public_rows
    assert not hasattr(scorecard, "low_level_canaries")
    assert scorecard.semantic_survival["source"] == "tier_group_semantic_survival"
    assert scorecard.semantic_survival["status"] == "pass"
    assert scorecard.semantic_survival["typed_intent_kind"] == "omnibus"
    assert scorecard.semantic_survival["typed_intent_term"] == "trial_type"
    assert scorecard.semantic_survival["statistic_family"] == "F"
    assert scorecard.semantic_survival["timings"]["status"] == "recorded"
    assert set(scorecard.win_axes) == {"design", "elegance", "power", "trust"}


def test_tier_d_render_writes_json_ready_proof_scorecard(tmp_path) -> None:
    rows = workflow.run_showcases()
    json_path, md_path = workflow.render(rows, tmp_path)

    payload = json.loads(json_path.read_text())
    assert payload["status"] == "pass"
    assert payload["caveats"] == []
    assert payload["proof_scorecard"]["public_seam"] is True
    assert payload["proof_scorecard"]["semantic_survival"]["typed_intent_term"] == (
        "trial_type"
    )
    assert "low_level_canaries" not in payload["proof_scorecard"]
    assert {row["case_id"] for row in payload["rows"]} == set(
        payload["proof_scorecard"]["public_rows"]
    )
    assert all("public-seam" in row["capability"] for row in payload["rows"])
    assert "Proof Scorecard" in md_path.read_text()

    canary_payload = json.loads((tmp_path / "showcase_canaries.json").read_text())
    assert canary_payload["status"] == "pass"
    assert {row["case_id"] for row in canary_payload["rows"]} == {
        "tier_d_ar2_robust",
        "tier_d_sketched_glm",
        "tier_d_lss_trialwise_oracle",
    }
