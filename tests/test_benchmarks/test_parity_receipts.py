"""Executable checks for regenerable parity proof receipts."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from benchmarks.parity.tier_a_f_confound_drift import public_workflow
from benchmarks.parity.verify_receipt import ROOT, verify_receipt

pytest.importorskip("nilearn")


def test_tier_a_f_confound_drift_receipt_renders_to_manifest_path() -> None:
    result = verify_receipt("tier_a_f_confound_drift")

    assert result["report_path"] == (
        "benchmarks/parity/tier_a_f_confound_drift/reports_public/parity_report.json"
    )
    assert result["status"] == "pass"
    assert result["caveats"] == []

    report_path = ROOT / result["report_path"]
    report = json.loads(Path(report_path).read_text())
    assert report["name"] == "tier_a_public_f_confound_drift"


def test_tier_d_showcase_receipt_renders_proof_scorecard() -> None:
    result = verify_receipt("tier_d_showcase")

    assert result["report_path"] == (
        "benchmarks/parity/tier_d_showcase/reports/showcase_report.json"
    )
    assert result["status"] == "pass"
    assert result["caveats"] == []

    report_path = ROOT / result["report_path"]
    report = json.loads(Path(report_path).read_text())
    scorecard = report["proof_scorecard"]
    assert scorecard["public_seam"] is True
    assert "low_level_canaries" not in scorecard
    assert scorecard["semantic_survival"]["typed_intent_term"] == "trial_type"
    assert scorecard["semantic_survival"]["statistic_family"] == "F"
    assert "fmrimod.group.GroupDataset" in scorecard["typed_objects"]
    assert {row["case_id"] for row in report["rows"]} == set(scorecard["public_rows"])
    assert all("public-seam" in row["capability"] for row in report["rows"])


def test_tier_a_f_confound_drift_uses_typed_omnibus_intent() -> None:
    manifest = json.loads((ROOT / "benchmarks/parity/proof_artifacts.json").read_text())
    artifact = next(
        item for item in manifest["artifacts"]
        if item["benchmark_id"] == "tier_a_f_confound_drift"
    )

    inputs = public_workflow.load_inputs(max_voxels=2)
    fmrimod_source = inspect.getsource(public_workflow.fmrimod_pipeline)

    assert inputs.omnibus.term == "trial_type"
    assert inputs.omnibus.levels == ("condition_a", "condition_b")
    assert inputs.reference_f_contrast.shape == (2, len(inputs.design_columns))
    assert not hasattr(inputs, "f_contrast")
    assert "inputs.omnibus" in fmrimod_source
    assert "reference_f_contrast" not in fmrimod_source
    assert "np.vstack" not in inspect.getsource(public_workflow)
    assert "fmrimod.contrast.OmnibusContrast" in artifact["typed_objects"]
