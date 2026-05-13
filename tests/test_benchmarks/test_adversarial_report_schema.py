"""Schema tests for adversarial parity benchmark reports."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from benchmarks.parity.adversarial_schema import validate_adversarial_report
from benchmarks.parity.tier_e_adversarial_gauntlet import workflow

pytest.importorskip("nilearn")


ROOT = Path(__file__).resolve().parents[2]


def test_tier_e_gauntlet_conforms_to_adversarial_schema() -> None:
    report = workflow.run_gauntlet(max_voxels=6)

    validate_adversarial_report(report)


def test_adversarial_schema_rejects_missing_engine_fields() -> None:
    report = workflow.run_gauntlet(max_voxels=4)
    broken = copy.deepcopy(report)
    del broken["cases"][0]["fmrimod"]["finite_stat_fraction"]

    with pytest.raises(ValueError, match="finite_stat_fraction"):
        validate_adversarial_report(broken)


def test_adversarial_schema_docs_name_status_contract() -> None:
    text = (
        ROOT / "docs" / "contracts" / "adversarial_benchmark_report_schema_v1.md"
    ).read_text()

    assert "adversarial-gauntlet/v1" in text
    assert "`boundary_observed`" in text
    assert "`undefined_t_policy`" in text
