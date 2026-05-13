"""Tests for fmrimod-distinctive Tier D showcases."""

from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.parity


def test_tier_d_showcases_pass_and_render(tmp_path):
    from benchmarks.parity.tier_d_showcase.workflow import render, run_showcases

    rows = run_showcases()
    assert {row.case_id for row in rows} == {
        "tier_d_ar2_robust",
        "tier_d_sketched_glm",
        "tier_d_lss_trialwise",
    }
    assert all(row.status == "pass" for row in rows)

    json_path, md_path = render(rows, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["status"] == "pass"
    assert len(payload["rows"]) == 3
    assert md_path.exists()

