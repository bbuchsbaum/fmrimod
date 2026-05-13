"""Contract tests for parity performance trend rows."""

from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.benchmark


def test_parity_performance_trend_rows_cover_required_stages(tmp_path):
    from benchmarks.performance.parity_trends import render, run_trends

    rows = run_trends()
    assert {row.stage for row in rows} == {
        "design_build",
        "glm_fit",
        "contrast",
        "ar_whitening",
        "run_combination",
        "lss",
    }
    assert all(row.status == "ok" for row in rows)
    assert all(row.seconds >= 0.0 for row in rows)

    out = render(rows, tmp_path)
    payload = json.loads(out.read_text())
    assert payload["gate_policy"] == "correctness-gated; performance tracked as trend"
    assert len(payload["rows"]) == 6
