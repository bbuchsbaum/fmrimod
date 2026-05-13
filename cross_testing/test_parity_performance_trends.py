"""Contract tests for parity performance trend rows."""

from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.benchmark


def test_parity_performance_trend_rows_cover_required_stages(tmp_path):
    from benchmarks.performance.parity_trends import render, run_trends

    rows = run_trends(
        repetitions=5,
        generated_at="2026-05-12T00:00:00+00:00",
        git_sha="test-sha",
        hardware_tag="test-hardware",
    )
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
    assert all(row.seconds_iqr >= 0.0 for row in rows)
    assert all(row.repetitions == 5 for row in rows)
    assert {row.git_sha for row in rows} == {"test-sha"}

    out = render(rows, tmp_path, append=False)
    payload = json.loads(out.read_text())
    assert payload["gate_policy"] == "correctness-gated; performance tracked as trend"
    assert payload["history_file"] == "parity_performance_trends.jsonl"
    assert len(payload["rows"]) == 6

    history_path = tmp_path / "parity_performance_trends.jsonl"
    history_rows = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(history_rows) == 6
    assert {row["hardware_tag"] for row in history_rows} == {"test-hardware"}


def test_parity_performance_regression_checker_flags_latest_slowdown():
    from benchmarks.performance.check_regression import find_regressions

    records = [
        {
            "generated_at": f"2026-05-12T00:00:0{idx}+00:00",
            "hardware_tag": "hw",
            "stage": "glm_fit",
            "seconds": 1.0,
        }
        for idx in range(3)
    ]
    records.append(
        {
            "generated_at": "2026-05-12T00:00:09+00:00",
            "hardware_tag": "hw",
            "stage": "glm_fit",
            "seconds": 1.4,
        }
    )

    regressions = find_regressions(records, threshold=0.25, window=10)
    assert regressions == [
        {
            "hardware_tag": "hw",
            "stage": "glm_fit",
            "latest_seconds": 1.4,
            "baseline_seconds": 1.0,
            "threshold": 0.25,
        }
    ]
