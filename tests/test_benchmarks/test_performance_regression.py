"""Tests for performance-history regression checks."""

from __future__ import annotations

import pytest

from benchmarks.performance.check_regression import (
    find_artifact_history_gaps,
    find_history_gaps,
    find_regressions,
)


def _row(
    *,
    seconds: float,
    generated_at: str,
    hardware_tag: str = "Darwin-arm64-arm",
    stage: str = "tier_d_showcase",
) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "git_sha": "fixture",
        "hardware_tag": hardware_tag,
        "stage": stage,
        "seconds": seconds,
        "seconds_iqr": 0.001,
        "repetitions": 5,
        "status": "ok",
    }


def test_history_gaps_fail_empty_and_single_row_histories() -> None:
    assert find_history_gaps([]) == [
        {
            "hardware_tag": "",
            "stage": "*",
            "count": 0,
            "min_records": 2,
            "reason": "no_records",
        }
    ]

    gaps = find_history_gaps(
        [_row(seconds=1.0, generated_at="2026-05-14T00:00:00+00:00")]
    )
    assert gaps == [
        {
            "hardware_tag": "Darwin-arm64-arm",
            "stage": "tier_d_showcase",
            "count": 1,
            "min_records": 2,
            "reason": "insufficient_history",
        }
    ]


def test_history_gaps_pass_two_rows_with_same_hardware_and_stage() -> None:
    records = [
        _row(seconds=1.0, generated_at="2026-05-14T00:00:00+00:00"),
        _row(seconds=1.1, generated_at="2026-05-14T00:15:00+00:00"),
    ]
    assert find_history_gaps(records) == []


def test_regression_detection_uses_prior_rows_on_valid_history() -> None:
    records = [
        _row(seconds=1.0, generated_at="2026-05-14T00:00:00+00:00"),
        _row(seconds=1.6, generated_at="2026-05-14T00:15:00+00:00"),
    ]
    regressions = find_regressions(records, threshold=0.25)
    assert regressions == [
        {
            "hardware_tag": "Darwin-arm64-arm",
            "stage": "tier_d_showcase",
            "latest_seconds": 1.6,
            "baseline_seconds": 1.0,
            "threshold": 0.25,
        }
    ]


def test_artifact_history_gaps_require_matching_hardware_stage_history() -> None:
    artifacts = [
        {
            "benchmark_id": "tier_d_showcase",
            "evidence_level": "flagship_workflow",
            "hardware_tag": "Darwin-arm64-arm",
            "performance_stage": "tier_d_showcase",
            "timings": {"status": "recorded", "seconds": 0.1},
        }
    ]
    single = [_row(seconds=0.1, generated_at="2026-05-14T00:00:00+00:00")]
    assert find_artifact_history_gaps(artifacts, single) == [
        {
            "benchmark_id": "tier_d_showcase",
            "hardware_tag": "Darwin-arm64-arm",
            "stage": "tier_d_showcase",
            "count": 1,
            "min_records": 2,
            "reason": "artifact_history_missing",
        }
    ]

    comparable = [
        _row(seconds=0.1, generated_at="2026-05-14T00:00:00+00:00"),
        _row(seconds=0.11, generated_at="2026-05-14T00:15:00+00:00"),
    ]
    assert find_artifact_history_gaps(artifacts, comparable) == []


def test_history_gap_min_records_must_be_trendable() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        find_history_gaps([], min_records=1)
