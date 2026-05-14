"""Timing-receipt gate for `proof_artifacts.json`.

`MISSION.md:62-65` and `MISSION.md:163-170` commit to stage-level timing
on every flagship/workflow proof artifact. Today most rows carry
``timings.status == "not_recorded"`` (or no ``timings`` dict at all),
which means the receipts don't actually back the speed-axis claim from
``MISSION.md:94-95``.

This module enforces two things:

1. Any flagship_workflow / workflow_parity row added in the future must
   ship with recorded timings (status="recorded" plus a numeric seconds
   field), or it must be added to the explicit ``_PENDING_TIMING_ROWS``
   allowlist below with a board-linked rationale.
2. Any row removed from the allowlist must demonstrate real recorded
   timings — flipping the status string alone is the cheap-pass and is
   disqualified by the numeric-seconds assertion.

Refs: bd-01KRHVJ8TM7SYBZVRTJK5VKCZY (board source:
major-issues-lets-talk/post-01KRHVA1NC98BP4KQ1Z29WYXWG, bullet 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks" / "parity" / "proof_artifacts.json"

# Rows whose flagship/workflow classification is real but whose stage-level
# timings are not yet wired into the parity harness. Each row in this set
# is a current TODO under the timing-recording slice. Removing a row from
# this set requires the row to actually carry recorded numeric timings —
# the assertions below verify that, so a no-op flip of `status` from
# "not_recorded" to "recorded" without a payload still fails the test.
_PENDING_TIMING_ROWS = frozenset(
    {
        "tier_a_f_confound_drift",
        "tier_a_multicollinear_baseline",
        "tier_a_multirun_concat_public_seam",
        "tier_c_second_level",
        "tier_e_parametric_centering",
        "tier_e_scrubbed_timebase_alignment",
        "tier_e_semantic_contrast_alignment",
    }
)

_GATED_LEVELS = frozenset({"flagship_workflow", "workflow_parity"})

# Numeric fields any of which counts as a real timing payload. Keeping the
# accepted set narrow and explicit makes the cheap pass ("set status to
# recorded with no payload") fail visibly.
_NUMERIC_TIMING_FIELDS = ("seconds", "seconds_total", "wall_seconds")


def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST.read_text())


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def test_pending_allowlist_only_names_actual_manifest_rows() -> None:
    """Every entry in the allowlist must point at a real manifest row.

    Catches stale entries: a row removed from the manifest must also be
    removed from the pending list, otherwise the gate silently widens.
    """
    declared_ids = {a["benchmark_id"] for a in _load_manifest()["artifacts"]}
    stale = sorted(_PENDING_TIMING_ROWS - declared_ids)
    assert not stale, (
        f"_PENDING_TIMING_ROWS references rows not in the manifest: {stale}. "
        f"Remove them from the set."
    )


def test_pending_allowlist_only_names_gated_levels() -> None:
    """Allowlist entries should be flagship/workflow rows.

    Numerical canaries are not subject to the timing gate, so adding
    one to the allowlist is a category error. Catches drift in the
    manifest's evidence_level for an allowlisted row.
    """
    by_id = {a["benchmark_id"]: a for a in _load_manifest()["artifacts"]}
    miscategorized = sorted(
        bid
        for bid in _PENDING_TIMING_ROWS
        if by_id.get(bid, {}).get("evidence_level") not in _GATED_LEVELS
    )
    assert not miscategorized, (
        f"_PENDING_TIMING_ROWS lists non-flagship/workflow rows: "
        f"{miscategorized}. Either move them off the allowlist (canaries "
        f"don't need timings) or fix the manifest's evidence_level."
    )


def test_each_gated_row_off_allowlist_carries_recorded_numeric_timings() -> None:
    """Strict gate: timings must be recorded with a numeric payload.

    Iterates every flagship/workflow row not in ``_PENDING_TIMING_ROWS``
    and asserts the timings dict has ``status == "recorded"`` plus one
    of the numeric payload fields in ``_NUMERIC_TIMING_FIELDS``.

    Cheap pass disqualified: setting ``timings.status = "recorded"``
    without any numeric payload. The assertion fails on that case, so
    a contributor can't flip the status string and call it done.

    To take a row off the ``_PENDING_TIMING_ROWS`` allowlist:
    1. Wire the parity harness to record stage-level timings into the
       row's ``timings`` dict (status="recorded" + numeric seconds).
    2. Remove the row's id from ``_PENDING_TIMING_ROWS`` in this file.
    3. This test will then verify the row at every CI run.
    """
    failures: list[str] = []
    for artifact in _load_manifest()["artifacts"]:
        if artifact["evidence_level"] not in _GATED_LEVELS:
            continue
        if artifact["benchmark_id"] in _PENDING_TIMING_ROWS:
            continue
        bid = artifact["benchmark_id"]
        timings = artifact.get("timings")
        if not isinstance(timings, dict):
            failures.append(f"{bid}: missing 'timings' dict (got {timings!r})")
            continue
        if timings.get("status") != "recorded":
            failures.append(
                f"{bid}: timings.status must be 'recorded' once off the "
                f"allowlist; got {timings.get('status')!r}"
            )
            continue
        payload_value = next(
            (timings[k] for k in _NUMERIC_TIMING_FIELDS if k in timings),
            None,
        )
        if payload_value is None or not _is_numeric(payload_value):
            failures.append(
                f"{bid}: status='recorded' but no numeric payload "
                f"(expected one of {list(_NUMERIC_TIMING_FIELDS)} with "
                f"numeric value)"
            )
    assert not failures, "\n  - " + "\n  - ".join(failures)


def test_no_new_flagship_or_workflow_row_skips_timings_silently() -> None:
    """Catches the 'forgot to register a new pending row' regression.

    If someone adds a new flagship/workflow row without recorded
    timings AND without listing it in ``_PENDING_TIMING_ROWS``, the
    parametrized test above will fail with a clear message. This test
    surfaces the same condition with a single explicit assertion in
    case the parametrize ID gets lost in long output.
    """
    declared = _load_manifest()["artifacts"]
    silent_skips = []
    for a in declared:
        if a["evidence_level"] not in _GATED_LEVELS:
            continue
        if a["benchmark_id"] in _PENDING_TIMING_ROWS:
            continue
        timings = a.get("timings") or {}
        if timings.get("status") != "recorded":
            silent_skips.append(a["benchmark_id"])
    assert not silent_skips, (
        f"flagship/workflow rows lack recorded timings AND are not on "
        f"the _PENDING_TIMING_ROWS allowlist: {silent_skips}. Either "
        f"record timings or add the id to the allowlist with a board-"
        f"linked rationale in the same commit."
    )
