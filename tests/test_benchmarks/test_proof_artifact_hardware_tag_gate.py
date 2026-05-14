"""Hardware-tag gate for `proof_artifacts.json`.

`MISSION.md:163-170` commits flagship proof artifacts to record
"runtime and stage-level timing on the recorded hardware tag."
The timing half is enforced by ``test_proof_artifact_timing_gate.py``;
this file enforces the hardware-tag half.

Why a separate gate: a recorded timing without a hardware tag is
meaningless (CPU/RAM/Python-version bound) and a hardware tag without
recorded timings is decoration. The two gates run independently so a
contributor wiring stage timings can land that work first and the
hardware-tag step second (or vice versa) without one blocking the
other.

Refs: bd-01KRHXD5DJ7CATXB0VNWE07JX0
(Board source: vision-drift-audit/post-01KRHXBHM321NJ0KK96CN81C27,
action 2 from the drift audit's reply chain).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from benchmarks.performance.check_regression import (
    find_artifact_history_gaps,
    load_records,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks" / "parity" / "proof_artifacts.json"
PERFORMANCE_HISTORY = (
    REPO_ROOT
    / "benchmarks"
    / "performance"
    / "fixtures"
    / "parity_performance_history.jsonl"
)

# Rows that need a hardware_tag but don't yet have one. Removing a row
# from this set requires actually adding a non-empty `hardware_tag`
# string to the manifest in the same commit — the assertions below
# verify that.
_PENDING_HARDWARE_TAG_ROWS = frozenset(
    {
        "tier_e_scrubbed_timebase_alignment",
    }
)

_GATED_LEVELS = frozenset({"flagship_workflow", "workflow_parity"})


def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST.read_text())


def test_pending_allowlist_only_names_actual_manifest_rows() -> None:
    """Allowlist entries must reference real manifest rows.

    Catches stale entries: a row removed from the manifest must also
    be removed from the pending list, otherwise the gate widens.
    """
    declared_ids = {a["benchmark_id"] for a in _load_manifest()["artifacts"]}
    stale = sorted(_PENDING_HARDWARE_TAG_ROWS - declared_ids)
    assert not stale, (
        f"_PENDING_HARDWARE_TAG_ROWS references rows not in the manifest: "
        f"{stale}. Remove them from the set."
    )


def test_pending_allowlist_only_names_gated_levels() -> None:
    """Allowlist entries should be flagship/workflow rows.

    Numerical canaries are not subject to the hardware-tag gate, so
    listing one is a category error. Catches drift in evidence_level
    for an allowlisted row.
    """
    by_id = {a["benchmark_id"]: a for a in _load_manifest()["artifacts"]}
    miscategorized = sorted(
        bid
        for bid in _PENDING_HARDWARE_TAG_ROWS
        if by_id.get(bid, {}).get("evidence_level") not in _GATED_LEVELS
    )
    assert not miscategorized, (
        f"_PENDING_HARDWARE_TAG_ROWS lists non-flagship/workflow rows: "
        f"{miscategorized}. Either move them off the allowlist (canaries "
        f"don't need hardware_tag) or fix the manifest's evidence_level."
    )


def test_each_gated_row_off_allowlist_carries_hardware_tag() -> None:
    """Strict gate: hardware_tag must be a non-empty string.

    Cheap pass disqualified: setting ``hardware_tag = ""`` (or
    ``None``) without recording a real platform identifier. The
    non-empty-string assertion fails on that case.

    To take a row off the ``_PENDING_HARDWARE_TAG_ROWS`` allowlist:
    1. Add a ``hardware_tag`` string to the row in
       ``benchmarks/parity/proof_artifacts.json`` (e.g.,
       ``"linux-x86_64-ci"`` or ``"darwin-arm64-2026"``).
    2. Remove the row's id from ``_PENDING_HARDWARE_TAG_ROWS`` here.
    3. This test then verifies the row at every CI run.
    """
    failures: list[str] = []
    for artifact in _load_manifest()["artifacts"]:
        if artifact["evidence_level"] not in _GATED_LEVELS:
            continue
        if artifact["benchmark_id"] in _PENDING_HARDWARE_TAG_ROWS:
            continue
        bid = artifact["benchmark_id"]
        tag = artifact.get("hardware_tag")
        if not isinstance(tag, str) or not tag:
            failures.append(
                f"{bid}: hardware_tag must be a non-empty string once off "
                f"the allowlist; got {tag!r}"
            )
    assert not failures, "\n  - " + "\n  - ".join(failures)


def test_no_new_flagship_or_workflow_row_skips_hardware_tag_silently() -> None:
    """Catches the 'forgot to register a new pending row' regression.

    If someone adds a new flagship/workflow row without ``hardware_tag``
    AND without listing it in ``_PENDING_HARDWARE_TAG_ROWS``, the
    strict-gate test above will fail. This test surfaces the same
    condition with a single explicit assertion in case the iteration
    order makes the failure hard to read.
    """
    declared = _load_manifest()["artifacts"]
    silent_skips = []
    for a in declared:
        if a["evidence_level"] not in _GATED_LEVELS:
            continue
        if a["benchmark_id"] in _PENDING_HARDWARE_TAG_ROWS:
            continue
        tag = a.get("hardware_tag")
        if not isinstance(tag, str) or not tag:
            silent_skips.append(a["benchmark_id"])
    assert not silent_skips, (
        f"flagship/workflow rows lack hardware_tag AND are not on "
        f"the _PENDING_HARDWARE_TAG_ROWS allowlist: {silent_skips}. "
        f"Either add the hardware_tag string or register the id in "
        f"the allowlist with a board-linked rationale in the same commit."
    )


def test_recorded_hardware_tag_rows_join_comparable_performance_history() -> None:
    """A hardware tag is useful only when it joins at least two history rows."""

    artifacts = _load_manifest()["artifacts"]
    gaps = find_artifact_history_gaps(
        artifacts,
        load_records(PERFORMANCE_HISTORY),
    )
    assert not gaps, (
        "Recorded proof-artifact timings must join a comparable "
        f"hardware_tag+stage history: {gaps}"
    )
