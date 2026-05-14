"""Freshness + baseline tests for the parity tolerance rationale audit.

Pairs with ``scripts/tolerance_audit.py`` and
``docs/contracts/parity_tolerance_audit_v1.json``. The audit is the
review target for the ``rtol``/``atol`` escape question raised in
``vision-drift-audit/post-01KRHX267HQWZKD270K9XD3PNA``.

Two assertions worth keeping at all times:

1. The audit JSON tracks the live AST probe — silent additions of new
   tolerance call-sites must be visible diffs.
2. The ``unjustified`` count does not silently grow above the pinned
   baseline. Adding a new elevated tolerance requires either pairing
   it with a rationale comment (caveat_id / numerical_floor / see
   CAVEATS) or raising the baseline with board-linked rationale.

Refs: bd-01KRHXER0A33DBM403ABW8EY15
(Board source: vision-drift-audit/post-01KRHX267HQWZKD270K9XD3PNA).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs" / "contracts" / "parity_tolerance_audit_v1.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "tolerance_audit.py"

_VALID_CLASSIFICATIONS = frozenset({
    "justified",
    "baseline_default",
    "unjustified",
    "unclassified",
})

_REQUIRED_ROW_KEYS = frozenset({
    "file",
    "lineno",
    "kwarg",
    "value",
    "value_repr",
    "classification",
    "context_marker_found",
})


def _load_audit() -> dict:
    if not AUDIT_PATH.exists():
        pytest.fail(
            f"audit missing at {AUDIT_PATH}; regenerate with "
            f"`python scripts/tolerance_audit.py`"
        )
    return json.loads(AUDIT_PATH.read_text())


def test_audit_schema_is_v1() -> None:
    payload = _load_audit()
    assert payload.get("schema_version") == "parity_tolerance_audit/v1"


def test_audit_rows_carry_required_columns() -> None:
    for row in _load_audit()["rows"]:
        missing = _REQUIRED_ROW_KEYS - set(row)
        assert not missing, (
            f"audit row {row.get('file')}:{row.get('lineno')} missing "
            f"columns: {sorted(missing)}"
        )


def test_audit_classifications_are_valid() -> None:
    invalid = []
    for row in _load_audit()["rows"]:
        c = row.get("classification")
        if c not in _VALID_CLASSIFICATIONS:
            invalid.append(f"{row['file']}:{row['lineno']} classification={c!r}")
    assert not invalid, (
        "rows with invalid classification (must be one of "
        f"{sorted(_VALID_CLASSIFICATIONS)}):\n  - "
        + "\n  - ".join(invalid)
    )


def test_audit_counts_track_live_probe() -> None:
    """Audit JSON must match a fresh AST walk.

    Failure means the audit was hand-edited or new tolerance call-sites
    were added without regenerating. Either regenerate via
    ``python scripts/tolerance_audit.py`` or treat the delta as a real
    soundness issue and review.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from tolerance_audit import build_audit
    finally:
        sys.path.pop(0)

    live = build_audit()
    on_disk = _load_audit()
    assert live["counts"] == on_disk["counts"], (
        f"tolerance audit counts drifted between live probe and on-disk:\n"
        f"  live:    {live['counts']}\n"
        f"  on disk: {on_disk['counts']}"
    )


def test_audit_unjustified_count_does_not_silently_grow() -> None:
    """Pin the unjustified-tolerance baseline at the moment of audit landing.

    Adding a new ``rtol=`` / ``atol=`` keyword with an elevated value
    and no nearby rationale comment will raise this count and trip
    this test. To add a tolerance:

    1. If the elevated value is masking a real divergence, file a
       structured ``caveat_id`` row in ``docs/contracts/CAVEATS.md``
       and add a ``# caveat_id: <id>`` comment within 3 lines of the
       call. The audit will reclassify it as ``"justified"`` and the
       unjustified count won't grow.
    2. If the value is genuinely a numerical floor (round-trip jitter,
       float64 epsilon, etc.), add a ``# numerical_floor`` comment.
       Same outcome.
    3. If neither applies and the count must grow, raise the baseline
       in this assertion with a board-linked rationale in the same
       commit.
    """
    counts = _load_audit()["counts"]["by_classification"]
    BASELINE_UNJUSTIFIED = 28
    assert counts["unjustified"] <= BASELINE_UNJUSTIFIED, (
        f"unjustified-tolerance count grew from {BASELINE_UNJUSTIFIED} "
        f"to {counts['unjustified']}; either justify the new call-site "
        f"with a rationale comment or raise the baseline."
    )


def test_audit_unclassified_count_does_not_silently_grow() -> None:
    """Pin the unclassified-tolerance baseline (8 at landing).

    ``unclassified`` rows are calls whose value couldn't be parsed as a
    literal (variables, expressions, dynamic constructions). They're
    not necessarily wrong but they hide from the gate, so growth here
    is also a regression.
    """
    counts = _load_audit()["counts"]["by_classification"]
    BASELINE_UNCLASSIFIED = 8
    assert counts["unclassified"] <= BASELINE_UNCLASSIFIED, (
        f"unclassified-tolerance count grew from {BASELINE_UNCLASSIFIED} "
        f"to {counts['unclassified']}; new dynamic-tolerance sites need "
        f"either a literal value or a rationale comment, or the "
        f"baseline raised with rationale."
    )


def test_audit_script_exists() -> None:
    assert SCRIPT_PATH.exists(), (
        "scripts/tolerance_audit.py is the generator referenced by "
        "the parity tolerance contract; it must exist."
    )
