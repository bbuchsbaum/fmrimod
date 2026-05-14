"""Absolute ratchet for the public-API inventory's review fields.

Pairs with ``docs/contracts/api_inventory_v1.json`` and the freshness
gate in ``test_api_inventory.py``. The freshness gate catches *new*
public names entering with ``tier=review_pending``; this file enforces
that *no* inventory row carries ``tier=review_pending`` or
``compatibility_status=review_pending`` after the burn-down landed in
bd-01KRHY6EQW0K06KESGF6MGVZT3 (closed 2026-05-14).

Two ratchets, one per review field:

1. **No ``tier=review_pending`` rows** (``test_review_pending_set_does_not_grow_beyond_baseline``).
   Every public name must enter the inventory with one of the ratified
   tier values: ``spine``, ``spine_review``, ``compat``,
   ``compat_pending_fix``, ``runtime_check``. The default
   ``review_pending`` is no longer a valid landing state. The
   ``BASELINE_REVIEW_PENDING_NAMES`` frozenset is intentionally empty;
   the structure is preserved so a future controlled re-baseline (e.g.,
   a documented API expansion that needs review_pending rows to land
   before classification) can add entries with a board-linked rationale
   without rewriting the tests.

2. **No ``compatibility_status=review_pending`` rows**
   (``test_no_review_pending_compatibility_status``). Parallel ratchet
   on the second review field — every row must carry a ratified
   ``compatibility_status``: ``spine``, ``compat``, ``deprecated``,
   ``hidden``, or ``internal_forwarder``.

Refs: bd-01KRHVVKM80SYC37BQ5HQKRMN1 (original tier-only baseline
ratchet), bd-01KRHY6EQW0K06KESGF6MGVZT3 (burn-down + parallel
compat_status ratchet).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "docs" / "contracts" / "api_inventory_v1.json"


# Public names whose tier is currently ``review_pending`` at the moment
# the burn-down ratchet landed. New entries to this set are forbidden;
# entries leave the set when the corresponding inventory row is tiered
# in the same commit.
BASELINE_REVIEW_PENDING_NAMES: frozenset[str] = frozenset()
"""Burn-down complete — every public name now carries an assigned tier.

Reaching the empty baseline tightens the ratchet: the
``test_review_pending_set_does_not_grow_beyond_baseline`` check now
fails for *any* ``tier=review_pending`` row, not just for new ones
beyond a pinned set. New public names must be tiered on entry; the
default ``review_pending`` is no longer a valid landing state.
"""


def _load_inventory() -> dict:
    if not INVENTORY_PATH.exists():
        pytest.fail(
            f"inventory missing at {INVENTORY_PATH}; regenerate with "
            f"`python scripts/api_inventory.py`"
        )
    return json.loads(INVENTORY_PATH.read_text())


def _current_review_pending() -> set[str]:
    return {
        row["name"]
        for row in _load_inventory()["rows"]
        if row["tier"] == "review_pending"
    }


def _current_review_pending_compatibility_status() -> set[str]:
    return {
        row["name"]
        for row in _load_inventory()["rows"]
        if row["compatibility_status"] == "review_pending"
    }


def test_baseline_size_is_known() -> None:
    """Sanity-check the pinned baseline size to catch silent drift."""
    assert len(BASELINE_REVIEW_PENDING_NAMES) == 0, (
        f"BASELINE_REVIEW_PENDING_NAMES size drifted from 0 to "
        f"{len(BASELINE_REVIEW_PENDING_NAMES)}. Update the assertion "
        f"in the same commit that intentionally changed the baseline."
    )


def test_review_pending_set_does_not_grow_beyond_baseline() -> None:
    """No new public names may carry tier=review_pending.

    Failure modes this catches:
    - A new name was added to ``fmrimod.__all__`` with the default
      ``tier=review_pending`` instead of being tiered on entry.
    - An already-tiered name was reverted to ``review_pending``.

    Cheap pass disqualified: silently widening
    ``BASELINE_REVIEW_PENDING_NAMES`` without an inventory tier change.
    The companion ``test_baseline_does_not_decay_silently`` catches that.
    """
    current = _current_review_pending()
    intruders = sorted(current - BASELINE_REVIEW_PENDING_NAMES)
    assert not intruders, (
        f"new review_pending names appeared in the inventory: {intruders}. "
        f"Either tier them in api_inventory_v1.json (the public-API "
        f"contract is no new public names enter without a tier), or — if "
        f"they are genuinely necessary additions — extend "
        f"BASELINE_REVIEW_PENDING_NAMES with a board-linked rationale "
        f"in the same commit (and update test_baseline_size_is_known)."
    )


def test_baseline_does_not_decay_silently() -> None:
    """Names removed from review_pending must come off the baseline.

    Failure modes this catches:
    - A row was tiered (now non-review_pending) but the baseline still
      lists it. The ratchet would silently allow the name back to
      review_pending later without surfacing the regression.
    - A name was removed from ``fmrimod.__all__`` entirely but the
      baseline still references it. Same silent-widening risk.

    To remove a name from the baseline: pair the deletion from
    ``BASELINE_REVIEW_PENDING_NAMES`` with the inventory tier change
    (or the ``__all__`` removal) in the same commit, and decrement the
    expected size in ``test_baseline_size_is_known``.
    """
    declared = {row["name"] for row in _load_inventory()["rows"]}
    current = _current_review_pending()

    # In the baseline but no longer in __all__ → stale (silently widens
    # the ratchet because the row can't appear in any future probe).
    no_longer_declared = sorted(BASELINE_REVIEW_PENDING_NAMES - declared)
    # In the baseline but no longer review_pending → tiered without
    # baseline cleanup.
    no_longer_pending = sorted(
        (BASELINE_REVIEW_PENDING_NAMES & declared) - current
    )

    stale = []
    if no_longer_declared:
        stale.append(f"removed from __all__: {no_longer_declared}")
    if no_longer_pending:
        stale.append(f"already tiered: {no_longer_pending}")
    assert not stale, (
        "BASELINE_REVIEW_PENDING_NAMES has stale entries — pair the "
        "tier or removal with a baseline cleanup in the same commit:\n  "
        + "\n  ".join(stale)
    )


def test_no_review_pending_compatibility_status() -> None:
    """Parallel ratchet on ``compatibility_status`` (bd-01KRHY6EQW0K06KESGF6MGVZT3).

    The ``tier`` ratchet above pins typing-soundness classification;
    this ratchet pins the public-compatibility classification
    (spine / compat / compat with debt / deprecated / hidden /
    internal_forwarder). Both must be assigned at the moment a name
    enters ``fmrimod.__all__`` — neither field may default to
    ``review_pending`` and stay there.

    Failure means a new public name was added without classifying
    its compatibility posture, or an already-classified row was
    reverted to ``review_pending``. Fix by setting
    ``compatibility_status`` to a ratified value
    (``spine``/``compat``/``deprecated``/``hidden``/``internal_forwarder``)
    in the same commit that introduces the row.
    """
    intruders = sorted(_current_review_pending_compatibility_status())
    assert not intruders, (
        "compatibility_status=review_pending rows in inventory: "
        f"{intruders}. Classify each as one of "
        "spine/compat/deprecated/hidden/internal_forwarder in the "
        "same commit (per public_api_policy_v1.md)."
    )
