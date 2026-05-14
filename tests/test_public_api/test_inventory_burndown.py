"""Burn-down ratchet for the public-API inventory.

Pairs with ``docs/contracts/api_inventory_v1.json`` and the freshness
gate in ``test_api_inventory.py``. The freshness gate catches *new*
public names entering with ``tier=review_pending``; this file pins the
*existing* review_pending set as a baseline and forces it to shrink
monotonically.

Two assertions, both catch real failure modes:

1. **No new review_pending rows.** Tiering a name to spine /
   spine_review / compat / compat_pending_fix / runtime_check is the
   only way ``tier=review_pending`` can change. Adding a new public
   name with ``review_pending`` (rather than tiering it on entry) or
   reverting an already-tiered name back to ``review_pending`` both
   fail the subset assertion below.

2. **Baseline doesn't decay silently.** When a name is tiered or
   removed from ``__all__``, the baseline ``BASELINE_REVIEW_PENDING_NAMES``
   set must be updated in the same commit. Otherwise the burn-down
   ratchet silently widens over time and a re-tiering becomes invisible.

To take a name off the baseline:
- Tier it in ``docs/contracts/api_inventory_v1.json`` (any value other
  than ``"review_pending"``).
- Remove it from ``BASELINE_REVIEW_PENDING_NAMES`` below.
- Both edits land in the same commit, with a board-linked rationale
  for the chosen tier.

Refs: bd-01KRHVVKM80SYC37BQ5HQKRMN1
(Board source: major-issues-lets-talk/post-01KRHVA1NC98BP4KQ1Z29WYXWG,
bullet 4 burn-down framing).
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
BASELINE_REVIEW_PENDING_NAMES = frozenset({
    "Fcontrasts", "HRF", "Poly", "SamplingFrame",
    "SpecSerializationError", "afni_restricted_plan", "as_hrf", "bootstrap_glm",
    "build_nuisance_projector", "condition_basis_list", "contrast_weights", "design_matrix",
    "estimate_betas", "estimate_hrf", "event_factor", "event_matrix",
    "event_variable", "fit_noise", "gen_empirical_hrf", "gen_hrf_library",
    "latent_dataset", "lsa_single_trial", "lss_single_trial", "simulate_simple_dataset",
    "soft_subspace_options", "stats", "write_results",
})


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


def test_baseline_size_is_known() -> None:
    """Sanity-check the pinned baseline size to catch silent drift."""
    assert len(BASELINE_REVIEW_PENDING_NAMES) == 27, (
        f"BASELINE_REVIEW_PENDING_NAMES size drifted from 27 to "
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
