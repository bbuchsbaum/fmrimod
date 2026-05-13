"""Freshness and integrity checks for the public-API inventory.

Pairs with ``scripts/api_inventory.py`` and
``docs/contracts/api_inventory_v1.json``. The inventory is the review
target for namespace audits — silent drift in ``fmrimod.__all__``, the
opaque-forwarder count, or the ``Any``-in-signature count fails here
and forces a visible diff against the JSON, not just a bigger
``dir(fmrimod)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "docs" / "contracts" / "api_inventory_v1.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "api_inventory.py"

REQUIRED_ROW_KEYS = frozenset({
    "name",
    "tier",
    "owner_module",
    "signature_source",
    "runtime_signature",
    "used_by_public_seam_artifact",
    "compatibility_status",
    "opaque_forwarder",
    "has_any_in_signature",
    "is_callable",
    "is_class",
    "is_function",
    "has_untagged_kwargs_dict",
})


def _load_inventory() -> dict:
    if not INVENTORY_PATH.exists():
        pytest.fail(
            f"inventory missing at {INVENTORY_PATH}; regenerate with "
            f"`python scripts/api_inventory.py`"
        )
    return json.loads(INVENTORY_PATH.read_text())


def test_inventory_schema_is_v1() -> None:
    payload = _load_inventory()
    assert payload.get("schema_version") == "api_inventory/v1"


def test_inventory_rows_carry_all_required_columns() -> None:
    """Every row exposes the audit columns named in the bead body."""
    payload = _load_inventory()
    for row in payload["rows"]:
        missing = REQUIRED_ROW_KEYS - set(row)
        assert not missing, (
            f"row for {row.get('name')!r} missing columns: {sorted(missing)}"
        )


def test_inventory_matches_live_probe_for_all_names() -> None:
    """The set of names in the inventory matches ``fmrimod.__all__`` at HEAD.

    This is the cheap-pass guard from the bead's acceptance: hiding a
    name from ``__all__`` (or quietly adding one) drifts the inventory
    without a visible diff, which the audit is designed to prevent.
    Regenerate via ``python scripts/api_inventory.py`` when this fails.
    """
    import fmrimod

    declared = set(fmrimod.__all__)
    indexed = {row["name"] for row in _load_inventory()["rows"]}
    only_in_runtime = declared - indexed
    only_in_inventory = indexed - declared
    assert not (only_in_runtime or only_in_inventory), (
        f"inventory drifted from fmrimod.__all__:\n"
        f"  added at runtime: {sorted(only_in_runtime)}\n"
        f"  removed at runtime: {sorted(only_in_inventory)}\n"
        f"Regenerate via: python scripts/api_inventory.py"
    )


def test_inventory_counts_track_live_probe() -> None:
    """Top-line counts match a fresh probe.

    Failure here means the inventory file was hand-edited or the
    introspection logic in scripts/api_inventory.py drifted from
    fmrimod.__all__. Either regenerate the JSON, or treat the count
    delta as a real signature-soundness regression and review.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from api_inventory import build_inventory
    finally:
        sys.path.pop(0)

    live = build_inventory()
    on_disk = _load_inventory()
    assert live["counts"] == on_disk["counts"], (
        f"counts drifted between live probe and inventory:\n"
        f"  live:    {live['counts']}\n"
        f"  on disk: {on_disk['counts']}"
    )


def test_inventory_script_exists() -> None:
    assert SCRIPT_PATH.exists(), (
        "scripts/api_inventory.py is the generator referenced by the "
        "public_api_policy_v1.md inventory contract; it must exist."
    )
