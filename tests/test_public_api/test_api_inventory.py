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


# Spine names — the four-stage seam dataset → lm → contrast → group_fit.
# Tier values are pinned so a future commit cannot silently demote a spine
# entry by changing its tier in the inventory or by removing it from
# fmrimod.__all__. Promotion of a new spine name is a visible diff to both
# this set and the inventory JSON.
_EXPECTED_SPINE_TIERS = {
    "fmri_dataset": "spine",
    "fmri_lm": "spine_review",
    "fit_glm_from_matrix": "spine",
    "fit_glm_from_suffstats": "spine",
    "Spec": "spine",
    "Term": "spine",
    "event_model": "spine_review",
    "baseline_model": "spine",
    "fmri_meta": "spine",
    "fmri_ttest": "spine",
    "group_data_from_fmrilm": "spine",
    "estimate_single_trial": "spine",
    "estimate_single_trial_from_dataset": "spine",
}


def test_spine_names_are_tier_assigned() -> None:
    """The 13 spine names hold their assigned tier values.

    Cheap-pass disqualified: relabeling a spine name to ``review_pending``
    or removing the row from the inventory would silently regress this
    audit. The fix for a real demotion is to update ``_EXPECTED_SPINE_TIERS``
    in the same diff that demotes the name, with a board-linked rationale.
    """
    rows_by_name = {row["name"]: row for row in _load_inventory()["rows"]}
    missing = sorted(set(_EXPECTED_SPINE_TIERS) - set(rows_by_name))
    assert not missing, (
        f"spine names absent from inventory: {missing}; either re-add "
        f"to fmrimod.__all__ or update _EXPECTED_SPINE_TIERS with a "
        f"board-linked rationale for the demotion."
    )
    mismatches = []
    for name, expected_tier in _EXPECTED_SPINE_TIERS.items():
        actual = rows_by_name[name]["tier"]
        if actual != expected_tier:
            mismatches.append(f"{name}: expected {expected_tier!r}, got {actual!r}")
    assert not mismatches, (
        "spine tier drift:\n  " + "\n  ".join(mismatches)
    )


_INTERNAL_AUDIT_PATH = REPO_ROOT / "docs" / "contracts" / "internal_any_audit.json"
_INTERNAL_REQUIRED_ROW_KEYS = frozenset({
    "module",
    "qualname",
    "lineno",
    "endlineno",
    "is_async",
    "has_any_annotation",
    "has_var_kwargs",
    "has_var_args_any",
    "seam_class",
    "coercion_exemption",
    "owner_bead",
})
_VALID_SEAM_CLASSES = frozenset({"public", "compat", "adapter", "internal"})


def _load_internal_audit() -> dict:
    if not _INTERNAL_AUDIT_PATH.exists():
        pytest.fail(
            f"internal audit missing at {_INTERNAL_AUDIT_PATH}; "
            f"regenerate with `python scripts/api_inventory.py --mode internal`"
        )
    return json.loads(_INTERNAL_AUDIT_PATH.read_text())


def test_internal_audit_schema_is_v1() -> None:
    payload = _load_internal_audit()
    assert payload.get("schema_version") == "internal_any_audit/v1"


def test_internal_audit_rows_carry_all_required_columns() -> None:
    for row in _load_internal_audit()["rows"]:
        missing = _INTERNAL_REQUIRED_ROW_KEYS - set(row)
        assert not missing, (
            f"internal audit row {row.get('module')}::{row.get('qualname')} "
            f"missing columns: {sorted(missing)}"
        )


def test_internal_audit_rows_have_valid_seam_class() -> None:
    """Every row's ``seam_class`` is one of the four documented values.

    Catches drift in the seam-classification logic — e.g., a new
    ``"compat"``-style suffix that the classifier doesn't recognize, or
    a manual JSON edit that put a free-text label in the column.
    """
    invalid = []
    for row in _load_internal_audit()["rows"]:
        seam = row.get("seam_class")
        if seam not in _VALID_SEAM_CLASSES:
            invalid.append(f"{row['module']}::{row['qualname']}: seam_class={seam!r}")
    assert not invalid, (
        f"rows with invalid seam_class (must be one of {sorted(_VALID_SEAM_CLASSES)}):\n  - "
        + "\n  - ".join(invalid)
    )


def test_internal_audit_counts_carry_per_seam_breakdown() -> None:
    """Counts include a per-seam breakdown so burn-down can be prioritized.

    The aggregate count of 394 ``with_any_annotation`` is unsortable
    without knowing which rows are public-facing vs internal vs
    legitimate boundary coercion. This test asserts the
    ``by_seam_class`` breakdown exists and the per-seam totals sum to
    the aggregate.
    """
    counts = _load_internal_audit()["counts"]
    by_seam = counts.get("by_seam_class")
    assert isinstance(by_seam, dict), (
        f"counts must include by_seam_class breakdown; got {by_seam!r}"
    )
    for seam in _VALID_SEAM_CLASSES:
        assert seam in by_seam, f"by_seam_class missing entry for {seam!r}"
        for key in ("total", "with_any_annotation", "with_var_kwargs"):
            assert key in by_seam[seam], (
                f"by_seam_class[{seam!r}] missing {key!r}: {by_seam[seam]}"
            )
    # Sums match the aggregates — no rows go uncounted.
    seam_total = sum(by_seam[s]["total"] for s in _VALID_SEAM_CLASSES)
    assert seam_total == counts["total_functions"], (
        f"per-seam totals sum to {seam_total} but aggregate is "
        f"{counts['total_functions']}"
    )
    seam_any = sum(by_seam[s]["with_any_annotation"] for s in _VALID_SEAM_CLASSES)
    assert seam_any == counts["with_any_annotation"], (
        f"per-seam Any sum to {seam_any} but aggregate is "
        f"{counts['with_any_annotation']}"
    )


def test_internal_audit_public_seam_any_count_does_not_silently_grow() -> None:
    """Pin the ``seam_class=='public'`` Any baseline at the moment of landing.

    The aggregate baseline (394) is broad — it lets internal Any-leaks
    swap with public Any-leaks 1:1. The public-seam-specific baseline
    is the priority gauge: a new ``Any`` in a public-facing function
    is more user-visible than a new ``Any`` in a deep internal helper.
    """
    by_seam = _load_internal_audit()["counts"].get("by_seam_class", {})
    public_any = by_seam.get("public", {}).get("with_any_annotation", 0)
    BASELINE_PUBLIC_WITH_ANY = 30
    assert public_any <= BASELINE_PUBLIC_WITH_ANY, (
        f"public-seam Any-annotation count grew from "
        f"{BASELINE_PUBLIC_WITH_ANY} to {public_any}; this is a higher-"
        f"priority regression than internal-seam growth. Either narrow "
        f"the offending Any or raise the baseline with rationale."
    )


def test_internal_audit_counts_track_live_probe() -> None:
    """The on-disk audit matches a fresh AST walk.

    Failure means the audit JSON was hand-edited or fmrimod source has
    changed since the audit was last regenerated. Either regenerate via
    ``python scripts/api_inventory.py --mode internal`` or treat the
    delta as a real soundness regression and review.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from api_inventory import build_internal_audit
    finally:
        sys.path.pop(0)

    live = build_internal_audit()
    on_disk = _load_internal_audit()
    assert live["counts"] == on_disk["counts"], (
        f"internal audit counts drifted between live probe and on-disk:\n"
        f"  live:    {live['counts']}\n"
        f"  on disk: {on_disk['counts']}"
    )


def test_internal_audit_any_count_does_not_silently_grow() -> None:
    """Pin the internal Any-annotation baseline.

    The gap between public-surface ``Any`` (~1) and internal ``Any``
    (~hundreds) is what the scathing-read post in
    ``general-discussion/post-01KRHT4H90S9695V0PWRW32Q6Y`` flagged.
    Adding new internal ``Any`` annotations in fmrimod modules should
    require a visible diff against this baseline, not slip past the
    public-surface inventory's narrower gate.

    To intentionally raise the baseline (e.g. you've added a new
    module wholesale and the `Any` is genuinely needed), update both
    the audit JSON and this assertion in the same commit with a
    board-linked rationale.
    """
    counts = _load_internal_audit()["counts"]
    # Baseline pinned at the moment the internal audit landed; raise
    # only with explicit board-linked rationale in the same commit.
    BASELINE_WITH_ANY = 394
    assert counts["with_any_annotation"] <= BASELINE_WITH_ANY, (
        f"internal Any-annotation count grew from {BASELINE_WITH_ANY} "
        f"to {counts['with_any_annotation']}; raise the baseline with "
        f"rationale or narrow the offending Any."
    )


def test_internal_audit_var_kwargs_count_does_not_silently_grow() -> None:
    """Pin the internal ``**kwargs`` baseline (165 at audit landing)."""
    counts = _load_internal_audit()["counts"]
    BASELINE_WITH_VAR_KWARGS = 165
    assert counts["with_var_kwargs"] <= BASELINE_WITH_VAR_KWARGS, (
        f"internal **kwargs count grew from {BASELINE_WITH_VAR_KWARGS} "
        f"to {counts['with_var_kwargs']}; raise the baseline with "
        f"rationale or type the kwargs."
    )


def test_spine_review_names_have_documented_soundness_debt() -> None:
    """Every ``spine_review`` row carries a real soundness flag.

    The ``spine_review`` tier exists for spine entries that are
    load-bearing but have a known typing debt — opaque ``**kwargs``,
    untagged ``Union`` over many cases, or ``Any`` in the resolved
    signature. If a row is tagged ``spine_review`` but has none of
    those flags, either the tier is wrong or the audit columns are
    stale.
    """
    rows_by_name = {row["name"]: row for row in _load_inventory()["rows"]}
    for name, expected_tier in _EXPECTED_SPINE_TIERS.items():
        if expected_tier != "spine_review":
            continue
        row = rows_by_name[name]
        flagged = (
            row.get("opaque_forwarder")
            or row.get("has_any_in_signature")
            or row.get("has_untagged_kwargs_dict")
        )
        assert flagged, (
            f"{name} is tier={expected_tier!r} but has no documented "
            f"soundness flag: {row}. Either reclassify or update the "
            f"introspection in scripts/api_inventory.py."
        )
