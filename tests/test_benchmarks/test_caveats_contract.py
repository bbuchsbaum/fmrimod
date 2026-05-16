"""Regression guards for the shared caveats-index contract.

Bead ``bd-01KRRNANSD3YPNBD5DCE8K8SNB``: the two caveats-index tests
enforced divergent contracts and the strict one was *red-by-construction*
for a documented, owned caveat that has no benchmark report by design
(the typed path raises ``NotImplementedError`` so no workflow can
honestly emit it).

These pin the fix so it cannot silently rot:

* a no-report-classed caveat is live under the strict matcher (the trap
  is removed);
* an *un*-classed extra index row still fails the strict matcher (drift
  detection is preserved, not widened away — the cheap-pass
  disqualifier);
* a ``no_report_caveats`` entry must be a real, owned, indexed caveat
  (no orphan exemptions);
* both test modules reference the *same* contract functions (they
  cannot diverge again).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.parity.caveats_contract import (
    BEAD_ID_PATTERN,
    caveat_index_rows,
    declared_caveat_ids,
    live_caveat_ids,
    no_report_caveat_entries,
    no_report_caveat_ids,
)

ROOT = Path(__file__).resolve().parents[2]
_OWNER = "bd-01KRRNANSD3YPNBD5DCE8K8SNB"  # valid bd-<ULID> shape


def _write_fixture_root(
    tmp: Path,
    *,
    caveat_rows: list[tuple[str, str]],
    no_report: list[dict],
    artifacts: list[dict] | None = None,
) -> Path:
    """Build a minimal fake repo root the shared contract can read."""
    caveats = tmp / "docs" / "contracts"
    caveats.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Parity Caveats Index",
        "",
        "| Caveat ID | First appearance | Affected tiers | Owner | Exit criterion |",
        "| --- | --- | --- | --- | --- |",
    ]
    for caveat_id, owner in caveat_rows:
        lines.append(
            f"| `{caveat_id}` | `src.py:1` (no parity report) | none "
            f"| `{owner}` | port the thing |"
        )
    (caveats / "CAVEATS.md").write_text("\n".join(lines) + "\n")

    parity = tmp / "benchmarks" / "parity"
    parity.mkdir(parents=True, exist_ok=True)
    (parity / "proof_artifacts.json").write_text(
        json.dumps(
            {
                "schema_version": "parity-proof-artifacts/v1",
                "no_report_caveats": no_report,
                "artifacts": artifacts or [],
            }
        )
    )
    return tmp


def test_no_report_class_makes_documented_caveat_live(tmp_path: Path) -> None:
    """The exact trap from the bead: report-less + owned ⇒ strict-green."""
    root = _write_fixture_root(
        tmp_path,
        caveat_rows=[("foo-not-implemented", _OWNER)],
        no_report=[
            {
                "caveat_id": "foo-not-implemented",
                "owner": _OWNER,
                "reason": "typed path raises NotImplementedError; no workflow",
            }
        ],
    )
    assert no_report_caveat_ids(root) == {"foo-not-implemented"}
    assert live_caveat_ids(root) == {"foo-not-implemented"}
    # The strict invariant (set(rows) == live) the strict test asserts:
    assert set(caveat_index_rows(root)) == live_caveat_ids(root)


def test_unclassed_extra_row_still_fails_strict_drift_detection(
    tmp_path: Path,
) -> None:
    """Cheap-pass disqualifier: an extra row NOT in the no-report class
    must still break ``set(rows) == live`` so stale/undocumented rows are
    caught. The fix must not have widened the strict test to ignore
    extras."""
    root = _write_fixture_root(
        tmp_path,
        caveat_rows=[("bar-extra", _OWNER)],
        no_report=[],
    )
    assert set(caveat_index_rows(root)) == {"bar-extra"}
    assert live_caveat_ids(root) == set()
    assert set(caveat_index_rows(root)) != live_caveat_ids(root)


def test_declared_caveat_is_live_without_no_report_class(
    tmp_path: Path,
) -> None:
    """The normal report-backed path is unaffected by the new term."""
    root = _write_fixture_root(
        tmp_path,
        caveat_rows=[("qux", _OWNER)],
        no_report=[],
        artifacts=[{"benchmark_id": "b", "caveats": ["qux"]}],
    )
    assert declared_caveat_ids(root) == {"qux"}
    assert no_report_caveat_ids(root) == set()
    assert set(caveat_index_rows(root)) == live_caveat_ids(root) == {"qux"}


def test_real_no_report_entries_are_owned_indexed_and_reasoned() -> None:
    """No orphan exemptions: every real no-report entry must be a
    documented, owned CAVEATS.md row with a reason. (Currently the list
    is empty — the meta-regression caveat was retired — so this locks
    the invariant for any future entry rather than asserting presence.)"""
    indexed = caveat_index_rows(ROOT)
    for entry in no_report_caveat_entries(ROOT):
        cid = entry.get("caveat_id")
        assert cid in indexed, (
            f"no_report_caveats entry {cid!r} has no CAVEATS.md row"
        )
        owner = str(entry.get("owner", ""))
        assert BEAD_ID_PATTERN.match(owner), (
            f"no_report_caveats {cid!r} owner {owner!r} not a bd-<ULID>"
        )
        assert str(entry.get("reason", "")).strip(), (
            f"no_report_caveats {cid!r} must give a reason"
        )
        assert indexed[cid] == owner, (
            f"no_report_caveats {cid!r} owner disagrees with the index row"
        )


def test_both_test_modules_share_one_contract_module() -> None:
    """Anti-divergence lock: the strict and structural test modules must
    bind the *same* contract functions, so they cannot re-fork."""
    strict = pytest.importorskip(
        "tests.test_benchmarks.test_parity_proof_artifacts"
    )
    structural = pytest.importorskip("cross_testing.test_caveats_index")
    from benchmarks.parity import caveats_contract

    assert strict.caveat_index_rows is caveats_contract.caveat_index_rows
    assert strict.live_caveat_ids is caveats_contract.live_caveat_ids
    assert structural.caveat_index_rows is caveats_contract.caveat_index_rows
    assert structural.caveat_index_table is caveats_contract.caveat_index_table
    assert structural.report_caveat_ids is caveats_contract.report_caveat_ids
