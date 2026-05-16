"""Structural caveats-index guarantees (built on the shared contract).

This module enforces the *structural* half of the caveats-index
contract: every caveat in a checked-in report is indexed, every row is
well-formed, and every owner is a canonical ``bd-<ULID>``. The strict
set-equality half (``set(rows) == declared ∪ reported ∪ no_report``)
lives in ``tests/test_benchmarks/test_parity_proof_artifacts.py``.

Both modules now import the same primitives from
:mod:`benchmarks.parity.caveats_contract` so the two contracts cannot
silently diverge again (the failure mode bead
``bd-01KRRNANSD3YPNBD5DCE8K8SNB`` was filed against).
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.parity.caveats_contract import (
    BEAD_ID_PATTERN,
    caveat_index_rows,
    caveat_index_table,
    report_caveat_ids,
    strip_backticks,
)

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_report_caveats_are_indexed() -> None:
    indexed = set(caveat_index_rows(ROOT))
    missing = sorted(
        caveat_id
        for caveat_id in report_caveat_ids(ROOT)
        if caveat_id not in indexed
    )
    assert missing == []


def test_caveats_index_table_parses_cleanly() -> None:
    """The index must remain in a known, machine-readable shape.

    Tests below depend on the parser; locking parseability here means a
    drift in the markdown shape fails with a clear "table couldn't be
    parsed" message rather than a misleading downstream assertion.
    """
    rows = caveat_index_table(ROOT)
    assert isinstance(rows, list)


def test_every_caveat_row_names_a_nonempty_owner_and_exit() -> None:
    """Every active caveat must name an owner and an exit criterion.

    The project's discipline rule (see the CAVEATS.md footer) is that
    a caveat without an owner and exit criterion is a tolerance escape
    hatch, not a caveat. Locking that here catches drift before
    reviewers do.
    """
    failures: list[str] = []
    for row in caveat_index_table(ROOT):
        if not strip_backticks(row.owner):
            failures.append(f"  {row.caveat_id}: owner cell is blank")
        if not row.exit_criterion or row.exit_criterion in {"TBD", "TODO", "n/a"}:
            failures.append(
                f"  {row.caveat_id}: exit criterion is missing or a "
                f"placeholder ({row.exit_criterion!r})"
            )
    assert not failures, (
        "Caveat rows are missing required content:\n" + "\n".join(failures)
    )


def test_every_caveat_owner_matches_bead_id_shape() -> None:
    """Owners must be canonical ``bd-<26-char ULID>`` strings.

    This is a shape check, not an existence check (the project's
    tracker substrate is in flux; the index can't claim a live bead
    *exists*, only that someone has committed to one in the canonical
    shape). It catches:

    - Blank owners (handled above too).
    - Owners that are free-text names rather than tracker IDs.
    - Typos that drop a character or use disallowed Base32 letters
      (I, L, O, U), which would otherwise route past the existing
      report-index gate.
    """
    bad: list[str] = []
    for row in caveat_index_table(ROOT):
        owner = strip_backticks(row.owner)
        if not BEAD_ID_PATTERN.match(owner):
            bad.append(
                f"  {row.caveat_id}: owner {owner!r} is not a "
                "canonical bd-<ULID> identifier"
            )
    assert not bad, (
        "Caveat owners must use the bd-<ULID> shape:\n" + "\n".join(bad)
    )
