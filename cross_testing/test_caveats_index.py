"""Ensure generated parity caveats are indexed with exit criteria."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, NamedTuple


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "benchmarks" / "parity"
CAVEATS_INDEX = ROOT / "docs" / "contracts" / "CAVEATS.md"


def _find_caveat_ids(value: Any) -> set[str]:
    if isinstance(value, dict):
        ids = set()
        caveat_id = value.get("caveat_id")
        if isinstance(caveat_id, str) and caveat_id:
            ids.add(caveat_id)
        for child in value.values():
            ids.update(_find_caveat_ids(child))
        return ids
    if isinstance(value, list):
        ids = set()
        for child in value:
            ids.update(_find_caveat_ids(child))
        return ids
    return set()


# ---------------------------------------------------------------------------
# Index-row parsing
# ---------------------------------------------------------------------------


class _CaveatRow(NamedTuple):
    caveat_id: str
    first_appearance: str
    affected_tiers: str
    owner: str
    exit_criterion: str


# A bead id is ``bd-`` followed by a 26-character Crockford Base32 ULID body.
# The exact alphabet (no I, L, O, U) makes typos easy to spot at this gate
# instead of letting them silently route past the index test.
_BEAD_ID_PATTERN = re.compile(r"^bd-[0-9A-HJKMNP-TV-Z]{26}$")


def _parse_index_rows() -> List[_CaveatRow]:
    """Parse the GitHub-Flavoured Markdown table at the top of CAVEATS.md.

    The format committed to the index::

        | Caveat ID | First appearance | Affected tiers | Owner | Exit criterion |
        | --- | --- | --- | --- | --- |
        | `caveat-id` | ... | ... | `bd-...` | ... |

    The parser skips the header row and the separator row, then returns
    one :class:`_CaveatRow` per data row.
    """
    rows: List[_CaveatRow] = []
    text = CAVEATS_INDEX.read_text()
    seen_separator = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            # We exit the table on the first non-table line after the
            # data block, but only after we've crossed the separator.
            if seen_separator and rows:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not seen_separator:
            # The separator row is the all-dashes row beneath the header.
            if all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
                seen_separator = True
            continue
        if len(cells) < 5:
            raise AssertionError(
                f"caveats index row has {len(cells)} cells, expected 5: {line!r}"
            )
        rows.append(
            _CaveatRow(
                caveat_id=cells[0],
                first_appearance=cells[1],
                affected_tiers=cells[2],
                owner=cells[3],
                exit_criterion=cells[4],
            )
        )
    return rows


def _strip_backticks(value: str) -> str:
    return value.strip().strip("`")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_checked_in_report_caveats_are_indexed():
    index = CAVEATS_INDEX.read_text()
    report_ids: set[str] = set()
    for report_path in REPORT_ROOT.glob("**/reports/*.json"):
        report_ids.update(_find_caveat_ids(json.loads(report_path.read_text())))

    assert report_ids, "Expected checked-in parity reports to contain caveats"
    missing = sorted(caveat_id for caveat_id in report_ids if f"`{caveat_id}`" not in index)
    assert missing == []


def test_caveats_index_table_parses_cleanly():
    """The index must remain in a known, machine-readable shape.

    Tests below depend on the parser; locking parseability here means a
    drift in the markdown shape fails with a clear "table couldn't be
    parsed" message rather than a misleading downstream assertion.
    """
    rows = _parse_index_rows()
    assert rows, (
        "Expected at least one caveat row in docs/contracts/CAVEATS.md; "
        "if all caveats have been retired, the table should still carry "
        "a header so the parser can confirm there are zero rows."
    )


def test_every_caveat_row_names_a_nonempty_owner_and_exit():
    """Every active caveat must name an owner and an exit criterion.

    The project's discipline rule (see the CAVEATS.md footer) is that
    a caveat without an owner and exit criterion is a tolerance escape
    hatch, not a caveat. Locking that here catches drift before
    reviewers do.
    """
    failures: list[str] = []
    for row in _parse_index_rows():
        if not _strip_backticks(row.owner):
            failures.append(f"  {row.caveat_id}: owner cell is blank")
        if not row.exit_criterion or row.exit_criterion in {"TBD", "TODO", "n/a"}:
            failures.append(
                f"  {row.caveat_id}: exit criterion is missing or a "
                f"placeholder ({row.exit_criterion!r})"
            )
    assert not failures, (
        "Caveat rows are missing required content:\n" + "\n".join(failures)
    )


def test_every_caveat_owner_matches_bead_id_shape():
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
    for row in _parse_index_rows():
        owner = _strip_backticks(row.owner)
        if not _BEAD_ID_PATTERN.match(owner):
            bad.append(
                f"  {row.caveat_id}: owner {owner!r} is not a "
                "canonical bd-<ULID> identifier"
            )
    assert not bad, (
        "Caveat owners must use the bd-<ULID> shape:\n" + "\n".join(bad)
    )
