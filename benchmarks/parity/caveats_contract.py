"""Single source of truth for the parity caveats-index contract.

Two test modules used to parse ``docs/contracts/CAVEATS.md`` and the
parity reports independently and assert *divergent* contracts:

* ``cross_testing/test_caveats_index.py`` — structural: every caveat in
  a checked-in report is indexed; rows are well-formed; owners are
  ``bd-<ULID>``. It tolerates extra index rows.
* ``tests/test_benchmarks/test_parity_proof_artifacts.py`` — strict:
  ``set(index_rows) == declared ∪ reported`` exactly. An index row that
  is neither declared by a manifest artifact nor emitted by a generated
  report is "extra-in-left" and fails.

The strict contract had no class for a caveat that is *documented and
owned but has no benchmark surface by design* — e.g. a typed path that
raises ``NotImplementedError`` so no workflow can honestly emit it
(``bd-01KRRMM5AEAKVQWAYWKTZRF263`` traced this; fabricating a canary was
disqualified as dishonest). Such a caveat was red-by-construction
against the strict test even though it satisfies every CAVEATS.md
maintenance rule.

This module fixes both halves of the problem (bead
``bd-01KRRNANSD3YPNBD5DCE8K8SNB``):

1. an explicit *no-report caveat class* — the manifest's top-level
   ``no_report_caveats`` list — that :func:`live_caveat_ids` treats as
   live without requiring a report;
2. one shared set of contract primitives both test modules import, so
   the two contracts cannot silently diverge again.

Drift detection is preserved, not widened away: only caveat ids that are
*explicitly* allow-listed in ``no_report_caveats`` (with an owner and a
reason) are exempt. Any other un-backed index row still fails the strict
matcher.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

__all__ = [
    "CaveatRow",
    "caveats_index_path",
    "manifest_path",
    "strip_backticks",
    "caveat_index_table",
    "caveat_index_rows",
    "report_caveat_ids",
    "declared_caveat_ids",
    "no_report_caveat_entries",
    "no_report_caveat_ids",
    "live_caveat_ids",
]


class CaveatRow(NamedTuple):
    caveat_id: str
    first_appearance: str
    affected_tiers: str
    owner: str
    exit_criterion: str


def caveats_index_path(root: Path) -> Path:
    return root / "docs" / "contracts" / "CAVEATS.md"


def manifest_path(root: Path) -> Path:
    return root / "benchmarks" / "parity" / "proof_artifacts.json"


def strip_backticks(value: str) -> str:
    return value.strip().strip("`")


def caveat_index_table(root: Path) -> list[CaveatRow]:
    """Parse the GitHub-Flavoured Markdown table at the top of CAVEATS.md.

    Skips the header and the all-dashes separator, then returns one
    :class:`CaveatRow` per data row. Raises if a data row does not have
    the committed five-column shape so a markdown drift fails loudly
    here rather than as a misleading downstream assertion.
    """
    rows: list[CaveatRow] = []
    seen_separator = False
    for line in caveats_index_path(root).read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if seen_separator and rows:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not seen_separator:
            if all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
                seen_separator = True
            continue
        if len(cells) < 5:
            raise AssertionError(
                f"caveats index row has {len(cells)} cells, expected 5: {line!r}"
            )
        rows.append(
            CaveatRow(
                caveat_id=strip_backticks(cells[0]),
                first_appearance=cells[1],
                affected_tiers=cells[2],
                owner=cells[3],
                exit_criterion=cells[4],
            )
        )
    return rows


def caveat_index_rows(root: Path) -> dict[str, str]:
    """``{caveat_id: owner}`` for every active row (owner backticks stripped)."""
    return {
        row.caveat_id: strip_backticks(row.owner)
        for row in caveat_index_table(root)
    }


def _walk_caveat_ids(value: Any) -> set[str]:
    caveat_ids: set[str] = set()
    if isinstance(value, dict):
        caveat_id = value.get("caveat_id")
        if isinstance(caveat_id, str) and caveat_id:
            caveat_ids.add(caveat_id)
        for child in value.values():
            caveat_ids.update(_walk_caveat_ids(child))
    elif isinstance(value, list):
        for child in value:
            caveat_ids.update(_walk_caveat_ids(child))
    return caveat_ids


def report_caveat_ids(root: Path) -> set[str]:
    """Caveat ids emitted by any generated report under ``benchmarks/parity``.

    Any ``*.json`` living under a ``reports`` or ``reports_public``
    directory counts (the union the strict matcher needs; supersedes the
    narrower ``**/reports/*.json`` glob the structural test used).
    """
    caveat_ids: set[str] = set()
    for path in (root / "benchmarks" / "parity").glob("**/*.json"):
        if not ({"reports", "reports_public"} & set(path.parts)):
            continue
        caveat_ids.update(_walk_caveat_ids(json.loads(path.read_text())))
    return caveat_ids


def _load_manifest(root: Path) -> dict:
    return json.loads(manifest_path(root).read_text())


def declared_caveat_ids(root: Path) -> set[str]:
    """Caveats a manifest artifact commits to (its ``caveats`` list)."""
    caveat_ids: set[str] = set()
    for item in _load_manifest(root)["artifacts"]:
        for caveat in item.get("caveats", []):
            if isinstance(caveat, str) and caveat:
                caveat_ids.add(caveat)
    return caveat_ids


def no_report_caveat_entries(root: Path) -> list[dict]:
    """The explicit no-report caveat class from the manifest.

    Each entry documents a caveat that is owned and exit-criteria'd in
    CAVEATS.md but has no benchmark report *by design* (the typed path
    raises, so no workflow can honestly emit it). The entry must name a
    ``caveat_id``, an ``owner`` bead, and a ``reason``.
    """
    raw = _load_manifest(root).get("no_report_caveats", [])
    if not isinstance(raw, list):
        raise AssertionError(
            "manifest 'no_report_caveats' must be a list of "
            "{caveat_id, owner, reason} objects"
        )
    return [entry for entry in raw if isinstance(entry, dict)]


def no_report_caveat_ids(root: Path) -> set[str]:
    return {
        str(entry["caveat_id"])
        for entry in no_report_caveat_entries(root)
        if entry.get("caveat_id")
    }


def live_caveat_ids(root: Path) -> set[str]:
    """A caveat is live if a manifest artifact declares it, a generated
    report emits it, or it is explicitly classed no-report by design.

    The third term is what makes a documented, owned, report-less
    not-implemented caveat green under the strict matcher without
    weakening drift detection for un-allow-listed extras.
    """
    return (
        declared_caveat_ids(root)
        | report_caveat_ids(root)
        | no_report_caveat_ids(root)
    )


# A bead id is ``bd-`` followed by a 26-char Crockford Base32 ULID body.
BEAD_ID_PATTERN = re.compile(r"^bd-[0-9A-HJKMNP-TV-Z]{26}$")
