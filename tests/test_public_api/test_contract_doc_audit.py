"""Contract-doc audit coverage checks."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs" / "contracts"
AUDIT = CONTRACTS / "contract_doc_audit.md"

_BEAD_PATTERN = re.compile(r"bd-[0-9A-HJKMNP-TV-Z]{26}")
_DISPOSITIONS = {"keep", "bead", "archive", "supersede"}


def _audit_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    seen_separator = False
    for line in AUDIT.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not seen_separator:
            if all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
                seen_separator = True
            continue
        if len(cells) != 5:
            continue
        document = cells[0].strip("`")
        rows[document] = {
            "owner": cells[1],
            "anchor": cells[2],
            "consumer": cells[3],
            "disposition": cells[4],
        }
    return rows


def _linked_paths(cell: str) -> list[str]:
    paths: list[str] = []
    for token in re.findall(r"`([^`]+)`", cell):
        if token.endswith(".py") or token.endswith(".json") or "/" in token:
            paths.append(token)
    return paths


def test_contract_doc_audit_covers_every_top_level_v1_doc() -> None:
    expected = {path.name for path in CONTRACTS.glob("*_v1.md")}
    rows = _audit_rows()

    assert set(rows) == expected


def test_contract_doc_audit_rows_have_owner_anchor_consumer_and_disposition() -> None:
    for document, row in _audit_rows().items():
        assert _BEAD_PATTERN.search(row["owner"]), document
        assert row["consumer"], document
        assert row["disposition"] in _DISPOSITIONS, document

        anchors = _linked_paths(row["anchor"])
        assert anchors, document
        assert any((ROOT / anchor).exists() for anchor in anchors), document
