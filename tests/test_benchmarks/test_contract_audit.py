"""Shape tests for the contract-document audit table."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = ROOT / "docs" / "contracts"
AUDIT = CONTRACT_ROOT / "CONTRACT_AUDIT.md"

ALLOWED_DISPOSITIONS = {"keep", "bead", "archive", "supersede"}


def _audit_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    seen_separator = False
    for line in AUDIT.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if seen_separator and rows:
                break
            continue
        cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
        if not seen_separator:
            if all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
                seen_separator = True
            continue
        if len(cells) != 5:
            continue
        contract, owner, red_check, consumer, disposition = cells
        if contract == "Contract":
            continue
        rows[contract] = {
            "owner": owner,
            "red_check": red_check,
            "consumer": consumer,
            "disposition": disposition,
        }
    return rows


def test_contract_audit_covers_every_top_level_v1_contract() -> None:
    expected = {
        str(path.relative_to(ROOT))
        for path in CONTRACT_ROOT.glob("*_v1.md")
        if path.is_file()
    }
    rows = _audit_rows()

    assert set(rows) == expected


def test_contract_audit_rows_have_machine_readable_routing() -> None:
    rows = _audit_rows()
    assert rows

    for contract, row in rows.items():
        assert row["owner"], contract
        assert row["owner"].startswith("bd-"), contract
        assert row["red_check"], contract
        assert row["consumer"], contract
        assert row["disposition"] in ALLOWED_DISPOSITIONS, contract


def test_contract_audit_bead_rows_name_follow_up_owner() -> None:
    rows = _audit_rows()

    bead_rows = {
        contract: row
        for contract, row in rows.items()
        if row["disposition"] == "bead"
    }

    assert bead_rows
    for contract, row in bead_rows.items():
        assert row["owner"].startswith("bd-"), contract


def test_contract_audit_keeps_archive_out_of_scope() -> None:
    rows = _audit_rows()
    archived_contracts = {
        str(path.relative_to(ROOT))
        for path in (CONTRACT_ROOT / "archive").glob("*_v1.md")
    }

    assert archived_contracts
    assert not (set(rows) & archived_contracts)
