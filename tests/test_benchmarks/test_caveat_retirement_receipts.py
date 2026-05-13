"""Receipt tests for retired parity caveats."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CAVEATS = ROOT / "docs" / "contracts" / "CAVEATS.md"
RETIRED_CAVEATS = ROOT / "docs" / "contracts" / "CAVEAT_RETIREMENTS.md"

EXPECTED_RETIRED_IDS = {
    "spm-auditory-hrf-grid-scale",
    "localizer-tstat-variance-outliers",
    "fitlins-ar1-coefficient-binning",
    "second-level-normal-vs-t-pvalues",
}

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")
_BEAD_PATTERN = re.compile(r"^bd-[0-9A-HJKMNP-TV-Z]{26}$")


def _retirement_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    seen_separator = False
    for line in RETIRED_CAVEATS.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in stripped.strip("|").split("|")]
        if not seen_separator:
            if all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
                seen_separator = True
            continue
        if len(cells) != 6:
            continue
        caveat_id, active, owner, commit, report_path, red_check = cells
        rows[caveat_id] = {
            "active": active,
            "owner": owner,
            "commit": commit,
            "report_path": report_path,
            "red_check": red_check,
        }
    return rows


def _walk_caveat_ids(value: object) -> set[str]:
    if isinstance(value, dict):
        found = set()
        caveat_id = value.get("caveat_id")
        if isinstance(caveat_id, str) and caveat_id:
            found.add(caveat_id)
        for child in value.values():
            found.update(_walk_caveat_ids(child))
        return found
    if isinstance(value, list):
        found = set()
        for child in value:
            found.update(_walk_caveat_ids(child))
        return found
    return set()


def test_retirement_receipt_has_expected_rows_and_shape() -> None:
    rows = _retirement_rows()
    assert EXPECTED_RETIRED_IDS <= set(rows)

    for caveat_id in EXPECTED_RETIRED_IDS:
        row = rows[caveat_id]
        assert row["active"] == "false"
        assert _BEAD_PATTERN.match(row["owner"])
        assert _COMMIT_PATTERN.match(row["commit"])
        assert row["red_check"].startswith("python3.9 -m pytest ")
        check_paths = row["red_check"].removeprefix("python3.9 -m pytest ")
        check_paths = check_paths.removesuffix(" -q")
        for check_path in check_paths.split():
            if "::" in check_path:
                check_path = check_path.split("::", maxsplit=1)[0]
            assert (ROOT / check_path).exists()
        assert row["report_path"].endswith(".json")
        assert (ROOT / row["report_path"]).exists()


def test_retirement_commits_exist() -> None:
    for caveat_id, row in _retirement_rows().items():
        subprocess.run(
            ["git", "cat-file", "-e", f"{row['commit']}^{{commit}}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ), caveat_id


def test_retired_caveats_are_not_active_or_emitted() -> None:
    active_index = ACTIVE_CAVEATS.read_text()
    emitted: set[str] = set()
    for report_path in (ROOT / "benchmarks" / "parity").glob("**/reports/*.json"):
        emitted.update(_walk_caveat_ids(json.loads(report_path.read_text())))

    for caveat_id in EXPECTED_RETIRED_IDS:
        assert f"`{caveat_id}`" not in active_index
        assert caveat_id not in emitted
