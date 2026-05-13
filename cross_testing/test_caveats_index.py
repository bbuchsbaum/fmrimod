"""Ensure generated parity caveats are indexed with exit criteria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def test_checked_in_report_caveats_are_indexed():
    index = CAVEATS_INDEX.read_text()
    report_ids: set[str] = set()
    for report_path in REPORT_ROOT.glob("**/reports/*.json"):
        report_ids.update(_find_caveat_ids(json.loads(report_path.read_text())))

    assert report_ids, "Expected checked-in parity reports to contain caveats"
    missing = sorted(caveat_id for caveat_id in report_ids if f"`{caveat_id}`" not in index)
    assert missing == []
