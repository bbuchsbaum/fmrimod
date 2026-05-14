"""Schema gate against free-prose ``caveats[]`` in parity reports.

Vision-drift-audit (`vision-drift-audit/post-01KRHX267HQWZKD270K9XD3PNA`,
Promise 3) flagged a real loophole in the caveat discipline: the
canonical caveat-index gate (`cross_testing/test_caveats_index.py`)
walks structured ``caveat_id`` keys, but a report can also emit a
top-level ``caveats`` list of *prose strings* — and those slip
straight past the structured-key check.

This module enforces the schema:

- ``caveats`` (when present) must be a list of dicts; every dict
  must carry a ``caveat_id`` whose value matches a row in
  ``docs/contracts/CAVEATS.md`` (the cross_testing index check
  enforces the latter half).
- Free-prose strings inside ``caveats`` are forbidden — they must
  move to a structured ``scope_limitations`` list with quantity /
  omitted_measure / reason / owner_or_followup columns.
- ``scope_limitations`` (when present) must be a list of dicts with
  the four required columns above. The reason a measure isn't being
  recomputed isn't a parity caveat (no divergence to retire); it's
  a scope statement, and the report should record it as such.

Refs: bd-01KRHXEJ0S3NV7FSS2CKMETAMM
(Board source: vision-drift-audit/post-01KRHX267HQWZKD270K9XD3PNA,
action 3 from the constructive routing reply).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_GLOB = "benchmarks/parity/**/reports/*.json"

_REQUIRED_SCOPE_LIMITATION_KEYS = frozenset({
    "quantity",
    "omitted_measure",
    "reason",
    "owner_or_followup",
})


def _walk_reports() -> list[tuple[Path, dict]]:
    """Return ``(path, parsed)`` for every JSON file under parity reports."""
    out = []
    for path in sorted((REPO_ROOT).glob(REPORTS_GLOB)):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            out.append((path, payload))
    return out


def test_no_parity_report_emits_prose_caveats() -> None:
    """``caveats`` (when present) must be a list of dicts, not strings.

    Catches the FitLins-style loophole where freeform prose explanations
    of *scope* live in the same field as parity *divergences*. The two
    have different lifecycles (scope is a permanent statement; caveats
    decay) and conflating them lets either type slip past its gate.
    """
    failures: list[str] = []
    for path, payload in _walk_reports():
        cav = payload.get("caveats")
        if cav is None:
            continue
        if not isinstance(cav, list):
            failures.append(
                f"{path.relative_to(REPO_ROOT)}: 'caveats' must be a list, got {type(cav).__name__}"
            )
            continue
        for i, item in enumerate(cav):
            if isinstance(item, str):
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}[{i}]: free-prose string "
                    f"in 'caveats' — move to 'scope_limitations' if it's "
                    f"a scope statement, or convert to a structured "
                    f"caveat_id dict if it's a real divergence."
                )
            elif isinstance(item, dict):
                if "caveat_id" not in item:
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)}[{i}]: dict in "
                        f"'caveats' missing required 'caveat_id' key: {item}"
                    )
            else:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}[{i}]: 'caveats' entry "
                    f"is neither dict nor string: {type(item).__name__}"
                )
    assert not failures, "\n  - " + "\n  - ".join(failures)


def test_scope_limitations_carry_structured_columns() -> None:
    """Every ``scope_limitations[]`` entry must have the four columns.

    Cheap pass disqualified: dropping the prose ``caveats`` entry
    without converting it to a structured scope row. The schema
    forces each row to name what's omitted, why, and where the
    follow-up lives.
    """
    failures: list[str] = []
    for path, payload in _walk_reports():
        sl = payload.get("scope_limitations")
        if sl is None:
            continue
        if not isinstance(sl, list):
            failures.append(
                f"{path.relative_to(REPO_ROOT)}: 'scope_limitations' must "
                f"be a list, got {type(sl).__name__}"
            )
            continue
        for i, item in enumerate(sl):
            if not isinstance(item, dict):
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}[{i}]: 'scope_limitations' "
                    f"entry must be a dict, got {type(item).__name__}"
                )
                continue
            missing = _REQUIRED_SCOPE_LIMITATION_KEYS - set(item)
            if missing:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}[{i}]: scope_limitation "
                    f"missing required columns: {sorted(missing)} "
                    f"(have: {sorted(item)})"
                )
                continue
            for key in _REQUIRED_SCOPE_LIMITATION_KEYS:
                value = item[key]
                if not isinstance(value, str) or not value:
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)}[{i}]: "
                        f"scope_limitation[{key!r}] must be a non-empty "
                        f"string; got {value!r}"
                    )
    assert not failures, "\n  - " + "\n  - ".join(failures)


def test_fitlins_report_carries_structured_scope_limitations() -> None:
    """Anchor test: the smoking-gun report carries the structured rows.

    `vision-drift-audit/post-01KRHX267HQWZKD270K9XD3PNA` cited the
    FitLins CLI derivative report as the canonical example of the
    free-prose-in-caveats loophole. Pin the structured-rows shape so
    the conversion can't silently regress.
    """
    path = (
        REPO_ROOT
        / "benchmarks/parity/tier_b_fitlins_bids/reports"
        / "fitlins_cli_derivative_report.json"
    )
    if not path.exists():
        return  # Optional fixture; skip silently.
    payload = json.loads(path.read_text())
    cav = payload.get("caveats")
    assert cav == [], (
        f"FitLins report's 'caveats' must be empty (real divergences are "
        f"per-delta caveat_id rows); got {cav!r}"
    )
    sl = payload.get("scope_limitations")
    assert isinstance(sl, list) and sl, (
        f"FitLins report must carry structured 'scope_limitations' rows "
        f"explaining the omitted FitLins-only outputs (p, z, report, "
        f"rSquare, log-likelihood); got {sl!r}"
    )
    quantities = {row["quantity"] for row in sl if isinstance(row, dict)}
    expected = {
        "p_value_map",
        "z_value_map",
        "fitlins_html_report",
        "r_squared_map",
        "log_likelihood_map",
    }
    assert expected <= quantities, (
        f"FitLins scope_limitations missing expected quantities: "
        f"{sorted(expected - quantities)}"
    )
