"""Audit parity ``rtol`` / ``atol`` call sites across the benchmarks tree.

Vision-drift-audit (`vision-drift-audit/post-01KRHX267HQWZKD270K9XD3PNA`)
flagged ~110 ``rtol``/``atol`` keyword usages across the parity workflows
and cross-testing harness — many of them masking real divergences whose
caveats have since been retired (``allow_rescale=True`` in
``tier_a_spm_auditory`` is the canonical smoking gun).

A flat ``rg -c "rtol="`` count is not a gate. Numerical comparisons need
tolerances; the problem is *unjustified* tolerances. This script walks
``benchmarks/parity/`` and ``cross_testing/`` via AST, captures every
call-site that passes ``rtol=`` or ``atol=``, and classifies each one:

- ``"justified"`` — a comment within 3 lines of the call cites a
  ``caveat_id`` / ``CAVEATS.md`` / a ``numerical_floor`` rationale. The
  tolerance is documented; it is not an escape hatch.
- ``"baseline_default"`` — the value matches a recognized default
  (``rtol=1e-5``, ``atol=1e-8``, etc.). These are functional defaults
  for ``np.testing.assert_allclose`` style assertions and don't need
  per-site rationale.
- ``"unjustified"`` — a non-default literal value with no nearby
  rationale comment. These are the burn-down targets.
- ``"unclassified"`` — non-literal value (variable, expression, etc.)
  or could not be parsed cleanly.

Run from the repo root::

    python scripts/tolerance_audit.py        # writes the audit JSON
    python scripts/tolerance_audit.py --check  # exit non-zero if stale

Refs: bd-01KRHXER0A33DBM403ABW8EY15
(Board source: vision-drift-audit/post-01KRHX267HQWZKD270K9XD3PNA).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "docs" / "contracts" / "parity_tolerance_audit_v1.json"
SCHEMA_VERSION = "parity_tolerance_audit/v1"

# Roots scanned for tolerance call-sites. The substrate-import lint and
# the public-API inventory cover ``fmrimod/`` itself; this audit
# specifically targets the parity / cross-testing surfaces called out
# by the vision-drift audit.
_SCAN_ROOTS = ("benchmarks/parity", "cross_testing")

_TOLERANCE_KWARGS = ("rtol", "atol")

# Comment markers that count as "justified" rationale near the call
# site. Matched case-insensitively as substrings.
_JUSTIFICATION_MARKERS = (
    "caveat_id",
    "caveats.md",
    "numerical_floor",
    "tolerance_rationale",
    "see caveats",
)

# Recognized default tolerance literal values. Anything in this set is
# classified ``baseline_default`` rather than ``unjustified``. The set
# is deliberately narrow — we want unfamiliar literals to surface as
# ``unjustified`` so an audit reader has to look at them.
_BASELINE_DEFAULT_VALUES = frozenset({
    1e-5,    # np.testing.assert_allclose default rtol
    1e-7,    # common float64 round-trip floor
    1e-8,    # np.testing.assert_allclose default atol
    1e-9,    # tighter float64 round-trip floor
    1e-12,   # algebraic-equality floor
    0.0,
    0,
})


def _surrounding_comments(source_lines: List[str], lineno: int, span: int = 3) -> str:
    """Return up to ``span`` lines of context around ``lineno`` joined with newlines."""
    start = max(0, lineno - 1 - span)
    end = min(len(source_lines), lineno - 1 + 1)  # include the call's line itself
    return "\n".join(source_lines[start:end])


def _has_justification(context: str) -> bool:
    lower = context.lower()
    return any(marker in lower for marker in _JUSTIFICATION_MARKERS)


def _literal_value(node: ast.expr) -> Tuple[bool, Optional[float]]:
    """Try to extract a numeric literal from an AST expression.

    Returns ``(True, value)`` if the value is a clean numeric literal
    (or unary-minus on one), ``(False, None)`` otherwise.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return True, float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        ok, inner = _literal_value(node.operand)
        if ok and inner is not None:
            return True, -inner
    return False, None


def _classify_value(value: Optional[float]) -> str:
    if value is None:
        return "unclassified"
    if value in _BASELINE_DEFAULT_VALUES:
        return "baseline_default"
    return "elevated"  # placeholder; refined by context below


def _scan_file(path: Path) -> List[Dict[str, Any]]:
    """Return one row per ``rtol=``/``atol=`` keyword in this file."""
    text = path.read_text()
    source_lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    rel = path.relative_to(REPO_ROOT)
    rows: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in _TOLERANCE_KWARGS:
                continue
            ok, value = _literal_value(kw.value)
            value_repr = ast.unparse(kw.value) if not ok else value
            context = _surrounding_comments(source_lines, kw.value.lineno or node.lineno)
            justified = _has_justification(context)
            literal_class = _classify_value(value if ok else None)

            if justified:
                classification = "justified"
            elif literal_class == "baseline_default":
                classification = "baseline_default"
            elif literal_class == "elevated":
                classification = "unjustified"
            else:
                classification = "unclassified"

            rows.append({
                "file": str(rel),
                "lineno": int(kw.value.lineno or node.lineno),
                "kwarg": kw.arg,
                "value": value if ok else None,
                "value_repr": str(value_repr),
                "classification": classification,
                "context_marker_found": justified,
            })
    return rows


def build_audit() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    files_scanned = 0
    for root_name in _SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files_scanned += 1
            rows.extend(_scan_file(path))

    rows.sort(key=lambda r: (r["file"], r["lineno"], r["kwarg"]))

    counts = {
        "files_scanned": files_scanned,
        "total_call_sites": len(rows),
        "by_classification": {
            klass: sum(1 for r in rows if r["classification"] == klass)
            for klass in ("justified", "baseline_default", "unjustified", "unclassified")
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "counts": counts,
        "rows": rows,
    }


def _format_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the on-disk audit differs from a fresh probe.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=AUDIT_PATH,
        help="Output path (default: docs/contracts/parity_tolerance_audit_v1.json).",
    )
    args = parser.parse_args()

    payload = build_audit()
    rendered = _format_json(payload)

    if args.check:
        if not args.out.exists():
            print(f"audit missing: {args.out}", file=sys.stderr)
            return 2
        on_disk = args.out.read_text()
        if on_disk != rendered:
            print(
                f"{args.out} is stale relative to the live probe.\n"
                f"Regenerate with: python scripts/tolerance_audit.py",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    counts = payload["counts"]
    by = counts["by_classification"]
    print(
        f"wrote {args.out}: total={counts['total_call_sites']}, "
        f"justified={by['justified']}, "
        f"baseline_default={by['baseline_default']}, "
        f"unjustified={by['unjustified']}, "
        f"unclassified={by['unclassified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
