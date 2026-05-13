"""Runtime probe of fmrimod's top-level public API surface.

Emits ``docs/contracts/api_inventory_v1.json`` — a per-name manifest of
every name in ``fmrimod.__all__`` with its owner module, runtime
signature, and soundness flags. Pairs with
``docs/contracts/public_api_policy_v1.md`` as the review target for
namespace audits.

Run from the repo root::

    python scripts/api_inventory.py        # writes docs/contracts/api_inventory_v1.json
    python scripts/api_inventory.py --check  # exit non-zero if the inventory is stale

Tier and compatibility-status columns are emitted as ``"review_pending"``;
they are deliberately not assigned by this script. Tier assignment is
the next slice (under the namespace-audit bead) and should land as a
visible diff against the JSON, not as a quiet rename.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "docs" / "contracts" / "api_inventory_v1.json"
SCHEMA_VERSION = "api_inventory/v1"


def _classify(obj: Any) -> Dict[str, Any]:
    """Return signature / soundness metadata for one public name."""
    record: Dict[str, Any] = {
        "is_callable": callable(obj),
        "is_class": inspect.isclass(obj),
        "is_function": inspect.isfunction(obj),
        "owner_module": getattr(obj, "__module__", None),
        "runtime_signature": None,
        "signature_source": "unavailable",
        "opaque_forwarder": False,
        "has_any_in_signature": False,
        "has_untagged_kwargs_dict": False,
    }
    if not record["is_callable"]:
        record["signature_source"] = "non_callable"
        return record

    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        record["signature_source"] = "introspection_failed"
        return record

    record["signature_source"] = "inspect.signature"
    sig_str = str(sig)
    record["runtime_signature"] = sig_str
    record["has_any_in_signature"] = "Any" in sig_str

    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            record["opaque_forwarder"] = True
            break
    return record


def build_inventory() -> Dict[str, Any]:
    """Probe ``fmrimod.__all__`` and return the inventory payload."""
    import fmrimod

    declared = list(getattr(fmrimod, "__all__", ()))
    rows: List[Dict[str, Any]] = []
    callable_count = 0
    opaque_count = 0
    any_count = 0
    classes = 0

    for name in declared:
        obj = getattr(fmrimod, name, None)
        info = _classify(obj)
        row: Dict[str, Any] = {
            "name": name,
            "tier": "review_pending",
            "used_by_public_seam_artifact": "review_pending",
            "compatibility_status": "review_pending",
            **info,
        }
        rows.append(row)
        if info["is_callable"]:
            callable_count += 1
        if info["opaque_forwarder"]:
            opaque_count += 1
        if info["has_any_in_signature"]:
            any_count += 1
        if info["is_class"]:
            classes += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "fmrimod_version": getattr(fmrimod, "__version__", "unknown"),
        "counts": {
            "all_names": len(declared),
            "callable": callable_count,
            "classes": classes,
            "opaque_forwarder": opaque_count,
            "any_in_signature": any_count,
        },
        "rows": sorted(rows, key=lambda r: r["name"]),
    }


def _format_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the on-disk inventory differs from a fresh probe.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=INVENTORY_PATH,
        help="Output path (default: docs/contracts/api_inventory_v1.json).",
    )
    args = parser.parse_args()

    payload = build_inventory()
    rendered = _format_json(payload)

    if args.check:
        if not args.out.exists():
            print(f"inventory missing: {args.out}", file=sys.stderr)
            return 2
        on_disk = args.out.read_text()
        if on_disk != rendered:
            print(
                f"inventory at {args.out} is stale relative to the live probe.\n"
                "Regenerate with: python scripts/api_inventory.py",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    counts = payload["counts"]
    print(
        f"wrote {args.out}: __all__={counts['all_names']}, "
        f"callable={counts['callable']}, "
        f"opaque_forwarder={counts['opaque_forwarder']}, "
        f"any_in_signature={counts['any_in_signature']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
