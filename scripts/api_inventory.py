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
import ast
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "fmrimod"
INVENTORY_PATH = REPO_ROOT / "docs" / "contracts" / "api_inventory_v1.json"
INTERNAL_AUDIT_PATH = REPO_ROOT / "docs" / "contracts" / "internal_any_audit.json"
SCHEMA_VERSION = "api_inventory/v1"
INTERNAL_SCHEMA_VERSION = "internal_any_audit/v1"

# Default sentinel reprs include process-local memory addresses
# (e.g., ``<object object at 0x7f5609721ed0>``) that vary across machines
# and Python sessions. Mask them so the inventory is reproducible across
# environments and the freshness gate doesn't flap on harmless re-runs.
_SENTINEL_ADDRESS_RE = re.compile(r"at 0x[0-9a-fA-F]+")
_SENTINEL_ADDRESS_PLACEHOLDER = "at 0x_SANITIZED_"

# Some pandas versions stringify return annotations through their
# implementation module (``pandas.core.frame.DataFrame``); others use
# the canonical alias (``pandas.DataFrame``). Collapse to the alias so
# the inventory is portable across pandas versions.
_PANDAS_INNER_PATH_RE = re.compile(r"\bpandas\.core\.[a-z_]+\.")


def _sanitize_signature(sig: str) -> str:
    """Strip process-local / version-dependent noise from a signature string."""
    sig = _SENTINEL_ADDRESS_RE.sub(_SENTINEL_ADDRESS_PLACEHOLDER, sig)
    sig = _PANDAS_INNER_PATH_RE.sub("pandas.", sig)
    return sig


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
    sig_str = _sanitize_signature(str(sig))
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


_PRESERVED_REVIEW_FIELDS = (
    "tier",
    "used_by_public_seam_artifact",
    "compatibility_status",
)


def _load_existing_overlays(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read previous inventory rows so manual review fields survive regen.

    Tier assignment is a separate audit step from runtime introspection;
    regenerating the inventory after a code change must not silently
    overwrite already-assigned ``tier`` (or sibling review fields). New
    names get the default ``"review_pending"``; names removed from
    ``__all__`` drop out naturally.
    """
    if not path.exists():
        return {}
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    overlays: Dict[str, Dict[str, Any]] = {}
    for row in existing.get("rows", []):
        name = row.get("name")
        if not name:
            continue
        overlays[name] = {
            field: row[field]
            for field in _PRESERVED_REVIEW_FIELDS
            if field in row
        }
    return overlays


def build_inventory(overlay_source: Path | None = None) -> Dict[str, Any]:
    """Probe ``fmrimod.__all__`` and return the inventory payload.

    ``overlay_source`` defaults to the canonical ``INVENTORY_PATH``;
    pass ``None`` to ignore on-disk overlays (useful in tests that want
    a clean defaults-only probe). Manual review fields on the existing
    inventory are preserved per name; runtime-introspected fields are
    always re-derived from the live module.
    """
    import fmrimod

    if overlay_source is None:
        overlay_source = INVENTORY_PATH
    overlays = _load_existing_overlays(overlay_source)

    declared = list(getattr(fmrimod, "__all__", ()))
    rows: List[Dict[str, Any]] = []
    callable_count = 0
    opaque_count = 0
    any_count = 0
    classes = 0

    for name in declared:
        obj = getattr(fmrimod, name, None)
        info = _classify(obj)
        existing = overlays.get(name, {})
        row: Dict[str, Any] = {
            "name": name,
            "tier": existing.get("tier", "review_pending"),
            "used_by_public_seam_artifact": existing.get(
                "used_by_public_seam_artifact", "review_pending"
            ),
            "compatibility_status": existing.get(
                "compatibility_status", "review_pending"
            ),
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


def _annotation_has_any(annotation: Any) -> bool:
    """Return True if an AST annotation contains a typing.Any reference.

    Catches ``Any``, ``typing.Any``, and ``Any`` nested inside subscripts /
    unions (``Optional[Any]``, ``Dict[str, Any]``, ``Union[X, Any]``).
    """
    if annotation is None:
        return False
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id == "Any":
            return True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "Any"
            and isinstance(node.value, ast.Name)
            and node.value.id == "typing"
        ):
            return True
    return False


_VALID_SEAM_CLASSES = frozenset({"public", "compat", "adapter", "internal"})


def _load_public_names_from_inventory(inventory_path: Path) -> set[str]:
    """Return the set of names in the public-API inventory.

    Reads :data:`INVENTORY_PATH` directly so the internal audit can
    classify rows as ``"public"`` without importing ``fmrimod``. The
    inventory is the source-of-truth for ``fmrimod.__all__`` membership.
    """
    if not inventory_path.exists():
        return set()
    try:
        payload = json.loads(inventory_path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {row["name"] for row in payload.get("rows", []) if row.get("name")}


def _classify_seam(module: str, qualname: str, public_names: set[str]) -> str:
    """Classify a function definition by its seam tier.

    The four buckets are:

    - ``"adapter"`` — anything under ``fmrimod.dataset.adapters`` (the
      one named boundary path between fmrimod and external image
      libraries; see MISSION.md:205-207 and the substrate import lint).
    - ``"compat"`` — modules ending in ``.compat`` or ``.meta_compat``.
      These are the R-port shim surfaces called out by
      ``compat_retirement_inventory_v1.md``.
    - ``"public"`` — top-level functions whose simple ``qualname``
      matches an entry in ``fmrimod.__all__`` (per the inventory).
    - ``"internal"`` — everything else; the implementation interior
      that should not be a freshness-gate target on its own but does
      count toward burn-down.

    Method definitions inside classes get ``qualname`` like
    ``ClassName.method`` and are routed to ``internal`` regardless of
    whether ``ClassName`` is public.
    """
    if module.startswith("fmrimod.dataset.adapters"):
        return "adapter"
    # Match ``.compat`` and ``.meta_compat`` exactly at module suffix.
    last_segment = module.rsplit(".", 1)[-1]
    if last_segment == "compat" or last_segment.endswith("_compat"):
        return "compat"
    if "." not in qualname and qualname in public_names:
        return "public"
    return "internal"


def _classify_function_node(
    node: Any,  # ast.FunctionDef | ast.AsyncFunctionDef
    module: str,
) -> Dict[str, Any]:
    """Classify a function/method definition for the internal audit."""
    args = node.args
    has_any = False
    has_var_args_any = False
    has_var_kwargs = False

    for arg in args.args + args.kwonlyargs + args.posonlyargs:
        if _annotation_has_any(arg.annotation):
            has_any = True
    if args.vararg is not None:
        if _annotation_has_any(args.vararg.annotation):
            has_var_args_any = True
            has_any = True
    if args.kwarg is not None:
        # **kwargs is the opaque-forwarder shape regardless of annotation.
        has_var_kwargs = True
        if _annotation_has_any(args.kwarg.annotation) or args.kwarg.annotation is None:
            has_any = has_any or args.kwarg.annotation is None or _annotation_has_any(args.kwarg.annotation)

    if _annotation_has_any(node.returns):
        has_any = True

    return {
        "module": module,
        "qualname": node.name,
        "lineno": int(node.lineno),
        "endlineno": int(getattr(node, "end_lineno", node.lineno) or node.lineno),
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "has_any_annotation": has_any,
        "has_var_kwargs": has_var_kwargs,
        "has_var_args_any": has_var_args_any,
    }


def build_internal_audit() -> Dict[str, Any]:
    """Walk every ``fmrimod/**.py`` file and audit Any/`**kwargs` usage.

    Complements :func:`build_inventory` (which audits only ``fmrimod.__all__``)
    by surfacing soundness debt in private/internal modules. The audit is
    AST-based so it doesn't trigger import-time side effects or depend on
    optional runtime dependencies.

    Each row carries a ``seam_class`` value
    (``"public"`` / ``"compat"`` / ``"adapter"`` / ``"internal"``) so
    follow-up burn-down work can prioritize by tier — see
    :func:`_classify_seam` for the rules. ``coercion_exemption`` and
    ``owner_bead`` are placeholders (``null``) for follow-up classification
    of legitimate boundary-coercion functions and per-row remediation
    ownership.
    """
    public_names = _load_public_names_from_inventory(INVENTORY_PATH)
    rows: List[Dict[str, Any]] = []
    files_scanned = 0
    parse_failures = 0
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT)
        module = str(rel).replace("/", ".").removesuffix(".py")
        try:
            tree = ast.parse(path.read_text(), filename=str(rel))
        except SyntaxError:
            parse_failures += 1
            continue
        files_scanned += 1
        # Track qualname stack so methods carry their owning class.
        def _walk(scope_qual: str, parent: Any) -> None:
            for child in ast.iter_child_nodes(parent):
                if isinstance(child, ast.ClassDef):
                    new_scope = f"{scope_qual}{child.name}." if scope_qual else f"{child.name}."
                    _walk(new_scope, child)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    row = _classify_function_node(child, module)
                    qualname = f"{scope_qual}{child.name}"
                    row["qualname"] = qualname
                    row["seam_class"] = _classify_seam(module, qualname, public_names)
                    row["coercion_exemption"] = None
                    row["owner_bead"] = None
                    rows.append(row)
                    # Functions can contain nested functions; recurse for completeness.
                    _walk(f"{scope_qual}{child.name}.", child)
        _walk("", tree)

    rows.sort(key=lambda r: (r["module"], r["lineno"]))
    counts = {
        "files_scanned": files_scanned,
        "parse_failures": parse_failures,
        "total_functions": len(rows),
        "with_any_annotation": sum(1 for r in rows if r["has_any_annotation"]),
        "with_var_kwargs": sum(1 for r in rows if r["has_var_kwargs"]),
        "with_var_args_any": sum(1 for r in rows if r["has_var_args_any"]),
    }
    # Per-seam soundness counts so burn-down can be prioritized by tier.
    by_seam: Dict[str, Dict[str, int]] = {}
    for seam in _VALID_SEAM_CLASSES:
        seam_rows = [r for r in rows if r["seam_class"] == seam]
        by_seam[seam] = {
            "total": len(seam_rows),
            "with_any_annotation": sum(1 for r in seam_rows if r["has_any_annotation"]),
            "with_var_kwargs": sum(1 for r in seam_rows if r["has_var_kwargs"]),
        }
    counts["by_seam_class"] = by_seam
    return {
        "schema_version": INTERNAL_SCHEMA_VERSION,
        "counts": counts,
        "rows": rows,
    }


def _format_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inventory", "internal"),
        default="inventory",
        help=(
            "'inventory' (default) regenerates the public-API inventory from "
            "fmrimod.__all__; 'internal' regenerates the AST-based audit of "
            "Any/**kwargs in non-__all__ modules."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the on-disk artifact differs from a fresh probe.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path. Defaults to docs/contracts/api_inventory_v1.json "
            "for --mode inventory, or docs/contracts/internal_any_audit.json "
            "for --mode internal."
        ),
    )
    args = parser.parse_args()

    if args.mode == "inventory":
        payload = build_inventory()
        default_out = INVENTORY_PATH
        summary_keys = ("all_names", "callable", "opaque_forwarder", "any_in_signature")
    else:
        payload = build_internal_audit()
        default_out = INTERNAL_AUDIT_PATH
        summary_keys = (
            "files_scanned",
            "total_functions",
            "with_any_annotation",
            "with_var_kwargs",
            "with_var_args_any",
        )

    out_path = args.out or default_out
    rendered = _format_json(payload)

    if args.check:
        if not out_path.exists():
            print(f"artifact missing: {out_path}", file=sys.stderr)
            return 2
        on_disk = out_path.read_text()
        if on_disk != rendered:
            print(
                f"{out_path} is stale relative to the live probe.\n"
                f"Regenerate with: python scripts/api_inventory.py --mode {args.mode}",
                file=sys.stderr,
            )
            return 1
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    counts = payload["counts"]
    summary = ", ".join(f"{k}={counts[k]}" for k in summary_keys if k in counts)
    print(f"wrote {out_path}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
