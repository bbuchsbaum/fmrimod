"""Guard the raw-image-library boundary promised in MISSION.md."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "fmrimod"

NIBABEL_BOUNDARY_ALLOWLIST = {
    Path("fmrimod/dataset/constructors.py"),
    Path("fmrimod/group/dataset.py"),
}
NIBABEL_BOUNDARY_PREFIXES = (
    Path("fmrimod/dataset/adapters"),
    Path("fmrimod/bids"),
    Path("fmrimod/io"),
)


def _is_boundary_path(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    return relative in NIBABEL_BOUNDARY_ALLOWLIST or any(
        relative.is_relative_to(prefix) for prefix in NIBABEL_BOUNDARY_PREFIXES
    )


def _nibabel_import_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    lines: list[int] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "nibabel" or alias.name.startswith("nibabel.")
                for alias in node.names
            ):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "nibabel" or (node.module or "").startswith("nibabel."):
                lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "nibabel"
        ):
            lines.append(node.lineno)

    return lines


def test_no_nibabel_imports_outside_explicit_data_boundaries() -> None:
    violations: list[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if _is_boundary_path(path):
            continue
        for lineno in _nibabel_import_lines(path):
            violations.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}")

    assert violations == []
