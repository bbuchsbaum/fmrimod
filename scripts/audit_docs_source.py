#!/usr/bin/env python3
"""Enforce the neuroim-first contract in executable tutorial source."""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

FENCE_RE = re.compile(r"^\s*```(?P<info>[^`]*)\s*$")
SCHEMATIC_MARKER_RE = re.compile(
    r"^\s*<!--\s*docs-schematic:\s*\S.+?\s*-->\s*$"
)
NONEXECUTED_MARKER_RE = re.compile(
    r"^\s*<!--\s*docs-nonexecuted:\s*\S.+?\s*-->\s*$"
)
CELL_EVAL_FALSE_RE = re.compile(r"(?m)^\s*#\|\s*eval:\s*false\s*$")
PAGE_EVAL_FALSE_RE = re.compile(r"(?m)^\s*eval:\s*false\s*$")

ALLOWED_NILEARN_IMPORTS = {
    "nilearn.datasets": frozenset({"fetch_fiac_first_level"}),
    "nilearn.plotting": frozenset({"plot_design_matrix"}),
}
FORBIDDEN_SPATIAL_CALLS = frozenset(
    {
        "Nifti1Image",
        "NiftiMasker",
        "apply_mask",
        "get_fdata",
        "new_img_like",
        "to_nibabel",
        "unmask",
    }
)


@dataclass(frozen=True)
class CodeBlock:
    """One Python fence and the source location of its opening line."""

    path: Path
    line: int
    info: str
    code: str
    preceding_line: str | None

    @property
    def executable(self) -> bool:
        return self.info.startswith("{python")

    @property
    def plain_python(self) -> bool:
        return self.info == "python"


@dataclass(frozen=True)
class Finding:
    """One source-contract violation."""

    path: Path
    line: int
    rule: str
    message: str

    def format(self, root: Path) -> str:
        try:
            shown = self.path.relative_to(root.parent)
        except ValueError:
            shown = self.path
        return f"{shown}:{self.line}: {self.rule}: {self.message}"


def _previous_nonblank(lines: list[str], opening_index: int) -> str | None:
    for index in range(opening_index - 1, -1, -1):
        if lines[index].strip():
            return lines[index]
    return None


def python_blocks(path: Path) -> list[CodeBlock]:
    """Return executable and plain Python fences from one Quarto file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[CodeBlock] = []
    opening_index: int | None = None
    info = ""

    for index, line in enumerate(lines):
        if opening_index is None:
            match = FENCE_RE.match(line)
            if match is None:
                continue
            candidate = match.group("info").strip()
            if candidate == "python" or candidate.startswith("{python"):
                opening_index = index
                info = candidate
        elif line.strip() == "```":
            code = "\n".join(lines[opening_index + 1 : index]) + "\n"
            blocks.append(
                CodeBlock(
                    path=path,
                    line=opening_index + 1,
                    info=info,
                    code=code,
                    preceding_line=_previous_nonblank(lines, opening_index),
                )
            )
            opening_index = None
            info = ""

    return blocks


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {"np": "numpy", "numpy": "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    aliases[alias.asname or alias.name] = "numpy"
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"numpy.{alias.name}"
    return aliases


def _normalized_call_name(node: ast.AST, aliases: dict[str, str]) -> str:
    name = _call_name(node)
    head, separator, tail = name.partition(".")
    resolved = aliases.get(head)
    if resolved is None:
        return name
    return f"{resolved}.{tail}" if separator else resolved


def _assigned_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(name for child in node.elts for name in _assigned_names(child))
    return ()


def _is_four(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 4


def _shape_like(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and any(
            token in child.id.lower() for token in ("mask", "spatial_shape")
        ):
            return True
        if isinstance(child, ast.Attribute) and child.attr == "shape":
            owner = _call_name(child.value).lower()
            if "mask" in owner or "spatial" in owner:
                return True
    return False


def _finding(block: CodeBlock, node: ast.AST, rule: str, message: str) -> Finding:
    return Finding(
        path=block.path,
        line=block.line + int(getattr(node, "lineno", 1)),
        rule=rule,
        message=message,
    )


def _audit_import(block: CodeBlock, node: ast.Import | ast.ImportFrom) -> list[Finding]:
    findings: list[Finding] = []
    if isinstance(node, ast.Import):
        modules = tuple(alias.name for alias in node.names)
        if any(module == "nibabel" or module.startswith("nibabel.") for module in modules):
            findings.append(
                _finding(
                    block,
                    node,
                    "nibabel-import",
                    "tutorial spatial IO must use neuroim",
                )
            )
        if any(module == "nilearn" or module.startswith("nilearn.") for module in modules):
            findings.append(
                _finding(
                    block,
                    node,
                    "nilearn-import",
                    "only the named FIAC acquisition/display imports are allowed",
                )
            )
        return findings

    module = node.module or ""
    if module == "nibabel" or module.startswith("nibabel."):
        findings.append(
            _finding(
                block,
                node,
                "nibabel-import",
                "tutorial spatial IO must use neuroim",
            )
        )
    if module == "nilearn" or module.startswith("nilearn."):
        imported = frozenset(alias.name for alias in node.names)
        allowed = ALLOWED_NILEARN_IMPORTS.get(module, frozenset())
        is_fiac_page = block.path.name == "real-data-fiac.qmd"
        if not is_fiac_page or not imported or not imported.issubset(allowed):
            findings.append(
                _finding(
                    block,
                    node,
                    "nilearn-import",
                    "only FIAC fetch_fiac_first_level and plot_design_matrix are allowed",
                )
            )
    return findings


def _audit_tree(block: CodeBlock, tree: ast.AST) -> list[Finding]:
    findings: list[Finding] = []
    aliases = _import_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            findings.extend(_audit_import(block, node))

        if isinstance(node, (ast.Name, ast.Attribute)):
            symbol = node.id if isinstance(node, ast.Name) else node.attr
            if symbol == "NiftiMasker":
                findings.append(
                    _finding(
                        block,
                        node,
                        "nilearn-masker",
                        "NiftiMasker bypasses the neuroim dataset adapter",
                    )
                )

        if isinstance(node, ast.Call):
            call_name = _normalized_call_name(node.func, aliases)
            if call_name.rsplit(".", 1)[-1] in FORBIDDEN_SPATIAL_CALLS:
                findings.append(
                    _finding(
                        block,
                        node,
                        "manual-spatial-boundary",
                        f"{call_name} bypasses the neuroim-first tutorial seam",
                    )
                )
            if call_name in {"np.reshape", "numpy.reshape"} or call_name.endswith(
                ".reshape"
            ):
                if any(_shape_like(argument) for argument in node.args):
                    findings.append(
                        _finding(
                            block,
                            node,
                            "manual-reconstruction",
                            "spatial or mask-shaped reconstruction must use neuroim",
                        )
                    )
            for keyword in node.keywords:
                if keyword.arg == "mask" and isinstance(keyword.value, ast.Call):
                    if _normalized_call_name(keyword.value.func, aliases) == "numpy.ones":
                        findings.append(
                            _finding(
                                block,
                                keyword.value,
                                "fabricated-mask",
                                "do not replace a spatial mask with np.ones",
                            )
                        )

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = tuple(name for target in targets for name in _assigned_names(target))
            if (
                isinstance(value, ast.Call)
                and _normalized_call_name(value.func, aliases) == "numpy.eye"
            ):
                if value.args and _is_four(value.args[0]) and any(
                    "affine" in name.lower() for name in names
                ):
                    findings.append(
                        _finding(
                            block,
                            value,
                            "fabricated-affine",
                            "tutorial spatial geometry must come from a neuroim object",
                        )
                    )
    return findings


def audit(root: Path) -> list[Finding]:
    """Audit all tutorial QMD files below ``root``."""

    findings: list[Finding] = []
    for path in sorted(root.rglob("*.qmd")):
        raw = path.read_text(encoding="utf-8")
        front_matter = raw.split("---", 2)[1] if raw.startswith("---") else ""
        if PAGE_EVAL_FALSE_RE.search(front_matter):
            findings.append(
                Finding(
                    path,
                    1,
                    "page-not-executed",
                    "tutorial pages must not disable execution globally",
                )
            )

        for block in python_blocks(path):
            if block.plain_python and not (
                block.preceding_line
                and SCHEMATIC_MARKER_RE.match(block.preceding_line)
            ):
                findings.append(
                    Finding(
                        block.path,
                        block.line,
                        "unlabeled-python-fence",
                        "plain Python must be executable or carry a docs-schematic reason",
                    )
                )
            if CELL_EVAL_FALSE_RE.search(block.code) and not (
                block.preceding_line
                and NONEXECUTED_MARKER_RE.match(block.preceding_line)
            ):
                findings.append(
                    Finding(
                        block.path,
                        block.line,
                        "unlabeled-nonexecution",
                        "eval:false cells require an adjacent docs-nonexecuted reason",
                    )
                )

            try:
                tree = ast.parse(block.code)
            except SyntaxError as exc:
                findings.append(
                    Finding(
                        block.path,
                        block.line + (exc.lineno or 1),
                        "python-syntax",
                        exc.msg,
                    )
                )
                continue
            findings.extend(_audit_tree(block, tree))

    return list(dict.fromkeys(findings))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tutorials", nargs="?", type=Path, default=Path("docs/tutorials"))
    args = parser.parse_args()
    root = args.tutorials.resolve()
    if not root.is_dir():
        raise SystemExit(f"tutorial directory does not exist: {root}")

    paths = sorted(root.rglob("*.qmd"))
    blocks = sum(len(python_blocks(path)) for path in paths)
    findings = audit(root)
    print(
        f"DOCS SOURCE AUDIT: {len(paths)} tutorials, {blocks} Python blocks, "
        f"{len(findings)} defects"
    )
    if findings:
        for finding in findings:
            print(f"- {finding.format(root)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
