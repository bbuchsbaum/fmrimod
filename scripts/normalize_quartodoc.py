#!/usr/bin/env python3
"""Normalize quartodoc output for stable, professional Quarto rendering."""

import json
import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
REFERENCE_DIR = DOCS_DIR / "reference"
SIDEBAR = REFERENCE_DIR / "_sidebar.yml"
OBJECTS = DOCS_DIR / "objects.json"
ROLE = re.compile(r":(?:class|func|meth|mod):`(?P<label>~?[^`]+)`")
ATTRIBUTE_LINK = re.compile(r"\[(?P<name>[^]]+)]\(#[^)]+\)")
REFERENCE_PAGE = re.compile(r"reference/(?P<name>[A-Za-z0-9_.-]+\.qmd)")
DIRECTIVE = re.compile(r"^\s*\.\. (?P<name>[a-z][a-z-]*)::(?P<arg>.*)$")
GROUP_DATA_ROW = re.compile(
    r"(\| \[dataset\.group_data\.group_data_from_fmrilm]\([^)]+\) \|)\s*\|"
)


def normalize_roles(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        if label.startswith("~"):
            label = label[1:].rsplit(".", 1)[-1]
        if " <" in label and label.endswith(">"):
            label = label.split(" <", 1)[0]
        return f"`{label}`"

    return ROLE.sub(replace, text)


def normalize_attribute_tables(text: str) -> str:
    lines = text.splitlines()
    in_attributes = False
    for index, line in enumerate(lines):
        if line == "## Attributes":
            in_attributes = True
            continue
        if in_attributes and line.startswith("## "):
            in_attributes = False
        if in_attributes and line.startswith("|"):
            lines[index] = ATTRIBUTE_LINK.sub(
                lambda match: f"`{match.group('name')}`", line
            )
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix


def normalize_index(text: str) -> str:
    if not text.startswith("---\n"):
        text = '---\npagetitle: "API reference"\n---\n\n' + text
    return GROUP_DATA_ROW.sub(
        r"\1 Construct group data from first-level model results. |", text
    )


def normalize_literal_blocks(text: str) -> str:
    """Turn RST ``Label::`` + indented block into ``Label:`` + a code fence.

    quartodoc passes docstring prose through as Markdown, so the RST literal
    marker would otherwise render as a stray ``::`` above an unhighlighted
    indented block.
    """

    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        directive = None if in_fence else DIRECTIVE.match(line)
        if directive and directive.group("name") in ("code", "code-block", "sourcecode"):
            # A code directive is a literal block whose language is its argument.
            line = "::"
            directive = None
        if directive:
            # ``.. deprecated:: 0.3`` + indented prose -> "**Deprecated** (0.3): prose".
            name = directive.group("name").replace("-", " ").capitalize()
            arg = directive.group("arg").strip()
            index += 1
            body: list[str] = []
            while index < len(lines) and (not lines[index].strip() or lines[index].startswith(" ")):
                if lines[index].strip():
                    body.append(lines[index].strip())
                index += 1
            if directive.group("name") == "math":
                out.extend(["$$", *([arg] if arg else []), *body, "$$", ""])
                continue
            head = f"**{name}**" + (f" ({arg})" if arg else "")
            out.extend([head + (": " + " ".join(body) if body else ""), ""])
            continue
        if in_fence or not line.rstrip().endswith("::"):
            out.append(line)
            index += 1
            continue
        label = line.rstrip()[:-2].rstrip()
        if label:
            out.append(label + ":")
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        block: list[str] = []
        while index < len(lines) and (
            not lines[index].strip() or lines[index].startswith("    ")
        ):
            block.append(lines[index][4:])
            index += 1
        while block and not block[-1].strip():
            block.pop()
        if block:
            # Doctest transcripts keep their prompts, so they must not claim to be
            # runnable Python (the copy button would paste ">>>" into a script).
            language = "pycon" if block[0].lstrip().startswith(">>>") else "python"
            out.extend(["", f"```{language}", *block, "```", ""])
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def normalize_whitespace(text: str) -> str:
    """Keep generated Markdown stable under Git's whitespace checks."""

    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def canonical_reference_pages() -> list[Path]:
    """Return only the pages owned by quartodoc's generated sidebar.

    Older qualified-name pages can be present in a developer checkout. They
    are intentionally left untouched: they are not canonical routes and are
    excluded from the Quarto render inventory.
    """

    sidebar = SIDEBAR.read_text()
    names = {match.group("name") for match in REFERENCE_PAGE.finditer(sidebar)}
    return sorted(REFERENCE_DIR / name for name in names)


def normalize_objects_inventory() -> None:
    """Do not advertise attribute fragments quartodoc does not emit."""

    data = json.loads(OBJECTS.read_text())
    items = [item for item in data["items"] if item.get("role") != "attribute"]
    data["items"] = items
    data["count"] = len(items)
    OBJECTS.write_text(json.dumps(data) + "\n")


def main() -> None:
    for path in canonical_reference_pages():
        text = path.read_text()
        normalized = normalize_literal_blocks(
            normalize_attribute_tables(normalize_roles(text))
        )
        if path.name == "index.qmd":
            normalized = normalize_index(normalized)
        normalized = normalize_whitespace(normalized)
        if normalized != text:
            path.write_text(normalized)
    normalize_objects_inventory()


if __name__ == "__main__":
    main()
