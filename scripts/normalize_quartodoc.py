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
        normalized = normalize_attribute_tables(normalize_roles(text))
        if path.name == "index.qmd":
            normalized = normalize_index(normalized)
        normalized = normalize_whitespace(normalized)
        if normalized != text:
            path.write_text(normalized)
    normalize_objects_inventory()


if __name__ == "__main__":
    main()
