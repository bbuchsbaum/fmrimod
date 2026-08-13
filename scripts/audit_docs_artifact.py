#!/usr/bin/env python3
"""Fail when a rendered documentation artifact has structural defects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

RAW_SPHINX_ROLES = (":class:", ":func:", ":meth:", ":mod:")


@dataclass
class Page:
    ids: set[str] = field(default_factory=set)
    hrefs: list[str] = field(default_factory=list)
    images: list[tuple[str, str | None]] = field(default_factory=list)
    title: str = ""
    _in_title: bool = False


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page = Page()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if identifier := values.get("id"):
            self.page.ids.add(identifier)
        if tag == "a" and (href := values.get("href")):
            self.page.hrefs.append(href)
        if tag == "img":
            self.page.images.append((values.get("src", ""), values.get("alt")))
        if tag == "title":
            self.page._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.page._in_title = False

    def handle_data(self, data: str) -> None:
        if self.page._in_title:
            self.page.title += data


def parse_page(path: Path) -> Page:
    parser = PageParser()
    parser.feed(path.read_text(errors="replace"))
    return parser.page


def local_target(root: Path, source: Path, href: str) -> tuple[Path, str] | None:
    split = urlsplit(href)
    if split.scheme or split.netloc or href.startswith(("mailto:", "javascript:")):
        return None

    raw_path = unquote(split.path)
    if raw_path.startswith("/"):
        target = root / raw_path.lstrip("/")
    elif raw_path:
        target = source.parent / raw_path
    else:
        target = source
    if raw_path.endswith("/"):
        target /= "index.html"
    return target.resolve(), unquote(split.fragment)


def audit(root: Path) -> list[str]:
    failures: list[str] = []
    html_paths = sorted(root.rglob("*.html"))
    pages = {path.resolve(): parse_page(path) for path in html_paths}
    link_count = 0
    image_count = 0

    for path in html_paths:
        resolved = path.resolve()
        page = pages[resolved]
        raw = path.read_text(errors="replace")
        relative = path.relative_to(root)

        if any(role in raw for role in RAW_SPHINX_ROLES):
            failures.append(f"{relative}: raw Sphinx role leaked into HTML")
        if "quarto-document-content" not in page.ids:
            failures.append(f"{relative}: missing main content target")
        if "#quarto-document-content" not in page.hrefs:
            failures.append(f"{relative}: missing skip-to-content link")
        if relative.as_posix() == "reference/index.html" and not page.title.startswith(
            "API reference"
        ):
            failures.append(f"{relative}: browser title is {page.title!r}")

        if relative.parent.as_posix() == "reference" and "." in relative.stem:
            failures.append(f"{relative}: noncanonical qualified API route rendered")

        for src, alt in page.images:
            image_count += 1
            if alt is None:
                failures.append(f"{relative}: image {src!r} has no alt attribute")

        for href in page.hrefs:
            target_info = local_target(root, resolved, href)
            if target_info is None:
                continue
            link_count += 1
            target, fragment = target_info
            try:
                target.relative_to(root.resolve())
            except ValueError:
                failures.append(f"{relative}: local link escapes artifact: {href!r}")
                continue
            if not target.is_file():
                failures.append(f"{relative}: missing local target for {href!r}")
                continue
            if fragment:
                target_page = pages.get(target)
                if target_page is None:
                    failures.append(f"{relative}: fragment targets non-HTML file: {href!r}")
                elif fragment not in target_page.ids:
                    failures.append(f"{relative}: missing fragment target for {href!r}")

    print(
        f"DOCS AUDIT: {len(html_paths)} pages, {link_count} local links, "
        f"{image_count} images, {len(failures)} defects"
    )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    root = args.artifact.resolve()
    if not root.is_dir():
        raise SystemExit(f"artifact directory does not exist: {root}")

    failures = audit(root)
    if failures:
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
