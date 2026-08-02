#!/usr/bin/env python
"""Check documentation prose against the checked-in style list.

The deterministic half of the prose pass. A prompt asking for a plainer voice
is a soft constraint that decays over a long document; this does not decay,
and it holds for anything written later by anyone.

It only sees words. Register, structure and whether a paragraph earns its
place are not checkable here -- that is what the reviewing agent is for. This
exists so the reviewer never has to spend attention on the mechanical part,
and so a tell caught once is caught forever.

Usage
-----
    python scripts/prose_style_lint.py docs/index.qmd README.md
    python scripts/prose_style_lint.py --all
    python scripts/prose_style_lint.py --all --format github
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO = Path(__file__).resolve().parent.parent
STYLE_LIST = REPO / "docs" / "contracts" / "prose_style_list.toml"

DEFAULT_TARGETS = [
    REPO / "README.md",
    REPO / "docs" / "index.qmd",
    REPO / "docs" / "get-started.qmd",
    REPO / "docs" / "tutorials",
]

# Below this many occurrences a per-10k rate says more about file length
# than about the writing.
_MIN_RATE_HITS = 3

_FENCED = re.compile(r"```.*?```", re.S)
_INLINE = re.compile(r"`[^`]*`")
_YAML = re.compile(r"\A---\n.*?\n---\n", re.S)
_URLPART = re.compile(r"https?://\S+|\]\([^)]*\)")


def _prose_lines(text: str) -> list[tuple[int, str]]:
    """Blank out code so a banned word inside an identifier is not a finding.

    Substitutions preserve line structure, so reported line numbers stay true
    to the source file.
    """

    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    for pat in (_YAML, _FENCED, _INLINE, _URLPART):
        text = pat.sub(blank, text)
    return list(enumerate(text.splitlines(), start=1))


def _word_count(lines: list[tuple[int, str]]) -> int:
    return sum(len(re.findall(r"[A-Za-z][A-Za-z'-]+", ln)) for _, ln in lines)


class Finding(tuple):
    __slots__ = ()

    def __new__(cls, path, line, col, rule, text, hint):
        return super().__new__(cls, (path, line, col, rule, text, hint))


def check_file(path: Path, cfg: dict) -> list[Finding]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    lines = _prose_lines(raw)
    words = _word_count(lines)
    out: list[Finding] = []

    for phrase, hint in cfg.get("banned", {}).items():
        pat = re.compile(rf"\b{re.escape(phrase)}\b", re.I)
        for lineno, ln in lines:
            for m in pat.finditer(ln):
                out.append(
                    Finding(path, lineno, m.start() + 1, "banned", m.group(0), hint)
                )

    if words:
        for word, cap in cfg.get("rate_cap", {}).items():
            pat = re.compile(rf"\b{re.escape(word)}\b", re.I)
            hits = [(n, m) for n, ln in lines for m in pat.finditer(ln)]
            rate = len(hits) * 10_000 / words
            # A rate is meaningless on one occurrence: a single "canonical" in
            # a 228-word tutorial scores 44/10k and trips a cap of 20, which
            # is an artifact of the short file, not a tic. Require a run of
            # them before the rate is allowed to mean anything.
            if len(hits) >= _MIN_RATE_HITS and rate > float(cap):
                n, m = hits[0]
                out.append(
                    Finding(
                        path,
                        n,
                        m.start() + 1,
                        "rate-cap",
                        word,
                        f"{len(hits)}x = {rate:.0f}/10k words, cap {cap} "
                        f"({words} words in file)",
                    )
                )

    for pattern, hint in cfg.get("patterns", {}).items():
        pat = re.compile(pattern, re.I)
        for lineno, ln in lines:
            for m in pat.finditer(ln):
                out.append(
                    Finding(
                        path, lineno, m.start() + 1, "pattern", m.group(0)[:48], hint
                    )
                )

    cap = cfg.get("limits", {}).get("em_dash_per_10k")
    if cap and words:
        dashes = [(n, m) for n, ln in lines for m in re.finditer("—", ln)]
        rate = len(dashes) * 10_000 / words
        if rate > float(cap):
            n, m = dashes[0]
            out.append(
                Finding(
                    path,
                    n,
                    m.start() + 1,
                    "cadence",
                    "em dash",
                    f"{len(dashes)}x = {rate:.0f}/10k words, cap {cap}",
                )
            )
    return out


def _expand(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            files += sorted(p for pat in ("*.qmd", "*.md") for p in t.rglob(pat))
        elif t.exists():
            files.append(t)
    return [
        f for f in files if not set(f.parts) & {"_site", "_freeze", ".quarto", "vendor"}
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="Check the default doc set.")
    ap.add_argument("--format", choices=("text", "github"), default="text")
    ap.add_argument("--style-list", type=Path, default=STYLE_LIST)
    args = ap.parse_args(argv)

    if not args.style_list.exists():
        print(f"error: style list not found: {args.style_list}", file=sys.stderr)
        return 2
    cfg = tomllib.loads(args.style_list.read_text(encoding="utf-8"))

    targets = args.paths or (DEFAULT_TARGETS if args.all else [])
    if not targets:
        ap.error("give paths, or --all")

    files = _expand([Path(t) for t in targets])
    findings = [f for path in files for f in check_file(path, cfg)]

    for path, line, col, rule, text, hint in findings:
        rel = path.relative_to(REPO) if REPO in path.parents else path
        if args.format == "github":
            print(
                f"::warning file={rel},line={line},col={col}::{rule}: {text} — {hint}"
            )
        else:
            print(f"{rel}:{line}:{col}: {rule}: {text!r} — {hint}")

    print(
        f"\n{len(findings)} finding(s) across {len(files)} file(s).",
        file=sys.stderr,
    )
    return 1 if findings else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
