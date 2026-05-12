#!/usr/bin/env python
"""Create WS01-WS10 GitHub issues from local issue templates."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import List, Sequence, Tuple


TEMPLATES: List[Tuple[str, str]] = [
    ("[WS01] Design matrix construction parity", ".github/ISSUE_TEMPLATE/ws01-design-matrix-construction-parity.md"),
    ("[WS02] Contrast engine parity", ".github/ISSUE_TEMPLATE/ws02-contrast-engine-parity.md"),
    ("[WS03] Variance and df parity", ".github/ISSUE_TEMPLATE/ws03-variance-and-df-parity.md"),
    ("[WS04] Run-combination parity", ".github/ISSUE_TEMPLATE/ws04-run-combination-parity.md"),
    ("[WS05] Censor/sample-mask parity", ".github/ISSUE_TEMPLATE/ws05-censor-sample-mask-parity.md"),
    ("[WS06] LSA/LSS parity and performance", ".github/ISSUE_TEMPLATE/ws06-lsa-lss-parity-performance.md"),
    ("[WS07] Rank-deficient design behavior", ".github/ISSUE_TEMPLATE/ws07-rank-deficient-design-behavior.md"),
    ("[WS08] Numeric precision parity", ".github/ISSUE_TEMPLATE/ws08-numeric-precision-parity.md"),
    ("[WS09] Residual diagnostic parity", ".github/ISSUE_TEMPLATE/ws09-residual-diagnostic-parity.md"),
    ("[WS10] Performance decomposition parity", ".github/ISSUE_TEMPLATE/ws10-performance-decomposition-parity.md"),
]


def _extract_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Strip YAML frontmatter if present.
    if lines and lines[0].strip() == "---":
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is not None:
            lines = lines[end + 1 :]
    return "\n".join(lines).strip() + "\n"


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create WS01-WS10 GitHub issues from issue templates."
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="GitHub repository slug (e.g., owner/repo).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create issues (default is dry-run).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]

    commands = []
    for title, rel in TEMPLATES:
        template_path = repo_root / rel
        if not template_path.exists():
            raise FileNotFoundError(f"Missing template: {template_path}")
        body = _extract_body(template_path)
        commands.append((title, body))

    if not args.apply:
        print("Dry run. Use --apply to create issues.")
        for idx, (title, _) in enumerate(commands, start=1):
            print(f"{idx:02d}. {title}")
        return 0

    for title, body in commands:
        proc = _run(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                args.repo,
                "--title",
                title,
                "--body",
                body,
            ]
        )
        if proc.returncode != 0:
            print(f"FAILED: {title}\n{proc.stderr.strip()}")
            return proc.returncode
        print(proc.stdout.strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
