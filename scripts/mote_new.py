#!/usr/bin/env python3
"""Source-gated wrapper for `mote new`.

Every new bead should carry one provenance source:

- --board <topic/post-id> for board-routed work
- --from-bead <bd-id> for follow-up work discovered under an existing bead
- --no-board "<reason>" for direct or mechanical capture that should not
  require a board post

The wrapper intentionally delegates creation to `mote new`; it only enforces
the local convention and prepends the provenance block to the bead body.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from collections.abc import Sequence


def _non_empty(value: str) -> str:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("must not be empty")
    return text


def _priority(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("priority must be an integer 0..3") from exc
    if parsed < 0 or parsed > 3:
        raise argparse.ArgumentTypeError("priority must be in 0..3")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a mote bead with required source provenance.",
        epilog=(
            "Examples:\n"
            "  python scripts/mote_new.py \"Implement X\" -p 1 --board work-requests/post-...\n"
            "  python scripts/mote_new.py \"Follow up Y\" -p 2 --from-bead bd-...\n"
            "  python scripts/mote_new.py \"Mechanical cleanup\" --no-board \"direct user request\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("title", type=_non_empty, help="Bead title")
    parser.add_argument(
        "-p",
        "--priority",
        type=_priority,
        default=2,
        help="Priority 0..3, where 0 is highest (default: 2)",
    )
    parser.add_argument("--body", default="", help="Additional bead body after provenance")
    parser.add_argument("--tag", action="append", default=[], help="Tag to add; repeatable")
    parser.add_argument(
        "--dep",
        action="append",
        default=[],
        help="Parent dependency to add; repeatable",
    )
    parser.add_argument("--assignee", type=_non_empty, help="Initial assignee")
    parser.add_argument("--actor", type=_non_empty, help="Override mote actor")
    parser.add_argument("--store", type=_non_empty, help="Override mote store path")
    parser.add_argument("--id", type=_non_empty, help="Exact id for migration/import workflows")
    parser.add_argument("--json", action="store_true", help="Ask mote for JSON output")
    parser.add_argument("--quiet", action="store_true", help="Suppress mote non-essential stderr")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the delegated mote command instead of running it",
    )

    source = parser.add_argument_group("required source provenance")
    source.add_argument(
        "--board",
        type=_non_empty,
        help="Board source, preferably <topic>/<post-id>",
    )
    source.add_argument(
        "--from-bead",
        dest="from_bead",
        type=_non_empty,
        help="Existing bead that produced this follow-up",
    )
    source.add_argument(
        "--no-board",
        dest="no_board",
        type=_non_empty,
        help="Reason this bead has no board or parent-bead source",
    )
    return parser


def _validate_sources(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    source_count = sum(
        source is not None
        for source in (args.board, args.from_bead, args.no_board)
    )
    if source_count != 1:
        parser.error(
            "new beads require exactly one source: --board <topic/post-id>, "
            "--from-bead <bd-id>, or --no-board \"<reason>\""
        )
    if args.from_bead is not None and not args.from_bead.startswith("bd-"):
        parser.error("--from-bead should be a mote bead id like bd-...")


def _provenance_body(args: argparse.Namespace) -> str:
    if args.board is not None:
        source_line = f"Board source: {args.board}"
    elif args.from_bead is not None:
        source_line = f"Bead source: {args.from_bead}"
    else:
        source_line = f"No board source: {args.no_board}"

    provenance = f"{source_line}\nCreated via: scripts/mote_new.py"
    body = args.body.strip("\n")
    if body:
        return f"{provenance}\n\n{body}"
    return provenance


def _mote_command(args: argparse.Namespace) -> list[str]:
    command = [
        "mote",
        "new",
        "-p",
        str(args.priority),
        "--body",
        _provenance_body(args),
    ]
    if args.actor:
        command.extend(["--actor", args.actor])
    if args.store:
        command.extend(["--store", args.store])
    if args.assignee:
        command.extend(["--assignee", args.assignee])
    if args.id:
        command.extend(["--id", args.id])
    if args.json:
        command.append("--json")
    if args.quiet:
        command.append("--quiet")
    for tag in args.tag:
        command.extend(["--tag", tag])
    for dep in args.dep:
        command.extend(["--dep", dep])
    command.append(args.title)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_sources(args, parser)
    command = _mote_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
