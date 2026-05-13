# Agent Instructions — fmrimod

This file is the canonical agent-instructions document for this repo. `CLAUDE.md`
is a symlink to it so Claude Code, Codex, and other agents read the same content.
Edit `AGENTS.md` — the symlink keeps `CLAUDE.md` in sync automatically.

## Project Overview

**fmrimod** is a Python port of three R packages (`fmrihrf` + `fmridesign` +
`fmrireg`) for fMRI analysis. The native group-analysis layer lives under
`fmrimod/group/`; the R `fmrigds` bridge is available only as the `fmrigds-r`
oracle/fallback backend.

## Build & Test

```bash
python3.9 -m pip install -e .              # Editable install
python3.9 -m pytest tests/ -k "not rpy2"   # Full Python suite (~1525 tests)
python3.9 -m pytest tests/ -x              # Stop on first failure
python3.9 -m pytest cross_testing/ -m parity      # Parity matrix
python3.9 -m pytest cross_testing/ -m benchmark   # Perf trends
```

Twenty-six rpy2 baseline-spline parity tests are pre-existing failures unrelated
to current work; the `-k "not rpy2"` filter skips them.

## Issue Tracking — mote

This project uses [`mote`](https://github.com/Dicklesworthstone/mote) for local,
daemonless issue and coordination tracking. The store lives in `.mote/`; ops are
append-only JSON files under `.mote/ops/`. Beads (`bd`/`br`) is retired here —
do not run `bd` commands against this repo.

The historical `bd-01K…` identifiers in `docs/contracts/CAVEATS.md` and similar
docs are now mote bead IDs (mote keeps the same ULID format), so references in
prose continue to resolve via `mote show <id>`.

### Essential commands

```bash
mote doctor                          # Sanity-check store layout and ops log
mote board                           # Compact status overview
mote actor show                      # Confirm the actor identity in use
mote ready                           # Open beads with no open blockers

mote ls --status open                # All open beads
mote show <bd-id>                    # Full state + history of one bead
mote new "Title" -p <0..4> --tag <area>   # Create a bead (priority 0=critical .. 4=backlog)
mote set <bd-id> status=in_progress       # Patch a scalar field
mote note <bd-id> --kind progress "..."   # Append a progress/decision/blocker note
mote close <bd-id> --reason "..."         # Close (idempotent)
```

### Reservations and coordination

Before editing shared paths, reserve them so other local agents see the lock:

```bash
mote preflight --issue <bd-id> --paths <path> [<path> ...]
mote begin <bd-id> --paths <path> [<path> ...] --note "starting"
mote who-has <path>                  # See which actor holds the reservation
```

Keep reservations narrow (specific files over directories). If `preflight` or
`begin` exits with code 2, inspect the overlap with `mote who-has` before
editing.

### Compound verbs

```bash
mote begin   <bd-id> --paths ... --note ...   # reserve + claim + progress note
mote handoff <bd-id> --to <actor> --note ...  # handoff note + claim transfer
mote done    <bd-id> --note "finished"        # completion note + close + release
```

### Actor identity

The configured actor for this checkout is recorded in `.mote/local/actor`
(currently `claude-typed-api`). Keep a stable actor across a logical work
session — do not invent a new one per turn. Use `mote --actor <name>` to
override for a single command.

## Session Protocol — "Land the plane"

When ending a work session:

1. **Capture remaining work** — `mote new "..." -p <pri>` for anything you
   discovered but did not finish.
2. **Update bead state** — `mote set <id> status=...` or `mote close <id>`
   for everything you progressed.
3. **Run quality gates** — at minimum `python3.9 -m pytest tests/ -k "not rpy2"`
   for code touches; add `cross_testing/` runs for parity/benchmark changes.
4. **Commit and push** — `git add <files> && git commit && git push`. Mote
   ops files under `.mote/ops/` are checked in alongside code; include them in
   the same commit so coordination state ships with the change.
5. **Provide handoff context** — note open beads, pending reservations, and
   any blockers in your final message.

## Commit Conventions

- One thematic change per commit. Title in imperative mood, ~70 chars max.
- Body explains *why* the change is shaped this way and notes the trade-offs
  considered. Reference bead IDs (`bd-01K…`) when a commit closes or advances
  one.
- Trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  when an agent authored or substantially edited the change.

## Parity & Caveats Discipline

- Every active parity caveat must appear in `docs/contracts/CAVEATS.md` with
  an owning mote bead and an explicit exit criterion. New caveats land in the
  same commit that introduces them; `cross_testing/test_caveats_index.py`
  enforces this.
- Performance regressions are tracked, not gated, via
  `benchmarks/performance/parity_trends.py` + `check_regression.py`. Wire CI
  to the latter when a hardware-tagged history file is available.
