# CLAUDE.md - Project Instructions

## Project Overview

**fmrimod** is a Python port of R packages (`fmrihrf` + `fmridesign` + `fmrireg`) for fMRI analysis.

## Build & Test

```bash
python3.9 -m pytest tests/ -k "not rpy2"   # Full test suite (~1525 tests)
python3.9 -m pytest tests/ -x              # Stop on first failure
python3.9 -m pip install -e .              # Editable install
```

## Issue Tracking

This project uses **beads** (`bd`/`br`) for issue tracking. See AGENTS.md for full workflow.

```bash
bd ready                  # Find next task (unblocked, not deferred)
bd list --status=open     # List all open issues
bd create --title="title" --priority=1  # Create issue (P0-P4)
bd show <id>              # View issue details
bd close <id>             # Close issue
bd sync --flush-only      # Sync to git
```

## Session Management

When ending a work session, always "land the plane":

1. File issues for remaining work (`bd create`)
2. Run quality gates (`python3.9 -m pytest tests/ -k "not rpy2"`)
3. Update issue statuses (`bd close <id>`)
4. Sync and commit: `bd sync --flush-only && git add . && git commit && git push`
5. Provide handoff context for next session
