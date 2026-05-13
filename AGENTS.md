# Agent Instructions — fmrimod

This file is the canonical agent-instructions document for this repo. `CLAUDE.md`
is a symlink to it so Claude Code, Codex, and other agents read the same content.
Edit `AGENTS.md` — the symlink keeps `CLAUDE.md` in sync automatically.

## Read First

- [`VISION.md`](VISION.md) — the world-state argument for fmrimod and what
  "done" looks like.
- [`MISSION.md`](MISSION.md) — present-day scope, deliverables, and the
  explicit out-of-scope list.
- [`docs/contracts/CAVEATS.md`](docs/contracts/CAVEATS.md) — every active
  parity caveat with owner and exit criterion.
- [`./benchmarks/`](benchmarks/) — the use-case parity workflows that
  validate the port against the R sources, Nilearn, and FitLins.

## Project Overview

**fmrimod** is the unified Python home of seven sibling R neuroimaging
packages: [`fmrihrf`](https://github.com/bbuchsbaum/fmrihrf),
[`fmridesign`](https://github.com/bbuchsbaum/fmridesign),
[`fmrireg`](https://github.com/bbuchsbaum/fmrireg),
[`fmrilss`](https://github.com/bbuchsbaum/fmrilss) (single-trial),
[`fmriAR`](https://github.com/bbuchsbaum/fmriAR),
[`fmrigds`](https://github.com/bbuchsbaum/fmrigds) (group stats), and
[`fmridataset`](https://github.com/bbuchsbaum/fmridataset). It is not a
mechanical port: the R behavior is the spec, but the Python types,
composition, and ergonomics are redesigned for Python. Numerical and
semantic parity is proven case-by-case in `./benchmarks/` and
`cross_testing/`.

The native group-analysis layer lives under `fmrimod/group/`; the R
`fmrigds` bridge is available only as the `fmrigds-r` oracle/fallback
backend. The data substrate is the dogfooded sibling Python port
[`neuroim-python`](https://github.com/bbuchsbaum/neuroim-python) (port
of `neuroim2`); friction with neuroim-python is fixed in
neuroim-python, not routed around in fmrimod.

## Module Map

Subpackages and modules, grouped by their R-source lineage. The
four-stage seam `dataset → model/glm → contrast/stats → group` runs
left-to-right across the modeling rows.

| Lineage | Domain | Subpackages / modules |
| --- | --- | --- |
| `fmrihrf` | HRFs, regressors, convolution | `hrf/`, `regressor/`, `convolve.py`, `condition_basis.py`, `hrf_dispatch.py`, `hrf_integration.py` |
| `fmridesign` | Event/baseline design, contrasts | `events/`, `design/`, `contrast/`, `baseline/`, `basis/`, `formula/`, `convolve_design.py`, `covariate.py`, `design_colmap.py` |
| `fmrireg` | First-level GLM, robust/lowrank variants | `model/`, `glm/`, `robust/`, `lowrank/`, `residualize.py`, `trialwise.py` |
| `fmrilss` | Single-trial (LSS) estimation | `single/`, `betas/` |
| `fmriAR` | AR modeling, whitening | `ar/`, `backends/` |
| `fmrigds` | Group stats and organization | `group/` (native), `stats/` (dispatch + R bridge) |
| `fmridataset` | Dataset containers | `dataset/` (with `dataset/adapters/` for neuroim, nibabel, BIDS) |

**Cross-cutting / glue.** `bids/` (BIDS Stats Model translator), `io/`,
`simulate/`, `visualization/`, `plotting.py`, `cli.py`, `afni.py`,
`accessors.py`, `dispatch.py`, `extension_registry.py`, `types.py`,
`validate.py`, `sampling.py`, `naming.py`, `utils/`.

**In-flight.** `spec/` (declarative design-spec layer),
`dataset/adapters/neuroim_adapter.py` (neuroim-python adapter). Treat
these as moving targets until they land in a thematic commit.

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
4. **Commit and push** — `git add <files> && git commit && git push`. The
   `.mote/` store is local coordination state and remains ignored by git; the
   portable record is the bead id, relevant board post ids, and a concise
   commit message / final handoff that names what changed.
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
