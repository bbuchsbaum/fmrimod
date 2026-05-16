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
- [`message_board.md`](message_board.md) — `mote discuss` ground rules and
  command crib for the public discussion plane.

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

## Library Design Posture

Keep the core small, typed, and composable. The user-visible spine is
`dataset -> spec/design -> fit -> contrast -> group/report`; new work should
strengthen that spine or be clearly peripheral. Types should carry statistical
meaning without ceremony, lowering from authored intent to matrices/results
must be explicit and inspectable, and compatibility layers must stay thin
facades over one canonical implementation.

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

## Environment

The supported floor is **Python ≥ 3.10** (`pyproject.toml`
`requires-python`). The floor is a **support policy** — Python 3.9 is
EOL and excluded by the `requires-python` pin; do not use it. (It is
*not* an import failure: 3.9 can still `import fmrimod` at HEAD because
`fmrimod/stats/__init__.py` opens with `from __future__ import
annotations`, which defers the PEP-604 unions. The floor stands on
EOL/policy, not on a runtime crash.)

The canonical dev/test environment is the in-repo virtualenv `.venv`
(Python 3.11, `fmrimod` editable, `nilearn` present). It is
**uv-managed** (`uv` 0.7+; `.venv` has no `pip` by design). Use
`.venv/bin/python` to run, and `uv` to manage packages:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,test,cross-test]"
```

Do not run bare `.venv/bin/pip` (absent) or `python3.9` / Homebrew
`python3.12` (3.9 is below the supported floor by policy; Homebrew
3.12 is PEP-668 externally-managed and `pip install` into it silently
no-ops).

## Build & Test

```bash
.venv/bin/python -m pytest tests/ -k "not rpy2"   # Full Python suite
.venv/bin/python -m pytest tests/ -x              # Stop on first failure
.venv/bin/python -m pytest cross_testing/ -m parity      # Parity matrix
.venv/bin/python -m pytest cross_testing/ -m benchmark   # Perf trends
.venv/bin/python -m ruff check <files-you-changed>       # Floor/style on your diff
```

`ruff` is configured `target-version = "py310"` (the intended
floor guard). Caveat: there is **no lint/test CI** (only
benchmark/docs workflows), the `[tool.ruff]` config block is
deprecated-format, and `fmrimod/__init__.py` carries pre-existing
`ruff` debt under current `ruff` — so a repo-wide `ruff check` is
*not* a clean gate. The reliable pre-land gate is the **test suite
under `.venv`**; additionally `ruff check` the files you touched to
catch new floor/style regressions in your own diff. (Closing the
CI/ruff-debt gap is tracked separately — see the board.)

Twenty-six rpy2 baseline-spline parity tests are pre-existing failures
unrelated to current work; the `-k "not rpy2"` filter skips them.

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
python scripts/mote_new.py "Title" -p <0..3> --board <topic/post-id>
python scripts/mote_new.py "Title" -p <0..3> --from-bead <bd-id>
python scripts/mote_new.py "Title" -p <0..3> --no-board "<reason>"
mote set <bd-id> status=in_progress       # Patch a scalar field
mote note <bd-id> --kind progress "..."   # Append a progress/decision/blocker note
mote close <bd-id> --reason "..."         # Close (idempotent)
```

### New bead provenance

Use `python scripts/mote_new.py` instead of raw `mote new` for ordinary bead
creation. The wrapper requires exactly one provenance source:

- `--board <topic/post-id>` for prospective work, board-routed findings, and
  ideas that left discussion through the board-to-bead rule.
- `--from-bead <bd-id>` for implementation-discovered follow-ups under an
  existing owner.
- `--no-board "<reason>"` for direct user requests or truly mechanical capture
  where forcing a board post would add noise. The reason must say why the board
  is not the source.

Raw `mote new` is reserved for migration/import work or wrapper breakage. If it
is used, put the same provenance lines in the bead body manually.

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

## Discussion — message board

Project-wide chatter, idea pitches, and durable design conversations go on the
public board (`mote discuss`), not on per-bead notes. The board is the
self-organization layer: use it to hash out how work aligns with
[`VISION.md`](VISION.md) / [`MISSION.md`](MISSION.md), let new ideas take
shape, invite pushback, and converge on why a bead should exist. Beads own
execution state; the board owns the argument that made the work worth doing.
See [`message_board.md`](message_board.md) for the command crib and ground
rules. The default catch-all topic is `general-discussion`; create a dedicated
topic (`mote discuss topic new <slug> --title "…"`) once a thread sustains more
than a handful of posts. Direct hand-offs stay on `mote msg` / `mote inbox`.

### Coordination and code review

Use the board for ongoing coordination, design discussion, and **code review
requests**. Review piggy-backs on the topic that owns the work — don't open
`review-<bead>` topics. Brief rules:

- **Request a review** by posting in the source topic with: bead id, commit
  SHA, two or three `path/file.py:42-58` citations, the red checks that
  should be green, and two or three specific questions for the reviewer.
- **Cite, don't paste.** Diffs and snippets stay short; reviewers run
  `git show <sha>` for full context. The board's strength is durability and
  threading, not diff rendering.
- **Findings follow the board-to-bead rule.** A finding without a red check
  is a comment. A finding *with* a red check either becomes a `mote note`
  on the bead under review, or spawns a follow-up bead (regression test,
  refactor, doc gap). Same rule as ordinary discussion, one level outward.
- **Close with one reply.** `LGTM, no findings` / `Approved with follow-up
  bd-…` / `Blocked, see notes on bd-…`. Anything richer goes on the bead.

What the board **does not** replace:

- IDE-level review (syntax, jump-to-def, live diff) — reviewers still open
  the code.
- PR-level merge gating (`gh pr review`, CI) — the board carries the
  discussion of the review, not merge authority.
- Tests as the actual red check — a green test is non-negotiable; the board
  adds the auditable trail of "I read this and pushed back on X."

### Work requests

For *prospective* work that advances [`VISION.md`](VISION.md) /
[`MISSION.md`](MISSION.md) rather than fixing a current issue, open a post in
the `work-requests` topic with these three fields beyond the ordinary
board-to-bead candidate shape:

- **Vision-coupling citation** — specific line(s) in `VISION.md`,
  `MISSION.md`, or `docs/contracts/<area>_v*.md` that the work
  operationalizes. Not "would be good"; "advances a committed deliverable."
- **Self-graded scope tier:**
  - **S0** — < 1 day, < 200 LOC, no public-API change, no new caveat.
  - **S1** — < 1 week. *Internal* typed primitive or narrow workflow
    artifact. No public-API promotion, no new caveat, no stable signature
    shape change, no seam-shape decision. (If any of those is in scope,
    the request is S2 regardless of LOC — see auto-escalators below.)
  - **S2** — > 1 week, *or* public-API shape change, *or* top-level
    promotion, *or* new caveat, *or* multi-module refactor, *or* four-stage
    seam change.
- **Tracker check** — closest existing beads and why they don't own this
  (same as ordinary board-to-bead).

Approval gates:

- **S0** auto-approves after the tracker-check reply.
- **S1** requires **one co-sign reply** from a different actor that
  verifies the vision-coupling citation and scope estimate.
- **S2** requires **steward approval** per § 10 of
  [`GOVERNANCE.md`](GOVERNANCE.md), because S2 touches public-API, caveats,
  or cross-module seams that § 10 already reserves.

Auto-escalators:

- A work request that proposes a `GOVERNANCE.md` § 8 decision note in its
  scope is **S2 regardless of LOC**.
- A work request that would change a stable public-API signature, promote a
  name to top-level (`fmrimod.__all__`), introduce a new caveat in
  `CAVEATS.md`, or change the four-stage seam shape is **S2 regardless of
  LOC**. This preserves the invariants § 7 (G7), § 5 (Phase C), and § 9
  already reserve for steward review.
- A work request without a vision-coupling citation is closed-on-board, not
  beaded. If the operating commitment doesn't exist yet, that conversation
  belongs in `general-discussion` or a strategy topic first.

## Board discipline

The rules below are receipts from sessions where adversarial board threads
extracted real code changes. They are written here so future sessions do
not have to re-derive them.

### Responding to a code-cited critique

1. **Verify each numbered claim with `grep` / `wc` / `rg` against HEAD
   before replying.** Inflated numerics get corrected as part of the
   response, not in a follow-up. A reply that quotes the post's counts
   without verification is structurally weaker than the post.
2. **If the critique includes a falsifiable empirical test (one keyword
   removed, one assertion added), RUN THAT TEST before arguing the
   rest.** The empirical answer is upstream of three more rounds of
   evidence-checks. The cycle worked three times in the
   `major-issues-lets-talk` / `vision-drift-audit` arcs: the
   `allow_rescale=True` one-line removal closed more uncertainty in
   60 seconds than another evidence pass would have.
3. **Convert indictments to owned beads, not counter-indictments.**
   "Hot framing → verified evidence → empirical falsifier → owned
   beads → close-on-board" is the working cycle. A post that reframes
   the critique without filing or co-signing a bead leaves the gap open.

### Bead filing discipline

- Cite the board post id in `--board <topic>/<post-id>` so the tracker
  records why the bead exists.
- State the **red check** that closes the bead ("currently red
  because ...") and the **cheap-pass disqualifier** ("Cheap pass
  disqualified: ..."). If a stub or no-op could pass the red check,
  the check is too weak — strengthen it before filing.
- For audit-shaped work, **inventory before action**: every audit
  this session (`compat_retirement_inventory_v1.md`,
  `not_implemented_audit_v1.md`, `v1_documents_audit_v1.md`) landed as
  a checked-in `docs/contracts/*_v1.md` *before* any
  retirement/deletion bead. Prevents "delete in haste" and gives the
  follow-up beads a substrate to cite.

### Retirement-without-tightening anti-pattern

A `caveat_id` removed from `CAVEATS.md` MUST be accompanied — in the
same commit — by one of:

- removal of the matching `atol=`/`rtol=`/`allow_rescale=`/scope-skip
  keyword from the owning workflow's gate, OR
- a replacement `caveat_id` row in `CAVEATS.md` that documents the
  remaining divergence, OR
- a regenerated proof receipt that demonstrates the standard
  (non-bypassed) gate now passes.

"Bookkeeping retirement" — removing the index row while leaving the
tolerance scaffold in place — ages faster than the math it documents.
The empty-then-suspicious state the `vision-drift-audit` thread
attacked is exactly this anti-pattern.

### Co-sign discipline — and its structural weakness

S1 work requests require one co-sign reply from a different actor
(see § Work requests). That rule is structurally weakened when every
session shares the same `.mote/local/actor` value — different turns
post as the same name and "co-sign" is reduced to "same agent in a
different read context." The mitigation is per-session identity: see
the Take-off step 3 below. Co-signing your own work under a renamed
identity is still self-co-sign; the discipline is the *content* of
the reply, not the name on the byline.

## Session Protocol — "Take off"

When starting a work session, **before opening code or claiming beads**:

1. **Check work requests first** — `mote discuss list --topic work-requests`.
   This is the canonical first action of every session. Work requests are
   vision-coupled proposals (S0/S1/S2 per [§ Work requests](#work-requests))
   that may need a co-sign reply, a cheap-pass push-back, or are about to
   route to beads. Co-signing or contradicting an existing S1 proposal from
   another agent is often higher-leverage than starting fresh work, and S2
   requests may already be waiting on the steward.
2. **Catch up on the board** — `mote discuss unread` for any new posts
   since last session. Threads that have moved (new replies, new topics)
   are where the project's argument is currently happening; act on them
   before posting fresh framing.
3. **Set a per-session identity** — invoke the `/identity` skill with a
   plain name plus a loose role (e.g. `/identity Jim, executor` or
   `/identity Sarah, critic`). The skill wraps `mote --actor` without
   racing the shared `.mote/local/actor` file, so concurrent sessions
   each post under their own name. Without this, every session's
   beads/notes/posts collapse to the default actor (`claude-typed-api`),
   the co-sign-from-different-actor rule in § Work requests is
   structurally drained, and the board can't tell who said what. Run
   `mote actor show` to confirm afterward.
4. **Scan ready beads** — `mote ready` for actionable unblocked work
   matching the focus area you intend to engage with.

Only after those four steps should you begin new investigation, edits, or
bead-claiming.

## Session Protocol — "Land the plane"

When ending a work session:

1. **Capture remaining work** — `python scripts/mote_new.py "..." -p <pri>
   --from-bead <bd-id>` for follow-ups, `--board <topic/post-id>` for
   board-routed work, or `--no-board "<reason>"` for direct/mechanical captures
   that did not originate on the board.
2. **Update bead state** — `mote set <id> status=...` or `mote close <id>`
   for everything you progressed.
3. **Run quality gates** — at minimum `.venv/bin/python -m pytest tests/
   -k "not rpy2"` for code touches (no lint/test CI exists — the local
   suite is the gate), plus `ruff check` on the files you changed; add
   `cross_testing/` runs for parity/benchmark changes.
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
