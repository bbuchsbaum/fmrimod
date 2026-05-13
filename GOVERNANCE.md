# Governance

How fmrimod stays globally aligned — stylistically, architecturally,
and statistically — while the codebase is brought, module by module,
into the shape that [`VISION.md`](VISION.md) and [`MISSION.md`](MISSION.md)
commit us to.

This file is the operating manual for *cross-module coherence*. It does
not duplicate the per-module style guidance encoded in tooling (ruff,
mypy, pyproject) or the per-skill refactor mechanics
(`/types-end-to-end`). It defines the **loop** that keeps those local
disciplines from drifting apart across `hrf/`, `regressor/`, `events/`,
`design/`, `glm/`, `stats/`, `group/`, `single/`, `ar/`, `dataset/`, and
the cross-cutting glue. If a future audit skill is added, name it here
only after the skill exists in this checkout.

## 1. Why a governance loop

The MISSION makes four baseline non-negotiables (§ "The baseline that
makes agility compound"):

1. Types end-to-end, with no string-keyed escape hatches.
2. Composition over monolithic fit functions
   (`fmri_dataset → fmri_lm → contrast → group_fit`).
3. neuroim-python as the only data substrate.
4. The four-stage seam is load-bearing, not convenient.

These are easy to honor inside a fresh module and hard to honor *across*
modules over time. The failure mode we are guarding against is the one
every typed library eventually inherits: each subpackage is individually
clean, but the **seams** between them grow strings, dicts, `Any`s, and
ad-hoc adapters that slowly turn the system back into a stateful
pipeline. The governance loop exists to make that drift detectable,
attributable, and reversible.

Governance is **not**:

- a code-review queue (that is the commit protocol);
- a tactical style guide (that is ruff, mypy, and the SKILL files);
- a parity tribunal (that is `docs/contracts/CAVEATS.md` plus the
  benchmarks); or
- a substitute for ownership (every invariant has an owning bead).

Governance is the rhythm that keeps those four things pointed at the
same target.

## 2. Cross-module invariants

These are the rules every module must hold *together*. Module-local
style rules live in the modules. The invariants below cross seams and
must not drift module to module.

### G1 — Typed seam, no string-keyed escape hatches

- Closed-set string parameters at module boundaries are `Literal` types
  or sealed dataclass hierarchies, never bare `str` validated by
  membership checks.
- Cross-module options travel as frozen dataclasses with
  `__post_init__` validation. No `Mapping[str, Any]` or `**kwargs`
  crossing a public seam.
- Public return values have a typed schema (frozen dataclass or
  `NDArray[<dtype>]`), never `dict[str, Any]` or `Any`.
- Alias normalization for legacy R names lives in a single normalizer
  per area, modeled on `fmrimod/stats/normalize.py`. Retired names
  raise an explicit `ValueError` with guidance; they are not silently
  rewritten.

Initial pattern references, not blanket T0 certifications until the
scoreboard exists:

- `fmrimod/stats/interfaces.py` plus `fmrimod/stats/normalize.py` for
  the Literal-plus-normalizer shape.
- `fmrimod/glm/engine.py` for frozen option dataclasses.
- `fmrimod/model/config.py` for constructor-time validation, while its
  remaining mutability is treated as transitional rather than exemplary.
- `fmrimod/group/dtypes.py` for array-boundary coercion.

### G2 — Composable seam, no monolithic entry points

- The four-stage seam `dataset → model/glm → contrast/stats → group`
  is the production pathway. Convenience wrappers may exist; they are
  thin façades over the seam, not parallel implementations.
- No subpackage owns a `do_everything()` that bypasses the
  intermediate typed objects.
- Each stage is independently testable, individually inspectable, and
  serializable. If a refactor makes any stage harder to instantiate in
  isolation, that is a regression.

### G3 — neuroim-python as the only substrate

- Modeling code does not import `nibabel` directly. The I/O boundary
  (`fmrimod/dataset/adapters/`, `fmrimod/io/`) is the only place that
  may.
- Volumes, surfaces, masks, and contrast outputs round-trip through
  neuroim's typed containers.
- When neuroim-python is the friction, the fix lives in neuroim-python
  (or in a documented adapter), not in a bypass.

### G4 — Statistical objects are explicit, not "effect-shaped"

Per VISION.md: parity does not mean copying an upstream library's
vocabulary. A multi-row F contrast has raw effects, an F statistic,
degrees of freedom, and p-values as distinct named quantities. Modules
do not invent generic "effect size" outputs to paper over the
distinction. The names of statistical quantities are part of the API.

### G5 — Parity is contractual

- Every named divergence from an external reference (Nilearn, FitLins,
  SPM, the R sources) appears in
  [`docs/contracts/CAVEATS.md`](docs/contracts/CAVEATS.md) with an
  affected quantity, owning mote bead, and exit criterion.
- New caveats land in the same commit that introduces the divergence;
  `cross_testing/test_caveats_index.py` enforces this.
- Caveats decay; they do not accumulate. The retirement bead is open
  until the caveat is gone.

### G6 — Reproducibility is operational

- Designs and fits carry the seed, the solver path, the HRF
  normalization mode, the AR configuration, the masking mode, and the
  fmrimod version that produced them.
- These fields are part of the typed result objects, not loose
  attributes attached after the fact.

### G7 — Public-API stability

- Public surfaces named in
  [`docs/contracts/public_api_policy_v1.md`](docs/contracts/public_api_policy_v1.md)
  and the active per-package inventories (currently
  [`trio_api_inventory_v1.md`](docs/contracts/trio_api_inventory_v1.md)
  for the `fmrihrf`/`fmridesign`/`fmrireg` lineage; per-package files
  under `docs/contracts/` for the other four R siblings) cannot change
  shape without updating the policy, the affected inventory, migration
  docs, and tests in the same commit.
- The four-stage seam (`fmri_dataset → fmri_lm → contrast →
  group_fit`) is part of the stable surface; changes to it
  additionally require a decision note (§ 8) in the same commit.
- The same rule applies to any future port-stability contract added to
  `docs/contracts/`.

## 3. Module readiness tiers

Every subpackage in [`fmrimod/`](fmrimod/) carries a readiness tier
against the invariants. Tier is a property of the module right now, not
a value judgment. The current tier per module is recorded in
`docs/contracts/alignment_scoreboard.md` once the first governance pass
creates it (see § 7). Until that file exists, no module is officially T0
or T1; references to existing code are pattern references only.

| Tier | Meaning | Requirements |
| --- | --- | --- |
| **T0 — Aligned** | Module holds G1–G6 within itself and at its seams. | Typed public surface, frozen-dataclass configs, Protocol or sealed-dataclass cross-module interfaces, no `Any` in public signatures, no `nibabel` outside the I/O boundary, parity caveats indexed. |
| **T1 — Aligned with caveats** | Holds the invariants except for caveats that are *explicitly written down* with owners and exit criteria. | All T0 requirements plus every divergence indexed in `CAVEATS.md` or in a `docs/contracts/<area>_v*.md` note. |
| **T2 — In flight** | A governance pass is open. Drift has been audited; refactor is planned or partially applied. | An owning bead in `in_progress` and a published audit (chat-pasted or short doc under `docs/contracts/`). May temporarily violate invariants under the audit's documented plan. |
| **T3 — Unaudited** | Module has not had a governance pass yet. | Default tier for any module not promoted. May silently violate invariants. Must not be cited as a reference example. |

Rules:

- **T0 modules are the only modules that may be cited as reference
  examples in skills, docs, or other modules' refactors.** A T2/T3
  module is not a pattern source.
- **Promoting from T3 → T2 is just "open a governance pass."** Anyone
  may. Demoting T0 → T2 is also free — discovering drift is a feature.
- **Promoting T2 → T0/T1 requires evidence:** the audit, the closed
  bead, the test surface, the updated scoreboard entry. See § 5.
- **A module that cannot reach T1 without breaking parity gets a
  decision logged** (§ 8) explaining what blocks it. T3 is fine; T3
  *with no plan* and *no decision* is not.

## 4. Current pilot

The first governed pass is `fmrimod/baseline/` under
`bd-01KRGFGW2JA7RQNV0RR950Z6VQ`, the active fmridesign design/formula
typed-seam epic. The detailed seam contract lives in
[`docs/contracts/fmridesign_design_formula_typed_seam_v1.md`](docs/contracts/fmridesign_design_formula_typed_seam_v1.md).

Baseline is the right pilot because it sits at the design seam and is
already half-refactored: `baseline_model()` exposes Literal annotations
and nuisance diagnostics, but the internal model still carries
string-keyed roles, mutable specs, use-time validation, and `Any` in the
nuisance/block spec surface.

The baseline promotion gate is:

1. Replace role-string term dictionaries with typed baseline roles or a
   typed term bundle.
2. Make `BaselineSpec`, `BaselineTerm`, `NuisanceSpec`, and `BlockSpec`
   frozen where compatibility allows.
3. Move closed-set validation to construction or boundary normalizers.
4. Make block-index semantics explicit and tested.
5. Preserve public wrappers as thin aliases only.
6. Add an API-shape diff and focused tests proving the public seam.
7. Create or update `docs/contracts/alignment_scoreboard.md` before any
   promotion claim.

## 5. The governance loop

The loop is a per-module cycle. Multiple modules can be in different
phases at once.

```
   ┌────────────────────────────────────────────────────────┐
   │                                                        │
   ▼                                                        │
[ Pick ] → [ Audit ] → [ Plan ] → [ Refactor ] → [ Verify ] │
   │           │          │            │             │     │
   │           ▼          ▼            ▼             ▼     │
   │       findings    caveats    one thematic   tests +    │
   │       + tier      + beads    commit each    benchmarks │
   │       update      indexed    step           pass       │
   │                                              │         │
   │                                              ▼         │
   └──────────────────────────────────── [ Record + reprioritize ]
```

### Phase A — Pick

The next governance pass is the highest-leverage of:

1. A module blocking a flagship benchmark from reaching strict-gate
   parity (`./benchmarks/parity/`).
2. A module cited as a "reference example" by a skill but not actually
   at T0.
3. A module that introduced a new caveat in the last cycle.
4. A module that other modules reach into for `Any`-typed values
   (detected by the seam check, § 6, once implemented).
5. The oldest T3 module touched by recent commits.

The picker is whoever opens the bead. Record the picked module in
`docs/contracts/alignment_scoreboard.md` and move it to T2.

### Phase B — Audit

Run the appropriate skill against the module:

- `/types-end-to-end <module>` for G1 (typed seam).
- For cross-file cohesion when the module spans many files, write a
  module-specific audit using the same findings schema below. Do not name
  a skill here unless it exists in this checkout.
- A bespoke audit for G3 (neuroim substrate) — grep for `import
  nibabel` and `np.asarray(...)` patterns that bypass neuroim.

The audit produces, at minimum:

- A list of findings keyed to invariants G1–G7, each with
  **block / lift / note** severity.
- A list of seam violations — places this module returns `Any`, takes
  `Mapping[str, Any]`, or accepts a string where a Literal belongs.
- A list of cross-module callers that would be affected by the
  refactor (so they can be scheduled in the same pass or filed as
  follow-on beads).
- A proposed target tier (T0 or T1) and, if T1, the caveats that
  remain.

The audit lives in chat for small modules and as
`docs/contracts/<module>_governance_audit_v<N>.md` for large ones. The
scoreboard entry links to it.

### Phase C — Plan

Every plan starts with a module refactor brief:

1. **Module and paths** — the exact public surface and private helpers in
   scope.
2. **Owning mote bead** — the epic/child bead that owns the work, plus
   dependencies.
3. **Exemplar imitated** — the existing fmrimod pattern being followed,
   and whether it is a formal T0 example or only a local pattern
   reference.
4. **Typed contract strengthened** — the invariant and concrete
   string/dict/`Any` surface being removed.
5. **Public seam served** — which part of
   `dataset → model/glm → contrast/stats → group` becomes clearer.
6. **Proof** — the test, parity case, or API-shape assertion that will
   fail if the new shape regresses.

After that brief, the plan answers:

1. **Target shape** — the typed primitives that replace the
   string/dict surface.
2. **Alias surface** — the legacy names the public API must still
   accept, where the normalizer lives, what retired names raise.
3. **Parity impact** — which `./benchmarks/parity/` and
   `cross_testing/` cases the refactor touches, and the gate they must
   continue to pass.
4. **Test plan** — added assertions, new tests for the typed surface,
   regression tests for each preserved alias.
5. **Commit shape** — one thematic commit per step in Phase D.

Plans that would introduce a new caveat **always require explicit user
approval** before edits begin. Plans that change a `ported` or
`pythonized` row in
[`trio_api_inventory_v1.md`](docs/contracts/trio_api_inventory_v1.md)
must update the inventory and migration docs in the same commit.

### Phase D — Refactor

Apply changes in the order that keeps the suite green between steps,
mirroring the types-end-to-end skill:

1. Add the new typed primitives alongside the old surface.
2. Migrate internal call sites.
3. Wire the alias normalizer and the retired-name `ValueError`.
4. Migrate tests; keep regression tests for accepted aliases.
5. Delete the old untyped surface and now-dead validation.

One thematic commit per step, each with a `Co-Authored-By: …` trailer
when an agent authored it. Commit bodies cite the invariant they
advance (e.g., "advances G1 in `fmrimod/single/`").

### Phase E — Verify

Required green before promotion:

- `python3.9 -m pytest tests/ -k "not rpy2"` — full Python suite.
- Any `./benchmarks/parity/<case>` whose inputs this module touches —
  strict-gate pass, no new `caveat-bypassed:` rows.
- Any `cross_testing/` case named in the plan.
- For modules that cross the I/O boundary: a smoke test that round-trips
  through neuroim-python without falling back to `nibabel`.
- An API-shape diff in the handoff: public signatures changed, aliases
  preserved or retired, new typed primitives added, and tests proving the
  new seam.

### Phase F — Record + reprioritize

- Update `docs/contracts/alignment_scoreboard.md`: new tier, links to
  audit, link to closing bead.
- Close the governance bead (`mote done <id>`).
- If the audit produced **note**-severity findings, file them as beads
  before closing.
- Re-rank Phase A inputs in light of what changed.

## 6. Drift detection

The loop only works if drift is detectable without manually re-auditing
every module. The target state is three lightweight checks that can run
on CI and feed Phase A. Until a check exists in the repository and is
wired into pytest or CI, it is planned governance tooling, not an active
gate.

### 6.1 Seam check (`tools/governance/seam_check.py` — planned)

Walks the public surface of every module declared T0 or T1 in the
scoreboard and asserts:

- No `Any` in the type signatures of names listed in `__all__`.
- No `**kwargs` on documented public functions.
- No parameter typed as bare `str` whose docstring lists a closed set
  of allowed values.
- No public return type of `dict[str, Any]` or `Mapping[str, Any]`.

A failure demotes the module to T2 in the next governance pass; it does
not block CI.

### 6.2 Substrate check (`tools/governance/substrate_check.py` — planned)

Greps modeling subpackages for `import nibabel` and similar bypass
patterns. Allowlists the I/O boundary. Any new bypass outside the
allowlist demotes the offending module to T2.

### 6.3 Caveats integrity (`cross_testing/test_caveats_index.py`)

Already in place. Every structured `caveat_id` in checked-in parity
reports must appear in `CAVEATS.md`. New caveats without an exit
criterion fail this test.

The planned seam and substrate checks are advisory in the scoreboard
sense. Today, the confirmed hard-failing check in this section is G5
caveats integrity. A G7 hard gate must be named by its actual test or
tool when it is added.

## 7. The alignment scoreboard

The single source of truth for "which modules are aligned right now"
lives at `docs/contracts/alignment_scoreboard.md`. The first governance
pass creates it. Before that file exists, no code path may be cited as a
formal T0/T1 reference example.

Schema:

```markdown
| Module | Tier | Last pass | Audit | Open beads | Caveats | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fmrimod/<module>/` | T2 | 2026-MM-DD | [audit](<audit-link>) | `bd-…` | `<caveat-id>` or — | short status note |
```

Updating the scoreboard is part of Phase F. The scoreboard is also the
*only* place where a module is named a "reference example" — skills and
docs that cite an example must point at a T0 row, not an arbitrary
file.

## 8. Decision log

Architectural decisions that resolve a tension between invariants — or
that explicitly accept a long-lived caveat — are recorded as short
notes under `docs/contracts/decisions/<NNNN>-<slug>.md`. Format:

```markdown
# <Title>

Status: Accepted | Superseded by <NNNN> | Reverted
Date: 2026-MM-DD
Invariants touched: G1, G5

## Context
What forced the decision.

## Decision
What we chose.

## Consequences
What this enables, what it costs, and the exit criterion if the
decision is provisional.
```

A decision note is required when:

- A module cannot reach T1 without breaking parity and we accept the
  caveat instead.
- An invariant is intentionally relaxed for a named module (with the
  scope of the relaxation written down).
- A cross-module refactor changes the four-stage seam.

Decisions are immutable once accepted; they are superseded by a later
note, never edited in place.

## 9. Cross-module changes

A change is **cross-module** when it touches more than one of the
seven lineages in [`AGENTS.md`](AGENTS.md) § Module Map. Cross-module
changes follow extra rules:

- The plan in Phase C names every affected module and lists which
  ones the commit migrates fully vs. which are scheduled as follow-on
  beads.
- The commit body lists the invariants it advances and, if any, the
  invariants it temporarily violates in intermediate modules (with the
  bead that closes the violation).
- A reservation is taken via `mote begin <id> --paths …` for every
  module touched, narrowed to the specific files.
- If the four-stage seam itself changes shape, a decision note (§ 8)
  is part of the same commit.

Cross-module changes that would touch four or more subpackages should
be decomposed before they are applied. If decomposition is impossible
(e.g., a seam rename), the plan must say so and the user must approve
explicitly.

## 10. Roles

This is a small project. "Roles" here are responsibilities, not
people:

- **Caller** — opens the governance bead, picks the module, runs the
  audit. Anyone (human or agent).
- **Refactorer** — produces the plan, applies the thematic commits,
  updates tests and docs. Often the same as the caller.
- **Verifier** — runs the strict-gate parity tests, confirms the
  scoreboard update, signs off on tier promotion. For material
  promotions, verification is a separate review pass from the refactorer
  when feasible: a fresh agent context, explicit user review, or a later
  audit turn. Do not block ordinary local cleanup on a named verifier role
  that is not configured in this checkout.
- **Steward** — the user. Approves new caveats, approves
  cross-module changes ≥ 4 subpackages, approves decisions that
  supersede earlier ones. The steward never auto-approves their own
  decision in the same session.

## 11. Cadence

There is no fixed schedule. The loop is event-driven:

- **A new R port lands** → governance pass on the receiving module
  before its surface is frozen.
- **A flagship benchmark misses the strict gate** → governance pass on
  the modules implicated by the parity report.
- **A drift check (§ 6) demotes a T0/T1 module** → governance pass
  before the next thematic commit in that area.
- **A skill, doc, or migration guide cites a reference example that is
  no longer T0** → pass on the cited module or update the citation.

The scoreboard is the heartbeat. Any month where the scoreboard does
not move is a signal that either (a) nothing changed, or (b) drift is
accumulating silently — and the next governance pass should investigate
which.

## 12. How this file evolves

- **Invariants (G1–G7) are stable.** They restate MISSION.md
  commitments. They change only when MISSION.md changes.
- **The loop (§ 5) and drift checks (§ 6) are tunable.** They evolve
  as the tooling improves. Changes here are normal thematic commits.
- **The scoreboard schema (§ 7) is data; the file is updated every
  Phase F.**
- **Decisions (§ 8) are append-only.** Never edited in place.

If the loop itself stops being useful — too heavy, too light, wrong
inputs — open a decision note proposing the change. Governance of
governance is also governance.

---

See also: [`MISSION.md`](MISSION.md) for the commitments this file
operationalizes, [`VISION.md`](VISION.md) for the world-state argument,
[`AGENTS.md`](AGENTS.md) for the day-to-day mote and commit workflow,
[`message_board.md`](message_board.md) for board-to-bead conversion,
[`docs/contracts/CAVEATS.md`](docs/contracts/CAVEATS.md) for the active parity
divergences, and the `/types-end-to-end` skill for the canonical G1 refactor
mechanics.
