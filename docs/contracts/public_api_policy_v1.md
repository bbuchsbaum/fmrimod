# Public API Policy v1

Status: Current API-stability contract for fmrimod

Date: 2026-05-13

## Purpose

This note defines what "stable public API" means for `fmrimod` as a
whole. It is the cross-module stability contract that pairs with the
governance loop in [`GOVERNANCE.md`](../../GOVERNANCE.md) and the
mission-level commitments in [`MISSION.md`](../../MISSION.md) and
[`VISION.md`](../../VISION.md).

It supersedes the earlier `trio_public_api_policy_v1.md`, which scoped
stability to only the `fmrihrf` + `fmridesign` + `fmrireg` port. Under
the current MISSION, `fmrimod` is the unified Python home of **seven**
sibling R packages (`fmrihrf`, `fmridesign`, `fmrireg`, `fmrilss`,
`fmriAR`, `fmrigds`, `fmridataset`), and the stability promise is
expressed in terms of fmrimod's own typed surface — not the union of
R signatures.

## What "stable" means here

A surface is **stable** when:

1. its public shape — name, types, and documented call contract —
   does not change without coordinated updates to the inventory,
   migration docs, tests, and (where relevant) parity reports in the
   same commit;
2. it is reachable through the canonical four-stage seam
   `fmri_dataset → fmri_lm → contrast → group_fit`, or it is a
   documented input or accessor of that seam; and
3. it satisfies the cross-module invariants G1–G7 in
   [`GOVERNANCE.md`](../../GOVERNANCE.md) — types end-to-end,
   composable seam, neuroim-python substrate, explicit statistical
   objects, parity contractuality, operational reproducibility, and
   coordinated public-API change control.

A surface that meets (1) but violates (3) is not stable under this
policy. Drift against the invariants demotes the owning module's
governance tier (`GOVERNANCE.md` § 3) and triggers a pass to bring it
back.

## The stable surface

The stable surface of fmrimod consists of:

- **Top-level names** exported through `fmrimod.__all__`.
- **The four-stage seam.** `fmri_dataset → fmri_lm → contrast →
  group_fit` is *load-bearing*, not convenient. Each stage is an
  independently testable, individually inspectable, serializable typed
  object. Alternative entry points (convenience wrappers, BIDS Stats
  Model translation, CLI) must reduce to this seam — they are façades,
  not parallel implementations.
- **Public types** named in the seam: `FmriDataset`, `FmriLm`,
  `FmriLmConfig` (and its option dataclasses), `FitProvenance` (the
  fit-level reproducibility metadata attached to `FmriLm.provenance`,
  operationalizing VISION.md:99-103 with explicit status fields for
  not-yet-wired provenance slots), `ReplayResult`, `ContrastDelta`,
  `ReplayContractError`, `ContrastSpec` and result
  objects, `GroupDataset`, `GroupSpace`, and the canonical reducer set
  under `fmrimod.group`.
- **Package-level compatibility modules** named in the inventory's
  Python-surface column for surfaces inherited from the R sibling
  packages.
- **Documented migration helpers** under
  `docs/source/design/fmrireg_migration.rst`, the HRF/design migration
  docs, and any future per-package migration guide added when a new R
  package is folded in.
- **Benchmark and cross-testing entry points** under `./benchmarks/`
  and `cross_testing/` that are used as parity or performance gates.

A surface that is not in one of these classes is not part of the
stable public API, regardless of whether it has a leading underscore.

## Status vocabulary (R-sibling rows)

Rows in `docs/contracts/trio_api_inventory_v1.md` (and any future
per-package inventory) carry one of four statuses. These describe an
R-sibling-relative *coverage* status; they do **not** define the
correctness or stability gate, which lives in `GOVERNANCE.md` and the
flagship benchmarks.

- **`ported`** — a matching Python API exists and is treated as part
  of the stable fmrimod surface. Concrete test, doc, and (where
  relevant) parity-benchmark anchors are required.
- **`pythonized`** — behavior is represented by an intentionally
  different Python API or naming convention. The stability promise is
  the documented Python behavior and migration rationale, not full R
  dispatch compatibility. R surface compatibility is *not* implied.
- **`scoped_out`** — not intended to become a public Python API.
  Typical reasons: R-only install/CLI helpers, leading-dot R helpers,
  Rcpp implementation details whose behavior is now exposed through
  higher-level Python functions, or niche R surfaces no flagship
  workflow exercises (MISSION § "deprioritizes function-by-function
  coverage").
- **`pending`** — no clear Python API exists yet. `pending` is a
  *current-state* marker. It is **not** a promise that the row will be
  ported: under MISSION, the picker is flagship-workflow demand, not
  exhaustive coverage. Rows may move from `pending` to `scoped_out`
  with documented rationale when no flagship exercises them.

## Change control

Changes that affect a `ported` or `pythonized` inventory row must
update, in the same commit:

- the implementation surface;
- the relevant unit, parity, golden, or benchmark tests;
- `docs/contracts/trio_api_inventory_v1.md` (or the per-package
  inventory that replaces it);
- migration docs when user-facing behavior changes;
- `docs/contracts/CAVEATS.md` *and* the affected
  `./benchmarks/parity/<case>` report when the change introduces or
  retires a parity caveat.

Changes that touch the four-stage seam — `FmriDataset`, `FmriLm`,
contrast assembly, or `group_fit` — additionally require:

- a decision note under `docs/contracts/decisions/` (per
  `GOVERNANCE.md` § 8) explaining the seam impact and any temporary
  invariant relaxations;
- explicit user approval before edits begin, per `GOVERNANCE.md`'s
  cross-module change rules.

Backward-incompatible changes to top-level names or documented call
shapes are release-boundary changes. They are called out explicitly in
the inventory notes and announced in the migration docs.

## Evidence standard

Stability is only as credible as the evidence behind it. For any row
or surface to be cited as stable, the evidence should include at
least one of:

- direct unit tests against the named Python surface;
- R or golden parity tests for numerical behavior;
- cross-testing artifacts for reference-package parity;
- a `./benchmarks/parity/<case>` strict-gate pass (the strongest
  form — flagship workflow under the canonical seam);
- migration documentation that demonstrates the supported Python
  workflow.

Namespace-level evidence is acceptable for small accessors, aliases,
and helper functions when the broader test file exercises the owning
subsystem and the inventory names the exact Python surface.

A surface that has only namespace-level evidence is stable in
*shape* but is not load-bearing for parity. Promotion to a `T0`
governance tier (`GOVERNANCE.md` § 3) requires at least one of the
first four bullets, not just docstring or namespace evidence.

## Runtime API inventory

The Python-side inventory of `fmrimod.__all__` is generated at
[`docs/contracts/api_inventory_v1.json`](api_inventory_v1.json) by
[`scripts/api_inventory.py`](../../scripts/api_inventory.py). It is the
review target for namespace audits and the freshness check that
prevents silent drift in the public surface.

Each row carries:

- `name` — the exported public symbol;
- `owner_module` — the module the object actually lives in
  (resolved via ``__module__``);
- `runtime_signature` — `inspect.signature(...)` string when callable;
- `opaque_forwarder` — true if the signature uses `*args` or `**kwargs`;
- `has_any_in_signature` — true if `Any` appears in the resolved
  signature string;
- `tier`, `used_by_public_seam_artifact`, `compatibility_status` —
  initially `"review_pending"`, assigned by review under the
  namespace-audit bead.

`tests/test_public_api/test_api_inventory.py` keeps the inventory
honest: if a name is added to or removed from `__all__`, or if the
opaque-forwarder count drifts, the test fails until the inventory is
regenerated. This forces every namespace change to land as a visible
diff against the JSON, not just a wider `dir(fmrimod)`.

Regenerate with:

```bash
python scripts/api_inventory.py          # write the inventory
python scripts/api_inventory.py --check  # fail if on-disk is stale
```

Promotion to top-level (a row appears in `fmrimod.__all__` for the
first time) and demotion (a row is removed) are both inventory diffs
that should be justified in the commit body alongside the policy/tier
reasoning.

### Tier vocabulary

Each inventory row carries a `tier` field. Until reviewed, every row
defaults to `"review_pending"`. The discrete tier values, ordered from
most-load-bearing to least, are:

- `spine` — load-bearing on the four-stage seam
  `fmri_dataset → fmri_lm → contrast → group_fit`. Sound public
  surface (no `*args`/`**kwargs` opacity, no `Any`-in-signature, no
  untagged-Union dispatch). Promotion-locked: removing or
  signature-changing a `spine` row requires explicit steward review per
  `GOVERNANCE.md` §10.
- `spine_review` — load-bearing as above, but with a known typing
  debt: opaque `**kwargs`, `Any` in resolved signature, or untagged
  `Union` over many cases. Each `spine_review` row must carry at least
  one of those flags in its inventory metadata; the regression test
  `test_spine_review_names_have_documented_soundness_debt` enforces
  this. The remediation path is recorded in the linked bead;
  reclassification to `spine` happens when the debt is paid.
- `compat` — supported public surface that is not on the spine. Stable
  but does not anchor the flagship workflow. Free to evolve under the
  ordinary signature-change discipline; not promotion-gated by §10.
- `compat_pending_fix` — `compat` with a known soundness debt that has
  a tracking bead but no scheduled fix. Visible to reviewers as
  "stable enough to depend on, but not the recommended shape."
- `runtime_check` — name whose `Any`-in-signature is *intentional*
  (e.g., a runtime predicate like `is_spec(obj: Any) -> bool`,
  analogous to `isinstance`). Exempt from the no-`Any` invariant on
  callable signatures by explicit policy.
- `review_pending` — default for any row not yet reviewed. The
  freshness gate accepts this value; the spine regression test does
  not.

The spine baseline at the time of the first tier assignment is
**13 rows** (11 `spine`, 2 `spine_review`); see
`tests/test_public_api/test_api_inventory.py::test_spine_names_are_tier_assigned`
for the exact pinned set. New spine entries land as additions to that
test's `_EXPECTED_SPINE_TIERS` map in the same commit that promotes
the name to top-level.

### Review-field vocabulary

The inventory also carries two review fields beyond `tier`:

- `used_by_public_seam_artifact` — the first proof artifact or proof
  bundle role that needs the symbol to be part of the user-facing
  seam. `release_1_0_api_spine` marks the symbols pulled into the 1.0
  release receipt's API-spine gate. `review_pending` means no artifact
  or bundle role has been reviewed yet.
- `compatibility_status` — the current public compatibility posture for
  the row. `spine` means the symbol is part of the release-facing seam.
  It does not erase `tier=spine_review` debt; the tier still records
  known signature soundness problems. Other ratified values are
  `compat`, `deprecated`, `hidden`, `internal_forwarder`, and
  `review_pending`.

## Relation to GOVERNANCE.md

| Asks | Answered in |
| --- | --- |
| What does "aligned" mean for a module? | `GOVERNANCE.md` § 2 (invariants G1–G7) |
| Which module is being worked on now? | `GOVERNANCE.md` § 4 (current pilot) and `docs/contracts/alignment_scoreboard.md` once it exists |
| What does "stable public API" mean? | this file |
| Which Python names live at top level today? | `docs/contracts/api_inventory_v1.json` (generated by `scripts/api_inventory.py`) |
| Which R surface is currently mapped to Python? | `docs/contracts/trio_api_inventory_v1.md` (and successor per-package inventories) |
| What divergences from external references are active? | `docs/contracts/CAVEATS.md` |
| What is the long-form completion bar? | `MISSION.md` (flagship benchmarks at strict gate) |

This policy is the *contractual* face of stability. Governance is the
*operational* face. They are deliberately kept as separate documents
so a future tooling refresh of governance does not silently widen or
narrow the stability promise — and vice versa.
