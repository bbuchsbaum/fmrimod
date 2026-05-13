# fmridesign Design/Formula Typed-Seam Plan v1

Status: Active

Owner: `bd-01KRGFGW2JA7RQNV0RR950Z6VQ`

## Objective

Harden the Python fmridesign port surface so `fmrimod/design`,
`fmrimod/formula`, baseline construction, design metadata, and contrast glue
honor the MISSION "types end-to-end" baseline while preserving the R package as
the behavioral source of truth.

This is not a rewrite of fmridesign into Python syntax. The public formula
styles can remain as migration adapters, but the implementation center must
move toward typed primitives with explicit validation and executable parity
contracts.

## Source of Truth

- R behavior: `/Users/bbuchsbaum/code/fmridesign`.
- Python target: `fmrimod.design`, `fmrimod.formula`, `fmrimod.baseline`,
  `fmrimod.design_colmap`, and contrast helpers used by design workflows.
- Type baseline: project-local `$types-end-to-end` skill and `MISSION.md`.

## Audit Findings

The current port has strong behavioral coverage, but the design/formula seam is
still permissive in ways that will not scale:

- `Term._kwargs`, formula `kwargs`, list specs, and builder options preserve
  untyped string-keyed escape hatches.
- Formula lowering is lossy: `hrf()` options such as `subset`, `durations`,
  `lag`, `prefix`, and multi-basis controls can parse but not necessarily reach
  design construction.
- The typed `fmrimod.spec` layer exists, but it is not yet the canonical
  implementation target for formula/list/DSL inputs.
- R nuisance diagnostics (`check_nuisance`, `clean_nuisance`,
  `nuisance_check`) were not represented in Python baseline construction.
- Design metadata is reconstructed from column names in places where R attaches
  per-column metadata during construction.
- A few R exports or extension points, notably contrast mask helpers, still
  need an explicit Python decision.

## First Slice Landed

The first slice makes the highest-signal problems executable without changing
the public formula entry point:

- `parse_formula(..., for_event_model=True)` now preserves core HRF term
  controls (`normalize`, `summate`, `id`) and retains remaining parsed HRF
  options in the term option surface for follow-up typed lowering.
- The functional pipe API hoists legacy HRF options to the fields read by
  `EventModel`.
- The DSL `hrf.spm_canonical` alias now emits the underscore name accepted by
  the HRF resolver.
- Baseline nuisance diagnostics now mirror fmridesign's public behavior:
  `check_nuisance()`, `clean_nuisance()`, and
  `baseline_model(..., nuisance_check={"warn","error","drop","none"})`.

## Remaining Work Items

| Bead | Priority | Scope | Exit criterion |
| --- | --- | --- | --- |
| `bd-01KRGG0B7K58PMDQ9NBZHPXX3W` | 1 | Canonical typed lowering | String/list/DSL/functional formula inputs lower through typed `fmrimod.spec` primitives; no public implementation path relies on private kwargs as its contract. |
| `bd-01KRGG0B84BPR1EJ2V74S2NST7` | 1 | Event formula semantic parity | R-fixture or oracle tests cover formula LHS onset handling, `hrf(subset=...)`, term-specific durations/onsets, HRF options beyond the first slice, and block validation/order. |
| `bd-01KRGG0B8A9S05JREP7Q77R1TF` | 2 | Design metadata parity | Per-column metadata is attached during construction; factor multi-basis terms emit one metadata row per expanded design column. |
| `bd-01KRGG0B8S79Q91QY83WV6277B` | 2 | Contrast mask helpers | `contrast_mask` / `contrast_from_mask` are ported or explicitly marked `pythonized`/`scoped_out` in the trio inventory with tests. |

## Target Shape

1. Typed `Spec` is the canonical internal design IR.
2. Formula parser, formula lists, builders, DSL, and functional pipes become
   adapters into that IR.
3. Option names become closed typed fields or small typed option objects.
4. Compatibility aliases are normalized once at the boundary.
5. Design construction attaches metadata as a first-class artifact rather than
   inferring it from generated column names.
6. Parity gaps get deterministic fixtures or documented caveats before claims
   in `trio_api_inventory_v1.md` are upgraded.

## Validation Contract

Every child slice must include:

- a focused Python test in `tests/design/`;
- an R fixture, R-oracle comparison, or a Python synthetic oracle when the R
  behavior is deterministic and small;
- an update to `docs/contracts/trio_api_inventory_v1.md` when public status
  changes;
- a `docs/contracts/CAVEATS.md` entry only if a generated parity report gains a
  new structured caveat ID.
