# Trio Public API Policy v1

Status: Current port-stability contract

Date: 2026-05-12

## Purpose

This note defines what "stable public API" means for the `fmrihrf`,
`fmridesign`, and `fmrireg` trio port in `fmrimod`. It complements
`docs/contracts/trio_api_inventory_v1.md`, which maps R exports to concrete
Python surfaces and evidence anchors.

## Stable Surface

The stable trio-port surface consists of:

- top-level names exported through `fmrimod.__all__`;
- package-level compatibility modules documented in the inventory's Python
  surface column;
- documented migration helpers in `docs/source/design/fmrireg_migration.rst`
  and the HRF/design docs;
- benchmark and cross-testing entry points under `cross_testing` that are used
  as parity/performance gates.

Rows marked `ported` in the inventory are expected to keep their Python name,
call shape, and return contract stable unless the inventory and migration docs
are updated in the same change.

Rows marked `pythonized` are intentionally not exact R spellings or S3/S4
contracts. Their stability promise is the documented Python behavior and
migration rationale in the inventory notes, not full R dispatch compatibility.

Rows marked `scoped_out` are not public Python APIs. These are primarily R
package install helpers, CLIs that have not been accepted for v1, leading-dot
R helpers, and Rcpp implementation details whose behavior should be exposed
through higher-level Python functions instead.

## Change Control

Changes that affect a `ported` or `pythonized` inventory row should update, in
the same patch:

- the implementation surface;
- the relevant unit, parity, golden, or benchmark tests;
- `docs/contracts/trio_api_inventory_v1.md`;
- migration docs when user-facing behavior changes;
- `docs/contracts/trio_verification_transcript_v1.md` when the change affects
  completion evidence.

Backward-incompatible changes to top-level names or documented call shapes
should be treated as release-boundary changes and called out explicitly in the
inventory notes.

## Evidence Standard

Inventory evidence anchors are concrete files, but they are not all
one-test-per-export proofs. Completion evidence is strongest when an export is
covered by at least one of:

- direct unit tests for the named Python surface;
- R or golden parity tests for numerical behavior;
- cross-testing artifacts for reference-package parity;
- migration documentation that demonstrates the supported Python workflow.

Namespace-level evidence is acceptable for small accessors, aliases, and helper
functions when the broader test file exercises the owning subsystem and the
inventory names the exact Python surface.
