# ROADMAP to 1.0

Status: draft roadmap, codified from the `roadmap-to-1-0` board thread.

Primary discussion source:

- Opening shot: `roadmap-to-1-0/post-01KRJ65WAPN60WZP062B5F1DNZ`
- Consensus reply: `roadmap-to-1-0/post-01KRJ6D0T2D4MP01ZZYCRFN65Q`
- Consensus reply: `roadmap-to-1-0/post-01KRJ6DEPB3Y61QRJ8X68SAETX`

## Thesis

fmrimod 1.0 is done when the flagship public seam is real, inspectable,
native where it matters, and defensible to a skeptical outside user. It is not
done when every inherited R export has been mechanically ported.

The 1.0 bar should be smaller than port-completeness and harder than a demo
checklist: one public path, one regenerable proof bundle, strict caveat
discipline, native group inference, and a classified API spine.

## Release Blockers

### 1. One Regenerable Proof Bundle

The release candidate must produce a single 1.0 receipt that a user can rerun
and inspect. This is probably a benchmark/report artifact, not a new public
API.

The bundle must show:

- the flagship workflows included in 1.0;
- the exact public fmrimod path used for each workflow;
- the typed objects emitted along the way;
- numerical agreement or documented divergence against the comparator;
- active caveats and their owner/exit criterion;
- timings or performance receipts;
- where Nilearn, FitLins, R, or an R bridge required a matrix shim or lost
  semantic information that fmrimod preserves.

A pile of demos is not enough. The proof must read as one release artifact.

### 2. The Four-Stage Seam Carries the Flagship Path

The mission spine is:

```text
fmri_dataset -> fmri_lm -> contrast -> group_fit
```

For 1.0, that path must be load-bearing for flagship workflows. It cannot be a
demo-only route, a partial facade over private matrix fixtures, or one path
among several incompatible paths.

At least one representative workflow from each promised flagship family needs
a strict public-seam receipt:

- SPM auditory / first-level modeling;
- FIAC or localizer-style fixed-effects analysis;
- FitLins/BIDS Stats Model translation where it is part of the public path;
- second-level/group inference;
- single-trial/LSS or trialwise estimation.

Numerical canaries remain useful, but a canary cannot masquerade as flagship
proof. If an artifact is `public_seam=false`, it is not a 1.0 flagship receipt.

### 3. The Group Path Is Native

`fmrimod.group` must be native enough for the flagship group workflow.
`fmrigds-r` may remain an oracle, regression comparator, or fallback backend,
but it cannot be required on the user-facing critical path for 1.0 group
inference.

This makes the StudyDataset/group container work and the native group reducer
boundary release-critical, not optional polish.

### 4. Dataset Is the Source of Truth for the Flagship Seam

`fmrimod.dataset` owns dataset, timing, backend, chunk, series, and study
semantics for the public seam. The facade can re-export. Optional backends can
remain uneven. But flagship proof must not depend on benchmark-local dataset
objects, matrix shims, or private fixture conventions.

For 1.0, this requirement is scoped to the workflows in the proof bundle. We
do not block the release on every optional backend constructor.

### 5. Caveats Are Boring

The caveat index does not need to be empty. It must be honest, small, owned,
and principled.

Release-blocking failures:

- vague tolerance escapes;
- retired caveats that leave bypass keywords in the proof gate;
- caveats without an owner and exit criterion;
- canary or benchmark caveats that obscure whether the public seam really
  passed.

The current `dfres-n-minus-rank` issue can be 1.0-compatible if it is recorded
as a statistically defensible, permanent divergence or resolved by a changed
policy. It cannot remain ambiguous drift.

### 6. The Public API Spine Is Classified

The whole public inventory does not need to be burned down before 1.0. The
release blocker is narrower:

- no `review_pending` names in the top-level product story;
- no `review_pending` names in the four-stage public spine;
- no `review_pending` names used by flagship docs, reports, or benchmark
  artifacts;
- no opaque public object on the flagship path.

The long tail becomes a ratcheted 1.x compatibility discipline. 1.0 should not
turn back into namespace-completeness work.

### 7. Analysis Objects Explain Themselves

Fits, contrasts, and group results must carry enough semantic information for
a user to understand the analysis without dropping to raw NumPy arrays.

For flagship artifacts, the user should be able to recover:

- modeled columns and design provenance;
- hypothesis intent and contrast semantics;
- degrees-of-freedom policy;
- active caveats;
- backend/comparator provenance;
- enough reproducibility metadata to understand why the result exists.

Full replay/serialization can be a 1.1 target if needed. Semantic opacity in
the flagship path is a 1.0 blocker.

### 8. The Evidence Gates Are Green

At minimum, a 1.0 candidate must pass:

```bash
python3.9 -m pytest tests/ -k "not rpy2"
```

It must also pass the flagship proof bundle and any parity/benchmark checks
attached to changed artifacts. The 26 rpy2 baseline-spline parity failures are
pre-existing and excluded by the documented filter.

## Critical Path

Current ordering from the board consensus:

1. Finish the StudyDataset/group source-of-truth slice needed by the flagship
   proof path: `bd-01KRFMVAY8PGBR86YERMD6DK7N`.
2. Promote or replace the remaining matrix/canary flagship-adjacent rows so
   they run through the canonical public typed seam:
   `bd-01KRFMY8FNTZM60BKZ7MW5W2Q9`.
3. Finish the native group reducer policy/kernel/registry boundary:
   `bd-01KRHTJ9WFSSBZSDGAN4V7PHGS`.
4. Build the 1.0 proof bundle target and release receipt. This roadmap is
   tracked by `bd-01KRJ6KVSHNJDVM374563P45S6`; a separate implementation bead
   should own the proof-bundle command once its exact red check is specified.
5. Classify the public spine and flagship-artifact symbols:
   `bd-01KRHY6EQW0K06KESGF6MGVZT3`.
6. Decide or permanently record the rank-deficient dfres divergence:
   `bd-01KRHTASRWPA5ZQNGV55BS6XFE`.

## Proof Overcounting Rules

These artifacts are useful, but they do not count as 1.0 flagship proof:

- a benchmark row labeled `public_seam=false`;
- a numerical canary whose report reads like a flagship workflow;
- a workflow that reaches through `fit_glm_from_suffstats` because the typed
  design entry is missing;
- a workflow that depends on matrix shims or benchmark-local dataset objects;
- a group workflow whose user-facing path still requires `fmrigds-r`;
- a namespace export that is present but still `review_pending` in the
  inventory and used by a flagship artifact.

## Non-Goals for 1.0

The following are post-1.0 unless they directly break the flagship proof:

- exhaustive R-export parity;
- every optional dataset backend constructor;
- every HRF/design helper being perfectly classified;
- exhaustive BIDS Stats Model transform coverage;
- all performance trend wiring beyond timings or tracked regressions in the
  release receipt;
- broad type hardening that does not touch release proofs;
- new public abstractions not needed by the flagship seam;
- full replay/serialization of every analysis object.

## Acceptance Checklist

Before tagging 1.0:

- [ ] The 1.0 proof bundle regenerates from a documented command.
- [ ] The proof bundle includes the exact public path for each flagship
      workflow.
- [ ] Every flagship workflow uses the four-stage public seam or explicitly
      documents why the stage is not applicable.
- [ ] No flagship workflow depends on private matrix shims or benchmark-local
      dataset objects.
- [ ] The flagship group path runs through native `fmrimod.group`.
- [ ] `docs/contracts/CAVEATS.md` contains only owned, principled caveats with
      exit criteria.
- [ ] No retired caveat leaves a bypass keyword in the relevant gate.
- [ ] Top-level exports, four-stage spine names, and flagship artifact symbols
      are classified in the public API inventory.
- [ ] Flagship fit, contrast, and group objects explain modeled columns,
      hypotheses, df policy, caveats, and provenance.
- [ ] `python3.9 -m pytest tests/ -k "not rpy2"` passes.
- [ ] Flagship parity/proof checks pass under their documented commands.

## Version Boundary

1.0 may ship with documented, principled caveats. It may not ship with
ambiguous proof, critical R-bridge dependence in the flagship group path,
unclassified public spine symbols, or flagship workflows that only work by
leaving the public seam.

The artifact is the argument. If the proof bundle is impossible to dismiss,
1.0 is in reach. If the proof still needs canary laundering, matrix shortcuts,
opaque objects, or bridge-only group inference, the release is not ready.
