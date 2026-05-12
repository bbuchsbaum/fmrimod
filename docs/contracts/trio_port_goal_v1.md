# Trio Port Goal Contract v1

Status: Draft

## Objective

Port the related R packages `fmrihrf`, `fmridesign`, and `fmrireg` into
`fmrimod` as one idiomatic Python package. The port is complete only when the
Python implementation has a stable public API, explicit R-parity contracts,
robust executable verification, migration documentation, and reproducible
performance evidence for the main first-level workflows plus a scoped
second-level interface.

## Source Contracts

The R packages remain the behavioral source of truth unless this document or a
more specific contract deliberately chooses a Python-first behavior.

- `fmrihrf`: HRF objects, basis functions, HRF generators/decorators,
  reconstruction/penalty helpers, sampling frames, event regressors, regressor
  sets, neural input construction, and HRF/regressor plotting.
- `fmridesign`: event objects, formula parsing, event terms, event models,
  design matrix assembly, baseline/nuisance models, parametric bases,
  condition/cell/naming utilities, contrasts/F-contrasts, validation,
  residualization, and design visualization.
- `fmrireg`: fMRI model objects, dataset adapters, first-level OLS/GLS/AR
  fitting, contrasts, fitted HRF summaries, trialwise LSA/LSS estimates,
  robust fitting, bootstrap helpers, AR diagnostics/whitening, simulation,
  result writing, group data ingestion, meta/ttest interfaces, spatial FDR, and
  performance-oriented fitting engines.

## Python Design Rules

- Prefer Pythonic names, dataclasses/protocols, NumPy/pandas conventions, and
  explicit string formulas over R non-standard evaluation.
- Preserve R semantics where they affect numeric results, naming, rank handling,
  censoring, block/run boundaries, contrasts, AR handling, or interpretation.
- Keep compatibility aliases only when they materially ease migration or protect
  parity fixtures.
- Every intentional divergence from R behavior must be documented with a
  contract note and test coverage.

## Required Deliverables

### D1. Public API Inventory

Produce and maintain an inventory mapping R exports to one of:

- `ported`: supported by a stable Python API.
- `pythonized`: supported by an intentionally different Python API.
- `scoped_out`: deliberately excluded with rationale.
- `pending`: not yet implemented or not yet verified.

The inventory must cover all exported user-facing functions/classes from the
three R packages and point to the Python module, test file, and documentation
location for each supported surface.

Current inventory artifact: `docs/contracts/trio_api_inventory_v1.md`.

### D2. HRF and Regressor Parity

`fmrihrf` parity must include:

- canonical, gamma, Gaussian, B-spline, Fourier, FIR, Daguerre/tent-like, LWU,
  weighted, empirical, boxcar, sine, half-cosine, inverse-logit, Mexican-hat,
  lagged, and blocked HRF behavior;
- basis size, derivative, reconstruction matrix, and penalty matrix contracts;
- sampling frame block/run semantics;
- regressor/regressor-set construction, durations, amplitudes, neural input,
  convolution, sparse/dense output, and single-trial regressors.

Verification must include golden tests or R comparison fixtures for numerical
outputs and shape/name semantics.

### D3. Design and Contrast Parity

`fmridesign` parity must include:

- categorical, continuous, matrix, and basis event representations;
- event formulas, interaction expansion, block handling, event tables, condition
  maps, cells, and column naming;
- HRF integration, parametric modulators, trialwise models, and baseline/nuisance
  terms;
- design matrix assembly, split-by-block behavior, residualization,
  collinearity/rank diagnostics, and visualization data contracts;
- unit, pair, pairwise, one-vs-all, one-way, interaction, polynomial, column,
  formula, set, and F-contrast behavior.

Verification must include deterministic fixtures for values, column order,
condition names, rank-deficient designs, and contrast weights.

### D4. First-Level Model Parity

`fmrireg` first-level parity must include:

- `fmri_model` construction from formula/design/dataset inputs;
- OLS, GLS, AR1, AR(p)/ARMA-oriented whitening, robust fitting, runwise fitting,
  fixed-effects run combination, contrasts, coefficient/statistic extraction,
  fitted HRF summaries, and result writing;
- censor/sample-mask handling without run-boundary leakage;
- LSA/LSS/trialwise beta workflows;
- simulation helpers used by parity and examples.

Verification must include contract tests plus parity benchmarks for beta, SE,
statistic, p-value, residual variance, df, and censor-sensitive fixtures.

### D5. Group/Second-Level Scope

The second-level surface is intentionally scoped. The canonical contract is
`docs/contracts/second_level_parity_v1.md` and must stay ahead of
implementation changes.

The required v1 surface is:

- group-data constructors for CSV, H5/HDF5, NIfTI, and fitted-model-like inputs;
- fixed/random meta-analysis and one-sample t-test wrappers;
- explicit backend selection, correction aliases, and canonical axis convention;
- spatial FDR only when required grouping metadata are supplied.

### D6. Performance Evidence

Performance claims must be backed by checked-in benchmark scripts and JSON
artifacts or CI-generated artifacts for:

- design matrix construction;
- GLM fit;
- contrast evaluation;
- whitening/AR estimation;
- run combination;
- LSA/LSS;
- end-to-end first-level workloads.

Core parity/performance gates are defined by
`cross_testing/PRD_CORE_PARITY_AND_PERFORMANCE.md`.

### D7. Documentation and Migration

Documentation must describe the current implementation, not a desired future
state. Required docs:

- R-to-Python migration guide for `fmrihrf`;
- R-to-Python migration guide for `fmridesign`;
- R-to-Python migration guide for `fmrireg`;
- examples for HRF/regressor, event design, baseline/contrast, first-level GLM,
  AR/whitening, LSA/LSS, simulation, and group/meta workflows;
- a clear list of unsupported or intentionally different behaviors.

### D8. Continuous Verification

CI or an equivalent local verification command set must cover:

- import/type-hint/import-order checks;
- unit tests for supported API surfaces;
- cross-language or golden parity tests;
- core parity matrix benchmarks;
- schema validation for benchmark artifacts;
- documentation build for public examples.

## Completion Audit Checklist

Before this goal can be marked complete, verify each item with concrete
evidence:

- `README.md` describes all three R sources, not only `fmrihrf` and
  `fmridesign`.
- A public API inventory exists and has no unmapped R exports.
- All `pending` inventory rows have either an open tracked issue or an explicit
  `scoped_out` decision.
- HRF/regressor golden tests cover the `fmrihrf` contract above.
- Design/contrast tests cover the `fmridesign` contract above.
- First-level GLM/AR/LSS tests cover the `fmrireg` contract above.
- `docs/contracts/second_level_parity_v1.md` matches the implemented group API.
- Core parity matrix workstreams WS01-WS10 pass with required gates.
- Benchmark artifact schemas validate for protected performance paths.
- Migration docs exist for all three R packages and examples import from
  `fmrimod`, not legacy package names.
- A clean verification transcript is recorded with exact commands and outcomes.

Current audit artifact: `docs/contracts/trio_completion_audit_v1.md`.

## Current Known Gaps

This repository already contains substantial implementation and parity
infrastructure, including HRF/design/GLM/AR modules, golden tests, a second-level
contract, and a core parity/performance PRD. The remaining work is to make the
port auditable end to end:

- create the full R-export to Python-API inventory;
- reconcile bead state against the current `.beads` database and JSONL export;
- expand README lineage from two R sources to the full trio;
- verify that existing golden/cross tests cover the full contract rather than
  only selected surfaces;
- add or close gaps for any unmapped `fmrireg` model/dataset/group/stat APIs.
