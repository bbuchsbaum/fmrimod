# fmrilss Test Surface v1

Status: Active coverage map

## Purpose

This file maps the R `fmrilss` `tests/testthat` surface in
`~/code/fmrilss` to the Python `fmrimod` test surface. It is not a claim that
R implementation-detail tests are all meaningful in Python. It records which
tests are directly ported, partially covered by Pythonized contracts, or scoped
out because the R test targets an R-only helper, Rcpp backend, optional
third-party backend, or benchmark harness that is not part of the accepted
Python public API.

## Summary

Observed R test files: 60.

- Direct or Pythonized coverage exists for core LSS, LSA, OASIS, voxel-HRF,
  SBHM, ITEM, mixed-model, prewhitening, and method-comparison surfaces.
- R-only C++ wrappers, `stglmnet`, benchmark report builders, and some LWU
  simulation/grid-search helpers are not currently ported as public Python APIs.
- The strongest executable anchors are `tests/test_single`, the top-level GLM
  wrapper tests in `tests/test_model/test_fmri_model.py`, and the design docs
  migration page `docs/source/design/fmrilss_migration.rst`.

## File Map

| R test file | Python coverage status | Python evidence | Notes |
| --- | --- | --- | --- |
| `test-aaa-utils.R` | scoped_out | none | R-only internal naming, matrix coercion, `%||%`, and dimname helpers. Python uses arrays and result objects. |
| `test-accuracy-validation.R` | partial | `tests/test_single/test_lss.py`, `tests/test_single/test_project.py` | Core numerical consistency and projection properties are covered; memory profiling is not. |
| `test-benchmark-methods.R` | scoped_out | none | R benchmark/report helper surface is not a Python public API. |
| `test-benchmark-mixed-solve.R` | partial | `tests/test_single/test_mixed.py` | Solver behavior is covered; benchmark object semantics are not. |
| `test-dof-se.R` | ported | `tests/test_single/test_lss.py`, `tests/test_single/test_oasis.py` | SE/dof behavior now includes nuisance-rank-sensitive LSS and OASIS SE checks. |
| `test-fmriAR-integration.R` | partial | `tests/test_single/test_dispatcher.py` | Python prewhitening dispatcher and config validation are covered; full `fmriAR` package parity is scoped out. |
| `test-item-algebra.R` | ported | `tests/test_single/test_item.py` | Closed-form, ridge, permutation, and invariance contracts are covered. |
| `test-item-contracts.R` | ported | `tests/test_single/test_item.py` | Bundle alignment, hashes, deterministic CV, tie handling, and metrics are covered. |
| `test-item-numerics.R` | ported | `tests/test_single/test_item.py` | Collinearity, solver fallback, sparse precision, and finite outputs are covered. |
| `test-item-sim-null-signal.R` | ported | `tests/test_single/test_item.py` | Null classification/regression and SNR improvement are covered. |
| `test-lsa-coverage.R` | ported | `tests/test_single/test_lsa.py`, `tests/test_model/test_fmri_model.py` | LSA dimensions, confounds, baseline regressors, SE, labels, 1-D response, and wrapper paths are covered. |
| `test-lss-correctness.R` | partial | `tests/test_single/test_lss.py` | Direct per-trial GLM oracle is covered; exact R implementation matrix is not. |
| `test-lss-cpp-optimized.R` | scoped_out | none | Rcpp/C++ backend helper tests are R implementation details. Python uses NumPy vectorization. |
| `test-lss-design.R` | partial | `tests/design`, `tests/test_single/test_dispatcher.py` | Event/design construction is covered elsewhere; exact `lss_design()` helper is not a Python public API. |
| `test-lss-equivalence.R` | ported | `tests/test_single/test_lss.py`, `tests/test_model/test_fmri_model.py` | LSS baseline, nuisance, naming via labels, edge guards, and wrapper behavior are covered. |
| `test-lss-fit-wrappers.R` | ported | `tests/test_model/test_fmri_model.py`, `tests/test_single/test_lss.py` | `glm_lss`/`estimate_betas_lss` wrapper paths and baseline adjustment are covered. |
| `test-lss-sbhm-design.R` | partial | `tests/test_single/test_sbhm.py` | SBHM pipeline exists; exact R design helper is not separately exposed. |
| `test-lss-with-hrf.R` | partial | `tests/test_single/test_voxel_hrf.py` | HRF-aware LSS is covered through matrix/basis APIs. |
| `test-lss-with-hrf-backends.R` | partial | `tests/test_single/test_voxel_hrf.py` | Python has one NumPy backend; R backend-comparison cases are scoped out. |
| `test-lss-with-hrf-cpp.R` | scoped_out | none | R C++ backend comparison is scoped out. |
| `test-lss-with-hrf-durations.R` | partial | `tests/test_single/test_voxel_hrf.py` | Matrix-based HRF-aware LSS is covered; onset/duration builder parity is not yet public. |
| `test-lss-with-hrf-pure-r.R` | partial | `tests/test_single/test_voxel_hrf.py` | Python matrix API validation is covered; pure-R fallback chain is scoped out. |
| `test-method-comparison.R` | ported | `tests/test_single/test_dispatcher.py`, `tests/test_single/test_mixed.py` | LSS/OASIS/LSA/mixed relationships are covered for representative synthetic cases. |
| `test-mixed-solve-equivalence.R` | partial | `tests/test_single/test_mixed.py` | Mixed solver shape, shrinkage, confounds, and labels are covered; R workspace precompute API is not. |
| `test-oasis-api-compat.R` | partial | `tests/test_single/test_oasis.py` | Result object semantics replace R matrix/list return modes; SE and validation are covered. |
| `test-oasis-defaults-and-io.R` | partial | `tests/test_single/test_oasis.py`, `tests/test_single/test_dispatcher.py` | OASIS config, AR prewhitening path, and finite outputs are covered; Matrix dimname preservation is R-specific. |
| `test-oasis-designspec-safety.R` | partial | `tests/design`, `tests/test_single/test_oasis.py` | Design safety is mostly covered through event/design tests; OASIS design_spec helper is not public. |
| `test-oasis-glue-coverage.R` | partial | `tests/test_single/test_oasis.py` | Input validation, ridge modes, block columns, K paths, and SE are covered; private R glue helpers are not. |
| `test-oasis-grid.R` | scoped_out | none | `fit_oasis_grid` is not currently a Python public API. |
| `test-oasis-hrf-recovery-funcs.R` | scoped_out | none | LWU design/grid helper functions are not currently public Python APIs. |
| `test-oasis-hrf-recovery-extended.R` | scoped_out | none | LWU grid/search simulation surface is not currently public Python API. |
| `test-oasis-improvements.R` | partial | `tests/test_single/test_oasis.py`, `tests/test_single/test_dispatcher.py` | Multi-basis, SE, K validation, ridge, and prewhitening smoke paths are covered. |
| `test-oasis-K-autodetect.R` | partial | `tests/test_single/test_oasis.py`, `tests/test_single/test_voxel_hrf.py` | Explicit K and basis-inferred K are covered; R heuristic autodetect helper is not public. |
| `test-oasis-se-coverage.R` | ported | `tests/test_single/test_oasis.py` | K=1 SE behavior and nuisance-aware checks are covered. |
| `test-oasis.R` | ported | `tests/test_single/test_oasis.py` | K=1, K>1, ridge, confounds, SE, labels, block chunking, and validation are covered. |
| `test-options.R` | ported | `tests/test_single/test_oasis.py`, `tests/test_single/test_dispatcher.py` | OASIS and prewhiten config validation are covered with Python dataclasses. |
| `test-prewhiten-robust.R` | partial | `tests/test_single/test_dispatcher.py` | Python prewhitening has smoke and validation coverage; exact rank-deficient fmriAR path is not. |
| `test-prewhiten-run-intercepts.R` | partial | `tests/test_single/test_dispatcher.py`, `tests/test_single/test_lss.py` | Adjustment rank and prewhitening smoke paths are covered; run-specific fmriAR logic is not. |
| `test-prewhitening-coverage.R` | partial | `tests/test_single/test_dispatcher.py` | `method="none"` and AR smoke paths are covered. |
| `test-regression-critical-guards.R` | ported | `tests/test_single/test_lss.py`, `tests/test_single/test_voxel_hrf.py` | Zero/degenerate design guards and finite output checks are covered. |
| `test-sbhm-amplitude-functions.R` | partial | `tests/test_single/test_sbhm.py` | Public amplitude path is covered; private R helper functions are not. |
| `test-sbhm-amplitude-helpers.R` | scoped_out | none | Private R helper functions are not public Python API. |
| `test-sbhm-amplitude-methods.R` | partial | `tests/test_single/test_sbhm.py` | Public amplitude methods are covered at smoke level. |
| `test-sbhm-amplitude-recovery.R` | partial | `tests/test_single/test_sbhm.py` | Pipeline shape and finite behavior are covered; recovery benchmark thresholds are not. |
| `test-sbhm-build-coverage.R` | partial | `tests/test_single/test_sbhm.py` | Library construction is covered. |
| `test-sbhm-isi-sweep.R` | scoped_out | none | Benchmark/sweep surface is not public Python API. |
| `test-sbhm-prepass-robust.R` | partial | `tests/test_single/test_sbhm.py` | Prepass shape/confounds are covered; dense/factorized and fmrihrf summate parity are not. |
| `test-sbhm-project.R` | scoped_out | none | `sbhm_project` is not currently exposed in Python. |
| `test-sbhm-softblend.R` | partial | `tests/test_single/test_sbhm.py` | `top_k` matching/blending is covered. |
| `test-sbhm.R` | ported | `tests/test_single/test_sbhm.py` | Public SBHM library, prepass, match, amplitude, and pipeline paths are covered. |
| `test-stglmnet-backend.R` | scoped_out | none | `stglmnet`/`glmnet` backend is not part of the Python dependency surface. |
| `test-voxel-hrf-coverage.R` | partial | `tests/test_single/test_voxel_hrf.py`, `tests/test_single/test_estimate_hrf.py` | Matrix and limited dataset estimate-HRF paths are covered. |
| `test-voxel-hrf-functions.R` | partial | `tests/test_single/test_voxel_hrf.py` | Core voxel-HRF estimation is covered. |
| `test-voxel-hrf.R` | partial | `tests/test_single/test_voxel_hrf.py` | Formula/dataset path is covered elsewhere; matrix path is covered here. |
| `test-voxhrf-extended.R` | partial | `tests/test_single/test_voxel_hrf.py` | Input validation, chunking, K inference, and confounds are covered; advanced shrinkage/AR/fmriAR controls are not. |
| `test-voxhrf.R` | ported | `tests/test_single/test_voxel_hrf.py` | Finite betas, dimensions, canonical approximation, and chunks are covered. |
| `test-zero-regressors.R` | ported | `tests/test_single/test_lss.py` | Zero-regressor warnings and finite output behavior are covered. |
| `test_oasis_hrf_recovery.R` | scoped_out | none | LWU/OASIS HRF recovery benchmark is not currently a Python public API. |
| `helper-skip-compat.R` | scoped_out | none | R test helper only. |
| `helper.R` | scoped_out | none | R test helper only. |

## Current Gaps

- No Python public `stglmnet`/`glmnet` backend.
- No public benchmark/report helper API equivalent to R benchmark suites.
- No public LWU grid-search/recovery helper API equivalent to the R OASIS HRF
  recovery utilities.
- No direct Python public wrappers for Rcpp/C++ helper entry points.
- Some SBHM private helper tests are represented only through the public SBHM
  pipeline, not one helper per R internal function.

## Maintenance Rule

When adding or changing `fmrimod.single` APIs, update this map in the same
commit as the executable tests. Do not count a row as `ported` unless there is
an executable Python test named in the evidence column.
