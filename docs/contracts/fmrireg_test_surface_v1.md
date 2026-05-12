# fmrireg Test Surface v1

Status: Active coverage map

## Purpose

This file maps the R `fmrireg` `tests/testthat` surface in
`~/code/fmrireg` to Python `fmrimod` tests and contracts. It is not a claim
that every R S3 method, C++ helper, GDS/fmrigds bridge, benchmark script, or
CLI path has a literal Python twin. It records which tests are directly
ported, covered through Pythonized public contracts, partially covered because
the accepted Python surface is narrower, or scoped out because the R test is
not part of the supported Python public API.

## Summary

Observed R testthat files in the local `~/code/fmrireg` checkout: 99.

- Executable R test files: 96.
- R helper/setup files: 3.
- Executable `test_that()` blocks: 531.
- Direct or Pythonized coverage exists for GLM solving, first-level result
  accessors, top-level `glm_ols`/`glm_lss` wrappers, contrasts, runwise and
  chunkwise engines, AR estimation and whitening, robust fitting, volume
  preprocessing, LSA/LSS beta extraction, simulation helpers, group-data
  ingestion, fixed/random meta-analysis, t-tests, spatial FDR, low-rank/sketch
  helpers, BIDS/result export, and migration-facing convenience APIs.
- The strongest executable anchors are `tests/test_glm`, `tests/test_ar`,
  `tests/test_model`, `tests/test_single`, `tests/test_stats`,
  `tests/test_dataset`, `tests/test_io`, `tests/test_accessors.py`, and
  selected `cross_testing` parity/benchmark harnesses.
- GDS/fmrigds, external R package adapters, R CLI behavior, R-specific helper
  files, and some benchmark/performance-only tests are recorded as partial or
  scoped out rather than overclaimed.

Focused verification snapshot on 2026-05-12:

- `uv run pytest tests/test_model/test_fmri_model.py tests/test_accessors.py tests/test_single/test_lss.py`:
  46 passed.

## File Map

| R artifact | Python coverage status | Python evidence | Notes |
| --- | --- | --- | --- |
| `helper-fmrigds.R` | scoped_out | none | R test helper for fmrigds fixtures; Python has fmrigds bridge payload tests but no public helper equivalent. |
| `helper-naming.R` | scoped_out | none | R test helper only; naming behavior is covered elsewhere through public design/accessor tests. |
| `setup.R` | scoped_out | none | R test setup file, not a public API. |
| `test-ar-by-cluster-benefit.R` | partial | `tests/test_ar/test_multiscale.py`, `tests/test_ar/test_integration.py` | Python covers parcel/multiscale AR helpers and whitening pipelines; exact residual-variance benefit benchmark is not a stable unit contract. |
| `test-ar-whitening.R` | ported | `tests/test_ar/test_whitening.py`, `tests/test_ar/test_numhelpers.py` | AR(1)/AR(2) whitening and helper kernels are covered. |
| `test-censor-ar-integration.R` | partial | `tests/test_glm/test_preprocess.py`, `tests/test_ar/test_integration.py`, `tests/test_glm/test_strategies.py` | Censor vector semantics and run slicing are covered; exact R dataset auto-extraction internals are not literal Python APIs. |
| `test-cli.R` | scoped_out | none | `fmrireg_cli` is scoped out in the public inventory; `fmrimod` currently supports the HRF-oriented CLI surface separately. |
| `test-convolution-fixes.R` | ported | `tests/design/test_convolve.py`, `tests/design/test_convolve_design.py`, `tests/design/test_convolve_formula_hrf.py` | Dimension, duration, multi-run, and convolution regression behavior is covered by the shared design surface. |
| `test-engine-capability-contracts.R` | partial | `tests/test_glm/test_engine.py`, `tests/test_lowrank/test_engine.py` | Built-in engine registration and low-rank config behavior are covered; R plugin capability routing is narrower in Python. |
| `test-fit-contrasts-fmri-lm.R` | ported | `tests/test_glm/test_contrasts.py`, `tests/test_glm/test_fmrireg_compat.py` | t/F contrast computation and compatibility wrappers are covered. |
| `test-fmri_ttest.R` | ported | `tests/test_stats/test_ttest.py`, `tests/test_stats/test_inference_core.py`, `tests/test_stats/test_group_fit_interface.py` | One-sample, paired/classic, Welch/meta-backed t-test and correction behavior is covered in the scoped Python second-level API. |
| `test-gds-bridges.R` | partial | `tests/test_stats/test_fmrigds_backend_payloads.py`, `tests/test_stats/test_fmrigds_backend_unit.py` | Python validates bridge payloads and subprocess failure handling; live fmrigds execution depends on external R infrastructure. |
| `test-gds-h5-parity.R` | partial | `tests/test_dataset/test_group_data.py`, `tests/test_stats/test_fmrigds_backend_payloads.py` | HDF5 group-data shape and bridge payloads are covered; exact R GDS parity is external-backend dependent. |
| `test-gds-integration.R` | partial | `tests/test_stats/test_fmrigds_backend_payloads.py`, `tests/test_stats/test_group_fit_fmrigds_parity.py` | Backend payload and optional parity harnesses exist; skipped external backend runs are not counted as local proof. |
| `test-gds-nifti-parity.R` | partial | `tests/test_dataset/test_group_data.py`, `tests/test_stats/test_fmrigds_backend_payloads.py` | NIfTI group-data metadata is covered through Python constructors and bridge payloads; live R GDS parity is optional. |
| `test-gds-parity-methods.R` | partial | `tests/test_stats/test_group_fit_fmrigds_parity.py`, `tests/test_stats/test_group_fit_interface.py` | Group-fit backend parity and unsupported custom-weight paths are represented where the Python backend contract exists. |
| `test-gds-posthoc.R` | partial | `tests/test_stats/test_group_fit_corrections.py`, `tests/test_stats/test_spatial_fdr.py` | Spatial correction contracts are covered without requiring the R GDS posthoc path. |
| `test-gds-ttest-parity.R` | partial | `tests/test_stats/test_ttest.py`, `tests/test_stats/test_group_fit_fmrigds_parity.py` | Python t-test/backend interfaces are covered; exact GDS t-test parity remains optional-backend scope. |
| `test-group_analysis.R` | ported | `tests/test_dataset/test_group_data.py`, `tests/test_dataset/test_group_data_top_level.py`, `tests/test_stats/test_meta.py`, `tests/test_stats/test_ttest.py` | Group-data constructors plus meta/t-test workflows are covered. |
| `test-ihs-iters.R` | partial | `tests/test_lowrank/test_engine.py`, `cross_testing/core_parity_matrix.py` | Low-rank execution exists; iteration-quality performance monotonicity is tracked by parity/benchmark harnesses rather than unit tests. |
| `test-landmarks-srht.R` | partial | `tests/test_lowrank/test_sketch.py`, `tests/test_lowrank/test_nystrom.py`, `cross_testing/benchmark_core_parity_matrix.py` | Sketch and Nyström primitives are covered; full landmark SRHT + AR threshold parity is benchmark-harness scope. |
| `test-latent-parity.R` | partial | `tests/test_lowrank/test_engine.py`, `cross_testing/core_parity_matrix.py` | Latent/sketch paths are covered at the Python contract level; exact R latent-dataset parity is not literal. |
| `test-lowrank-ar-srht-ihs.R` | partial | `tests/test_lowrank/test_engine.py`, `cross_testing/benchmark_arma_paths.py` | Low-rank and AR benchmark paths exist; the combined R fixture is a parity/performance harness rather than a small unit port. |
| `test-lowrank-engine-contract.R` | ported | `tests/test_lowrank/test_engine.py`, `tests/test_lowrank/test_imports.py` | Low-rank config, engine imports, and contract behavior are covered. |
| `test-meta-contrast-diagnostics.R` | ported | `tests/test_stats/test_meta_compat.py` | Packed covariance and exact low-level meta contrast shape contracts are covered. |
| `test-meta-t-combination-diagnostics.R` | partial | `tests/test_stats/test_inference_core.py`, `fmrimod/stats/inference.py` | p/z/Fisher helpers exist, but the exact R Stouffer/Fisher diagnostic API is not currently public Python surface. |
| `test-nystrom-identity.R` | ported | `tests/test_lowrank/test_nystrom.py` | Identity and extension behavior are covered. |
| `test-nystrom-weights.R` | ported | `tests/test_lowrank/test_nystrom.py` | Landmark weight sparsity and row-sum behavior are covered. |
| `test-plugin-api.R` | partial | `tests/design/test_extension_registry.py`, `tests/test_glm/test_engine.py`, `tests/test_lowrank/test_engine.py` | Registry/engine hooks are covered for accepted Python extension points; R plugin S3 dispatch is not literal. |
| `test-preprocessing.R` | ported | `tests/test_glm/test_preprocess.py` | DVARS, volume weights, censoring, and soft-subspace preprocessing are covered. |
| `test-rrr-gls-engine.R` | partial | `tests/test_lowrank/test_engine.py`, `cross_testing/core_parity_matrix.py` | Python low-rank/sketch engine coverage exists; exact RRR GLS diagnostics/bootstrap are not all public Python contracts. |
| `test-rrr-parity-matrix.R` | partial | `cross_testing/core_parity_matrix.py`, `cross_testing/benchmark_core_parity_matrix.py` | The parity matrix harness covers the supported benchmark view. |
| `test-spatial_fdr.R` | ported | `tests/test_stats/test_spatial_fdr.py`, `tests/test_stats/test_group_fit_corrections.py` | Spatial FDR, smoothing, input validation, and grouped correction behavior are covered. |
| `test-srht-plan.R` | ported | `tests/test_lowrank/test_sketch.py` | SRHT dimensions and deterministic sketch behavior are covered. |
| `test-tidy-fitted-hrf.R` | ported | `tests/test_accessors.py` | `tidy_fitted_hrf()` now exposes the R-compatible `estimate` column while preserving the Python `value` alias. |
| `test_ar_advanced.R` | partial | `tests/test_ar/test_integration.py`, `tests/test_ar/test_estimation.py`, `tests/test_ar/test_plan.py`, `tests/test_ar/test_diagnostics.py` | AR(p), pooled/runwise, convergence, and diagnostics are broadly covered; some exact R iterative solver internals are not literal. |
| `test_ar_args.R` | ported | `tests/test_model/test_config.py`, `tests/test_glm/test_fmri_lm_api.py` | AR config and fit-entry options are covered. |
| `test_ar_components.R` | ported | `tests/test_ar/test_estimation.py`, `tests/test_ar/test_whitening.py`, `tests/test_ar/test_numhelpers.py` | AR estimation, transforms, and autocorrelation reduction are covered. |
| `test_ar_glm_integration.R` | ported | `tests/test_ar/test_integration.py`, `tests/test_glm/test_strategies.py` | AR + GLM integration and runwise behavior are covered. |
| `test_ar_integration.R` | ported | `tests/test_ar/test_integration.py` | AR structures and iterative GLS behavior are covered. |
| `test_ar_pure_noise.R` | ported | `tests/test_ar/test_estimation.py`, `tests/test_ar/test_whitening.py` | Pure-noise AR recovery and whitening are covered. |
| `test_ar_robust_combined.R` | partial | `tests/test_robust/test_irls.py`, `tests/test_glm/test_strategies.py`, `tests/test_ar/test_integration.py` | Robust and AR paths are covered separately and in strategy tests; exact R combined process internals are partial. |
| `test_ar_robust_sanity.R` | partial | `tests/test_robust/test_irls.py`, `tests/test_ar/test_integration.py`, `tests/test_glm/test_strategies.py` | AR/robust sanity is represented by public Python paths, not every R internal helper. |
| `test_ar_whiten.R` | ported | `tests/test_ar/test_whitening.py` | In-place/exact-first AR whitening behavior is covered. |
| `test_benchmark_datasets.R` | ported_pythonized | `tests/test_dataset/test_fmrireg_compat.py`, `tests/test_simulate/test_compat.py` | Python ships synthetic compatibility fixtures and summary/design helpers rather than R data-package fixtures. |
| `test_betas.R` | ported | `tests/test_betas/test_extraction.py`, `tests/test_model/test_fmri_model.py`, `tests/test_single/test_lsa.py`, `tests/test_single/test_lss.py` | OLS/LSS beta extraction and top-level wrappers are covered. |
| `test_bootstrap.R` | ported | `tests/test_glm/test_bootstrap.py` | Multiresponse bootstrap validation, shape, and reproducibility behavior is covered. |
| `test_convenience_functions.R` | ported_pythonized | `tests/test_model/test_fmri_model.py`, `tests/test_accessors.py` | Top-level `glm_ols`/`glm_lss` wrappers, progress compatibility, voxel-axis behavior, deterministic outputs, non-finite LSS validation, and tidy aliases are covered for the accepted matrix-first Python API. |
| `test_critical_functionality.R` | ported | `tests/test_glm/test_solver.py`, `tests/test_robust/test_irls.py`, `tests/test_ar/test_whitening.py`, `tests/test_glm/test_contrasts.py`, `tests/test_glm/test_bootstrap.py` | Critical solver, robust, AR, contrast, and bootstrap contracts are covered. |
| `test_critical_gaps.R` | ported | `tests/test_glm/test_solver.py`, `tests/test_robust/test_irls.py`, `tests/test_ar/test_whitening.py`, `tests/design/test_event_model.py`, `tests/design/test_baseline_model.py` | Rank, robust, AR, mixed/design, and model basics are covered by Python public tests. |
| `test_data_chunks.R` | ported | `tests/test_dataset/test_protocols_and_adapters.py`, `tests/test_dataset/test_group_data.py`, `tests/test_glm/test_chunkwise_engine.py` | Dataset adapter and chunkwise processing behavior is covered. |
| `test_dataset.R` | ported | `tests/test_dataset/test_protocols_and_adapters.py`, `tests/test_dataset/test_top_level_constructors.py`, `tests/test_dataset/test_fmrireg_compat.py` | Dataset construction and compatibility helpers are covered. |
| `test_effective_df.R` | partial | `tests/test_glm/test_strategies.py`, `tests/test_model/test_config.py`, `fmrimod/glm/effective_df.py` | Effective-df and sandwich helpers exist, with public-path coverage stronger than internal formula-by-formula parity. |
| `test_estimate_ar.R` | ported | `tests/test_ar/test_estimation.py` | AR(1)/AR(2) recovery is covered. |
| `test_f_contrasts.R` | ported | `tests/test_glm/test_contrasts.py`, `tests/design/test_fcontrast.py`, `tests/design/test_fcontrast_system.py` | F contrasts and design contrast construction are covered. |
| `test_fitted_hrf.R` | ported | `tests/test_accessors.py`, `tests/test_single/test_estimate_hrf.py` | Fitted-HRF accessor and validation behavior is covered. |
| `test_fmriAR_integration.R` | partial | `tests/test_ar/test_rpy2_parity.py`, `tests/test_ar/test_fmriar_export_compat.py`, `tests/test_ar/test_integration.py` | fmriAR-facing compatibility tests exist but may skip when external R packages are unavailable. |
| `test_fmri_ar_modeling.R` | ported | `tests/test_ar/test_estimation.py`, `tests/test_ar/test_whitening.py` | AR parameter recovery, whitening, and non-finite guards are covered. |
| `test_fmri_latent_lm.R` | partial | `tests/test_lowrank/test_engine.py`, `cross_testing/core_parity_matrix.py` | Latent/sketch workflows are Pythonized; R latent-dataset constructors are not literal. |
| `test_fmri_lm_config.R` | ported | `tests/test_model/test_config.py`, `tests/test_glm/test_fmri_lm_api.py` | Config creation, validation, propagation, robust options, AR options, and defaults are covered. |
| `test_fmri_lm_dispatch_regressions.R` | ported | `tests/test_glm/test_strategies.py`, `tests/test_glm/test_chunkwise_engine.py`, `tests/test_glm/test_engine.py` | Strategy dispatch and chunkwise/runwise routing are covered. |
| `test_fmri_lm_fmriAR_structures.R` | partial | `tests/test_ar/test_rpy2_parity.py`, `tests/test_glm/test_chunkwise_engine.py`, `tests/test_ar/test_integration.py` | Public AR structure behavior is covered; live fmriAR parity is optional. |
| `test_fmri_meta_csv.R` | ported | `tests/test_stats/test_meta.py`, `tests/test_stats/test_meta_computational.py` | CSV meta-analysis, weights, random-effects estimators, and covariates are covered. |
| `test_fmri_ttest_meta_group_data.R` | ported | `tests/test_stats/test_ttest.py`, `tests/test_dataset/test_group_data.py` | Meta-backed and group-data t-test paths are covered. |
| `test_fmridesign_integration.R` | ported | `tests/design/test_event_model.py`, `tests/design/test_event_model_contrasts.py`, `tests/design/test_convolve_formula_hrf.py`, `tests/design/test_rpy2_parity.py` | Event-model formula, HRF, and contrast integration are covered by the design port. |
| `test_fmriglm.R` | ported | `tests/test_glm/test_end_to_end.py`, `tests/test_glm/test_fmri_lm_api.py`, `tests/test_model/test_fmri_model.py` | In-memory GLM construction, contrasts, and result APIs are covered. |
| `test_fmrimodel.R` | ported | `tests/test_model/test_fmri_model.py`, `tests/test_dataset/test_protocols_and_adapters.py` | Model construction, design slicing, baseline/event matrices, and edge cases are covered. |
| `test_glm_context.R` | ported_pythonized | `tests/test_glm/test_solver.py`, `tests/test_glm/test_strategies.py` | Python exposes solver/projector objects rather than a literal R `glm_context` S3 class; equivalent numeric contracts are covered. |
| `test_glm_regressions_and_routing.R` | partial | `tests/test_glm/test_strategies.py`, `tests/test_glm/test_chunkwise_engine.py`, `tests/test_ar/test_integration.py`, `tests/test_model/test_config.py` | Many public routing regressions are covered; R private `process_run_*` internals are not literal. |
| `test_glm_solver.R` | ported | `tests/test_glm/test_solver.py` | Core OLS, multiple responses, rank deficiency, finite validation, and weighted solves are covered. |
| `test_integration_scenarios.R` | ported | `tests/test_glm/test_end_to_end.py`, `tests/test_glm/test_chunkwise_engine.py`, `tests/test_glm/test_contrasts.py`, `tests/test_robust/test_irls.py` | End-to-end GLM, strategy, contrast, and robust scenarios are covered. |
| `test_iter_gls_residual_domain.R` | partial | `tests/test_ar/test_integration.py`, `tests/test_glm/test_strategies.py` | Iterative GLS public behavior is covered; low-rank raw-domain residual internals are partial. |
| `test_iterators.R` | partial | `tests/test_dataset/test_protocols_and_adapters.py`, `tests/test_glm/test_chunkwise_engine.py` | Dataset and chunkwise iteration are covered; R iterator objects are Pythonized. |
| `test_ls_svd_benchmark.R` | partial | `cross_testing/benchmark_core_parity_matrix.py`, `tests/design/test_event_model.py` | Benchmark fixture integration exists; the R-specific benchmark script is not a unit-test contract. |
| `test_memory_performance.R` | partial | `tests/test_glm/test_chunkwise_engine.py`, `cross_testing/benchmark_core_parity_matrix.py` | Chunkwise/parallel consistency is covered; memory ceilings are benchmark rather than unit-test gates. |
| `test_meta_fit.R` | ported | `tests/test_stats/test_meta_compat.py`, `tests/test_stats/test_meta_computational.py` | Low-level meta fit helpers, random effects, robust/outlier behavior, and effective n are covered. |
| `test_modularization.R` | ported | `tests/test_glm/test_strategies.py`, `tests/test_glm/test_contrasts.py`, `tests/test_glm/test_chunkwise_engine.py` | Modular solver/strategy/contrast consistency is covered. |
| `test_near_collinear_qr_regression.R` | ported | `tests/test_glm/test_solver.py` | Near-collinear design behavior is covered against a least-squares oracle. |
| `test_numerical_stability.R` | ported | `tests/test_glm/test_solver.py`, `tests/test_robust/test_irls.py` | Ill-conditioned, rank-deficient, extreme-scale, non-finite, and robust edge behavior is covered. |
| `test_robust_args.R` | ported | `tests/test_model/test_config.py`, `tests/test_robust/test_irls.py` | Robust options and fit behavior are covered. |
| `test_robust_components.R` | ported | `tests/test_robust/test_estimators.py`, `tests/test_robust/test_irls.py` | Huber/bisquare weights, scale estimation, and multi-response IRLS behavior are covered. |
| `test_robust_convergence.R` | partial | `tests/test_robust/test_irls.py`, `tests/test_robust/test_estimators.py` | Public robust refit behavior is covered; detailed R convergence-warning matrix is not fully literal. |
| `test_robust_glm_results.R` | ported | `tests/test_robust/test_irls.py`, `tests/test_robust/test_estimators.py` | Clean-data equivalence and outlier downweighting are covered. |
| `test_runwise_ar_fit.R` | ported | `tests/test_ar/test_integration.py`, `tests/test_glm/test_strategies.py` | Runwise AR fit paths are covered. |
| `test_runwise_rankdef.R` | ported | `tests/test_glm/test_strategies.py`, `tests/test_glm/test_solver.py` | Rank-deficient runwise behavior is covered. |
| `test_simulation.R` | ported | `tests/test_simulate/test_simple_dataset.py`, `tests/test_simulate/test_fmri_matrix.py`, `tests/test_simulate/test_compat.py` | Simulation validation, structure, reproducibility, and compatibility helpers are covered. |
| `test_solver_architecture.R` | ported | `tests/test_glm/test_solver.py`, `tests/test_glm/test_strategies.py`, `tests/test_ar/test_integration.py`, `tests/test_robust/test_irls.py` | Solver architecture, WLS, AR integration, rank deficiency, and context-like outputs are covered. |
| `test_solver_edge_cases.R` | ported | `tests/test_glm/test_solver.py`, `tests/test_robust/test_irls.py`, `tests/test_ar/test_estimation.py` | Rank, near-singular, extreme value, robust, short-series, and mixed-solve edge cases are covered. |
| `test_spatial_fdr.R` | ported | `tests/test_stats/test_spatial_fdr.py` | The newer Python spatial-FDR API has direct unit coverage. |
| `test_standardized.R` | ported | `tests/design/test_basis.py`, `tests/design/test_basis_comprehensive.py` | Standardization basis behavior and zero-variance handling are covered by the design basis tests. |
| `test_statistical_inference.R` | partial | `tests/test_glm/test_solver.py`, `tests/test_glm/test_contrasts.py`, `tests/test_stats/test_inference_core.py`, `tests/test_stats/test_ttest.py` | Core SE/statistic/correction behavior is covered; robust sandwich/permutation APIs are narrower in Python. |
| `test_thread_option.R` | scoped_out | none | R `.onLoad` thread option/env-var behavior is not a Python package API. |
| `test_tidy_fmri_lm.R` | ported | `tests/test_accessors.py` | `tidy()` now exposes R-compatible `statistic` while preserving `stat`. |
| `test_trialwise.R` | ported | `tests/design/test_trialwise.py`, `tests/test_single/test_lsa.py`, `tests/test_single/test_lss.py` | Trialwise design and beta-estimation behavior is covered. |
| `test_voxelwise_ar.R` | partial | `tests/test_ar/test_integration.py`, `tests/test_glm/test_chunkwise_engine.py` | Voxelwise AR-oriented public behavior is covered; exact R voxelwise engine internals are partial. |
| `test_voxelwise_ar_robust.R` | partial | `tests/test_robust/test_irls.py`, `tests/test_ar/test_integration.py` | Robust and AR behavior is covered, with exact combined voxelwise R engine scope marked partial. |
| `test_voxelwise_comprehensive.R` | partial | `tests/test_glm/test_chunkwise_engine.py`, `tests/test_ar/test_integration.py`, `tests/test_glm/test_solver.py` | Single-voxel, chunked, constant-voxel, and effective-df behavior are represented in public paths. |
| `test_voxelwise_qr.R` | ported | `tests/test_glm/test_solver.py` | QR/SVD/least-squares oracle behavior is covered in solver tests. |
| `test_write_results.R` | partial | `tests/test_io/test_write_results.py`, `fmrimod/bids/export.py` | Result-writing and BIDS export basics are covered; full R fmristore/HDF5/GDS failure-mode matrix is partial. |

## Intentional Non-Literal Ports

- Python uses explicit NumPy/pandas/dataclass contracts instead of R S3
  `glm_context`, `matrix_dataset`, and plugin classes. Rows that target those
  internals are counted as `ported_pythonized` or `partial` only when public
  Python behavior is executable.
- Live GDS/fmrigds and fmriAR parity depends on external R packages and local
  backend availability. Bridge payloads and optional parity harnesses are
  useful evidence, but skipped external tests are not counted as local proof.
- R CLI and `.onLoad` thread-option behavior are scoped out of the current
  Python public API.
- Benchmark-only tests are not treated as unit-test ports unless there is a
  checked-in Python benchmark/parity harness or a fast invariant test covering
  the same behavior.

## Current Gaps

- No accepted Python `fmrireg_cli` equivalent.
- No literal Python GDS/fmrigds implementation; the supported local contract is
  backend payload construction plus optional external execution.
- Some combined AR + robust + low-rank R internals are represented through
  public Python pipelines and benchmark harnesses rather than one private helper
  per R test.
- Full memory/performance ceilings are benchmark-gate work, not always-on unit
  tests.

## Maintenance Rule

When changing `fmrimod` APIs that correspond to `fmrireg`, update this map in
the same commit as the executable tests. Do not count a row as `ported` unless
there is an executable Python test named in the evidence column.
