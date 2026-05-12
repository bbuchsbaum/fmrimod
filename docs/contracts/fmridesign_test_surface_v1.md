# fmridesign Test Surface v1

Status: Active coverage map

## Purpose

This file maps the R `fmridesign` `tests/testthat` surface in
`~/code/fmridesign` to the Python `fmrimod` design, event, HRF, validation,
contrast, and visualization tests. It is not a claim that every R S3 internal
or snapshot fixture has a literal Python twin. It records which tests are
ported directly, covered by Pythonized public contracts, or scoped out because
they target R-only helper fixtures or snapshot infrastructure.

## Summary

Observed R testthat artifacts in the local `~/code/fmridesign` checkout: 32.

- R files: 30.
- Executable R test files: 29.
- Executable `test_that()` blocks: 611.
- Helper/data/snapshot artifacts: 3.
- Direct or Pythonized coverage exists for event classes and terms, event
  models, baseline models, basis classes, sampling frames, HRF formulas and
  generators, convolution, contrasts, validation, residualization, column maps,
  visualization helpers, naming helpers, and extension-registry behavior.
- R-only helper files and `_snaps` snapshot data are not Python public APIs.
- The strongest executable anchors are `tests/design`, `tests/hrf`, and
  fixture-backed R parity checks in `tests/design/test_r_parity.py` and
  `tests/design/test_rpy2_parity.py`.

Verification snapshot on 2026-05-12:

- `pytest tests/design`: 1323 passed.
- `pytest tests/hrf`: 243 passed.
- `pytest tests/design/test_rpy2_parity.py tests/design/test_r_parity.py`:
  63 passed.

## File Map

| R artifact | Python coverage status | Python evidence | Notes |
| --- | --- | --- | --- |
| `_snaps` | scoped_out | none | R snapshottest output directory; Python uses explicit assertions and generated JSON fixtures instead. |
| `events_testdat.txt` | scoped_out | none | R test fixture file, not a Python public API. |
| `helper-naming.R` | scoped_out | none | R test helper only; naming behavior is covered by executable Python naming tests. |
| `test-baseline_model_methods.R` | ported | `tests/design/test_baseline_model.py`, `tests/design/test_baseline_specs.py`, `tests/design/test_validation.py`, `tests/design/test_visualization.py` | Baseline constructors, term matrices, block/global intercepts, design matrices, cells, colmaps, and plotting paths are covered. |
| `test-basis_classes.R` | ported | `tests/design/test_basis.py`, `tests/design/test_basis_comprehensive.py`, `tests/design/test_basis_functions.py`, `tests/design/test_as_hrf.py` | Polynomial, spline, transform, HRF basis, naming, and evaluation contracts are covered. |
| `test-contrast_weights.R` | ported | `tests/design/test_contrast_weights_coverage.py`, `tests/design/test_contrast_system.py`, `tests/design/test_contrast.py`, `tests/design/test_fcontrast_system.py` | Unit, pair, oneway, interaction, column, basis-filtered, and weighted contrast contracts are covered. |
| `test-convolution-fixes.R` | ported | `tests/design/test_convolve.py`, `tests/design/test_convolve_design.py`, `tests/design/test_event_model.py`, `tests/design/test_rpy2_parity.py` | Duration propagation, block boundaries, multi-run convolution, normalization, and dimension regressions are covered. |
| `test-covariate.R` | ported | `tests/design/test_covariate.py`, `tests/design/test_covariate_trialwise_comprehensive.py`, `tests/design/test_events.py` | Continuous covariates, centering/scaling, trialwise terms, and event-variable behavior are covered. |
| `test-design_colmap.R` | ported | `tests/design/test_validation.py`, `tests/design/test_event_model.py` | Event-model and baseline-model column metadata contracts are covered. |
| `test-design_map.R` | ported | `tests/design/test_visualization.py`, `tests/design/test_visualization_comprehensive.py`, `tests/design/test_plot_contrasts.py` | Design maps, correlation maps, contrast plots, and option paths are covered with matplotlib/plotly objects rather than ggplot objects. |
| `test-event_classes.R` | ported | `tests/design/test_events.py`, `tests/design/test_event_cells.py`, `tests/design/test_generics.py`, `tests/design/test_dispatch_event_generics.py` | Event factor, variable, matrix, basis, levels, cells, elements, labels, columns, and type predicates are covered. |
| `test-event_model.R` | ported | `tests/design/test_event_model.py`, `tests/design/test_event_model_contrasts.py` | Basic event-model construction and contrast accessors are covered. |
| `test-event_model_methods.R` | ported | `tests/design/test_event_model.py`, `tests/design/test_generics.py`, `tests/design/test_dispatch_columns.py`, `tests/design/test_event_term_coverage.py` | Event-model accessors, term names, columns, conditions, contrast weights, and model methods are covered. |
| `test-event_vector_methods.R` | ported_pythonized | `tests/design/test_event_term_coverage.py`, `tests/design/test_generics.py`, `tests/design/test_dispatch_event_generics.py`, `tests/design/test_exports.py` | Python exposes event/term objects plus generic accessors rather than a literal R `event_vector` S3 class. Public accessor behavior is covered. |
| `test-extension_registry.R` | ported | `tests/design/test_extension_registry.py` | Registration, validation, object class lookup, AFNI fallback, unique function aggregation, and list/info accessors are covered. |
| `test-hrf_formula.R` | ported | `tests/design/test_convolve_formula_hrf.py`, `tests/design/test_hrf_integration.py`, `tests/hrf/test_hrf_generators.py` | HRF formula parsing, `hrf()`, `trialwise()`, duration handling, and generator integration are covered. |
| `test-naming-utils.R` | ported | `tests/design/test_naming.py`, `tests/design/test_term_utils.py`, `tests/design/test_event_model.py` | Sanitization, stable term tags, clash suffixes, basis suffixes, and golden heading tests are covered. |
| `test-normalize-regressors.R` | ported | `tests/design/test_covariate_trialwise_comprehensive.py`, `tests/design/test_convolve.py`, `tests/design/test_convolve_formula_hrf.py` | Peak normalization across trialwise and condition-level regressors is covered, including zero-column guards. |
| `test-residualize.R` | ported | `tests/design/test_validate_residualize.py`, `tests/design/test_validation.py` | Matrix/DataFrame/EventModel residualization, column subsets, row mismatch, and QR oracle behavior are covered. |
| `test-sampling_frame.R` | ported | `tests/hrf/test_sampling_frame.py`, `tests/design/test_exports.py`, `tests/design/test_event_model.py` | Sampling frame construction, samples, acquisition onsets, block lengths, block IDs, and global-onset semantics are covered. |
| `test-validate.R` | ported | `tests/design/test_validate_residualize.py`, `tests/design/test_validation.py` | Contrast validation, estimability checks, collinearity checks, threshold/tolerance validation, and EventModel inputs are covered. |
| `test_baseline.R` | ported | `tests/design/test_baseline_model.py`, `tests/design/test_baseline_specs.py`, `tests/design/test_r_parity.py` | Baseline constructor and fixture-backed matrix contracts are covered. |
| `test_basic_functionality.R` | ported | `tests/design/test_imports.py`, `tests/design/test_event_model.py`, `tests/design/test_design_matrix.py`, `tests/design/test_r_parity.py` | Basic imports, model construction, design matrix generation, and fixture parity are covered. |
| `test_condition_basis_list.R` | ported | `tests/design/test_basis_comprehensive.py` | Condition-basis list and matrix outputs, multi-basis HRFs, and invalid output mode are covered. |
| `test_contrast.R` | ported | `tests/design/test_contrast.py`, `tests/design/test_contrast_diff_spec.py`, `tests/design/test_contrast_system.py`, `tests/design/test_fcontrast.py`, `tests/design/test_fcontrast_system.py` | t/F contrasts, formula contrasts, pairwise/one-against-all contrasts, interactions, basis filters, and basis weights are covered. |
| `test_covariate_length.R` | ported | `tests/design/test_covariate.py`, `tests/design/test_covariate_trialwise_comprehensive.py`, `tests/design/test_events.py` | Covariate/event length validation and trialwise data length checks are covered. |
| `test_event_model.R` | ported | `tests/design/test_event_model.py`, `tests/design/test_rpy2_parity.py`, `tests/design/test_r_parity.py` | Formula/list interfaces, mixed bases, interactions, naming, subsets, block labels, and fixture parity are covered. |
| `test_event_vector.R` | ported_pythonized | `tests/design/test_events.py`, `tests/design/test_event_cells.py`, `tests/design/test_event_term_coverage.py`, `tests/design/test_generics.py` | Python has typed event and term classes instead of a literal R event-vector constructor. Public behavior is covered through Python event APIs. |
| `test_hrf.R` | ported | `tests/hrf/test_hrf_core.py`, `tests/hrf/test_hrf_generators.py`, `tests/hrf/test_hrf_registry.py`, `tests/hrf/test_derivatives_module.py`, `tests/hrf/test_r_equivalence.py`, `tests/design/test_hrf_integration.py` | HRF objects, canonical HRFs, decorators, binding, normalization, lag/block behavior, and registry paths are covered. |
| `test_hrf_generator.R` | ported | `tests/design/test_convolve_formula_hrf.py`, `tests/hrf/test_hrf_generators.py`, `tests/hrf/test_trial_varying.py` | Per-event HRF generators, boxcar/weighted generators, validation, recycling, and multi-block behavior are covered. |
| `test_parse_event_formula.R` | ported | `tests/design/test_formula.py`, `tests/design/test_convolve_formula_hrf.py` | Formula parsing and HRF formula terms are covered by the Python parser API. |
| `test_parse_term.R` | ported | `tests/design/test_formula.py`, `tests/design/test_convolve_formula_hrf.py` | Term parsing and parsed-term evaluation are covered by parser tests. |

## Intentional Non-Literal Ports

- There are no known unported executable `fmridesign` public-contract tests in
  this map.
- There is no literal Python `event_vector` S3 class. The supported Python
  surface is typed event objects, event terms, and generic accessors, and the
  R accessor behavior is covered through those APIs.
- R snapshot output under `_snaps` is not ported; Python asserts concrete
  object structure, arrays, labels, and plot object types directly.
- R helper/data files are not counted as public APIs.
- The extension registry intentionally accepts registered class-name strings
  for `requires_external_processing()`. The upstream R test expects `FALSE`
  for that string path because of S3 `class()` dispatch; Python keeps class-name
  registry lookup consistent across registry helpers.

## Maintenance Rule

When changing design/event/HRF/contrast APIs that correspond to `fmridesign`,
update this map in the same commit as the executable tests. Do not count a row
as `ported` unless there is an executable Python test named in the evidence
column.
