# fmrihrf Test Port Audit v1

Objective: every test in `~/code/fmrihrf/tests/testthat` has a Python
counterpart in `fmrimod` and the ported suite passes.

Upstream inventory: 15 upstream `testthat` files containing 120 `test_that()`
blocks.

- `test-cli.R` (9 blocks): ported in `tests/hrf/test_fmrihrf_cli.py`.
- `test_acquisition_onsets.R` (7 blocks): covered by
  `tests/hrf/test_sampling_frame.py` plus
  exact edge-case ports in `tests/hrf/test_fmrihrf_ported_edge_cases.py`.
- `test_deriv.R` (6 blocks): covered by `tests/hrf/test_derivatives_module.py`,
  `tests/hrf/test_hrf_core.py`, and exact analytic/edge-case ports in
  `tests/hrf/test_fmrihrf_ported_edge_cases.py`.
- `test_fft_and_toeplitz.R` (2 blocks): covered by
  `tests/hrf/test_r_equivalence.py`.
- `test_hrf.R` (55 blocks): covered by `tests/hrf/test_hrf_core.py`,
  `tests/hrf/test_hrf_decorators.py`, `tests/hrf/test_hrf_generators.py`,
  `tests/hrf/test_new_hrfs.py`, `tests/hrf/test_trial_varying.py`, and
  `tests/hrf/test_r_equivalence.py`.
- `test_hrf_from_coefficients.R` (1 block): covered by
  `tests/hrf/test_hrf_core.py` and `tests/hrf/test_r_equivalence.py`.
- `test_hrf_library.R` (2 blocks): covered by `tests/hrf/test_r_equivalence.py`.
- `test_hrf_lwu_extra.R` (2 blocks): ported in
  `tests/hrf/test_fmrihrf_ported_edge_cases.py`.
- `test_penalty_matrix.R` (4 blocks): covered by
  `tests/hrf/test_r_equivalence.py` plus Fourier/default penalty ports in
  `tests/hrf/test_fmrihrf_ported_edge_cases.py`.
- `test_reg_constructor.R` (3 blocks): covered by `tests/hrf/test_regressor.py`
  and `tests/hrf/test_r_equivalence.py`.
- `test_regressor.R` (15 blocks): covered by `tests/hrf/test_regressor.py`,
  `tests/hrf/test_trial_varying.py`, and `tests/hrf/test_r_equivalence.py`.
- `test_regressor_design.R` (2 blocks): covered by
  `tests/hrf/test_regressor.py` and `tests/hrf/test_fmrihrf_cli.py`.
- `test_regressor_set.R` (2 blocks): covered by `tests/hrf/test_regressor.py`
  and `tests/hrf/test_r_equivalence.py`.
- `test_sampling_frame.R` (9 blocks): covered by
  `tests/hrf/test_sampling_frame.py`.
- `test_utils.R` (1 block): covered by `tests/hrf/test_r_equivalence.py`.

Known implementation changes made to satisfy the audited tests:

- Added `fmrimod.cli.fmrihrf_cli()` plus `fmrihrf`/`fmrimod` console-script
  entry points so the R CLI contract has a Python runtime surface.
- Added `install_cli()` wrapper installation parity for local command wrappers.
- Matched the R LWU `normalize = "area"` behavior: warn and return the
  unnormalised response.
- Fixed generator-named Fourier/FIR/Tent/B-spline/Daguerre penalty dispatch so
  generated HRFs receive the same penalty family as canonical library objects.

Verification commands:

- `pytest tests/hrf`
- `pytest cross_testing/test_hrf_equivalence.py cross_testing/test_regressor_equivalence.py`
