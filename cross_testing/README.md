# Cross-Testing Infrastructure

This directory contains the infrastructure for automated cross-testing between the R `fmrihrf` package and the Python `fmrimod` implementation.

## Overview

The cross-testing framework ensures that both implementations produce identical results for equivalent operations. This is critical for:

1. **Validation**: Ensuring the Python port is accurate
2. **Regression Testing**: Catching unintended changes
3. **Documentation**: Providing clear equivalence mappings
4. **Performance**: Comparing execution speed
5. **Fitlins Parity**: Validating first-level OLS parity against fitlins-aligned GLM outputs

## Requirements

- Python 3.8+
- R 4.1+
- `rpy2` Python package
- `nilearn` Python package (required for fitlins parity tests)
- `fmrihrf` R package

## Installation

```bash
# Install Python dependencies
pip install rpy2 pytest pytest-json-report

# Install R dependencies
R -e "install.packages('remotes')"
R -e "remotes::install_github('bbuchsbaum/fmrihrf')"
```

## Running Tests

### All Cross-Tests

```bash
pytest cross_testing/ -v
```

### Specific Test Categories

```bash
# HRF equivalence tests only
pytest cross_testing/test_hrf_equivalence.py -v

# Regressor tests only
pytest cross_testing/test_regressor_equivalence.py -v

# Complex scenarios
pytest cross_testing/test_complex_scenarios.py -v

# Performance benchmarks
pytest cross_testing/test_performance.py -v -m benchmark

# Fitlins OLS parity + speed contract
pytest cross_testing/test_fitlins_parity.py -v
```

### Generate Report

```bash
python cross_testing/generate_report.py

# Fitlins parity + benchmark JSON
python cross_testing/benchmark_fitlins.py \
  --output cross_testing/reports/fitlins_parity_benchmark.json

# Enforce fitlins parity benchmark contract
python cross_testing/check_fitlins_benchmark_contract.py \
  --report cross_testing/reports/fitlins_parity_benchmark.json \
  --min-speedup 1.0

# Create WS01-WS10 GitHub issues from templates
python cross_testing/create_workstream_issues.py \
  --repo owner/repo \
  --apply

# fmrimod optimization-lever benchmark (parallel/cache/float32)
python cross_testing/benchmark_fmrimod_levers.py \
  --output cross_testing/reports/fmrimod_levers_benchmark.json

# AR/ARMA path benchmark (fit_noise + whitening)
python cross_testing/benchmark_arma_paths.py \
  --output cross_testing/reports/fmrimod_arma_benchmark.json

# Threshold check for CI gating
python cross_testing/check_arma_benchmark_thresholds.py \
  --report cross_testing/reports/fmrimod_arma_benchmark.json

# Validate benchmark artifact JSON against canonical schema
python cross_testing/validate_benchmark_schema.py \
  --artifact cross_testing/reports/fmrimod_arma_benchmark.json \
  --schema cross_testing/schemas/core_parity_benchmark.schema.json

# Build core parity matrix artifact (WS01-WS10 computed; WS01/WS02/WS03/WS04/WS05/WS06/WS07/WS08/WS09/WS10 gates)
python cross_testing/benchmark_core_parity_matrix.py \
  --output cross_testing/reports/core_parity_matrix.json \
  --require-ws01-ws02 \
  --require-ws03 \
  --require-ws04 \
  --require-ws05 \
  --require-ws06 \
  --require-ws07 \
  --require-ws08 \
  --require-ws09 \
  --require-ws10

# PR profile gate
python cross_testing/benchmark_core_parity_matrix.py \
  --output cross_testing/reports/core_parity_matrix_pr_profile.json \
  --ws01-n-scans 140 \
  --ws01-tr 1.0 \
  --ws01-seed 7 \
  --ws02-n-timepoints 160 \
  --ws02-n-regressors 8 \
  --ws02-n-voxels 900 \
  --ws02-noise-sd 1.0 \
  --ws02-seed 1234 \
  --ws03-n-timepoints 160 \
  --ws03-n-regressors 8 \
  --ws03-n-voxels 900 \
  --ws03-noise-sd 1.0 \
  --ws03-phi 0.45 \
  --ws03-seed 2026 \
  --ws04-n-timepoints 150 \
  --ws04-n-regressors 8 \
  --ws04-n-voxels 800 \
  --ws04-noise-sd 1.0 \
  --ws04-seed 3030 \
  --ws05-n-timepoints 160 \
  --ws05-n-regressors 8 \
  --ws05-n-voxels 900 \
  --ws05-noise-sd 1.0 \
  --ws05-seed 505 \
  --ws06-n-timepoints 180 \
  --ws06-n-trials 40 \
  --ws06-n-voxels 900 \
  --ws06-n-confounds 6 \
  --ws06-noise-sd 1.0 \
  --ws06-seed 6060 \
  --ws06-repeats 2 \
  --ws06-warmup 0 \
  --ws06-chunk-size 2000 \
  --ws07-n-timepoints 160 \
  --ws07-n-regressors 8 \
  --ws07-n-voxels 900 \
  --ws07-noise-sd 1.0 \
  --ws07-seed 7070 \
  --ws08-n-timepoints 160 \
  --ws08-n-regressors 8 \
  --ws08-n-voxels 900 \
  --ws08-noise-sd 1.0 \
  --ws08-seed 8080 \
  --ws09-n-timepoints 160 \
  --ws09-n-regressors 8 \
  --ws09-n-voxels 900 \
  --ws09-noise-sd 1.0 \
  --ws09-phi 0.45 \
  --ws09-seed 9090 \
  --ws10-n-timepoints 160 \
  --ws10-n-regressors 8 \
  --ws10-n-voxels 900 \
  --ws10-noise-sd 1.0 \
  --ws10-phi 0.45 \
  --ws10-seed 5050 \
  --ws10-repeats 2 \
  --ws10-warmup 0 \
  --ws10-design-n-scans 140 \
  --ws10-design-tr 1.0 \
  --ws10-run-combine-runs 4 \
  --require-ws01-ws02 \
  --require-ws03 \
  --require-ws04 \
  --require-ws05 \
  --require-ws06 \
  --require-ws07 \
  --require-ws08 \
  --require-ws09 \
  --require-ws10

# Nightly profile gate (stricter/larger fixture)
python cross_testing/benchmark_core_parity_matrix.py \
  --output cross_testing/reports/core_parity_matrix_nightly_profile.json \
  --ws01-n-scans 240 \
  --ws01-tr 0.9 \
  --ws01-seed 17 \
  --ws02-n-timepoints 300 \
  --ws02-n-regressors 10 \
  --ws02-n-voxels 2500 \
  --ws02-noise-sd 1.0 \
  --ws02-seed 2234 \
  --ws03-n-timepoints 300 \
  --ws03-n-regressors 10 \
  --ws03-n-voxels 2500 \
  --ws03-noise-sd 1.0 \
  --ws03-phi 0.45 \
  --ws03-seed 3026 \
  --ws04-n-timepoints 260 \
  --ws04-n-regressors 10 \
  --ws04-n-voxels 2200 \
  --ws04-noise-sd 1.0 \
  --ws04-seed 4030 \
  --ws05-n-timepoints 300 \
  --ws05-n-regressors 10 \
  --ws05-n-voxels 2200 \
  --ws05-noise-sd 1.0 \
  --ws05-seed 1505 \
  --ws06-n-timepoints 320 \
  --ws06-n-trials 100 \
  --ws06-n-voxels 2500 \
  --ws06-n-confounds 6 \
  --ws06-noise-sd 1.0 \
  --ws06-seed 7060 \
  --ws06-repeats 3 \
  --ws06-warmup 1 \
  --ws06-chunk-size 4000 \
  --ws07-n-timepoints 300 \
  --ws07-n-regressors 10 \
  --ws07-n-voxels 2500 \
  --ws07-noise-sd 1.0 \
  --ws07-seed 8070 \
  --ws08-n-timepoints 300 \
  --ws08-n-regressors 10 \
  --ws08-n-voxels 2500 \
  --ws08-noise-sd 1.0 \
  --ws08-seed 9080 \
  --ws09-n-timepoints 300 \
  --ws09-n-regressors 10 \
  --ws09-n-voxels 2500 \
  --ws09-noise-sd 1.0 \
  --ws09-phi 0.45 \
  --ws09-seed 10090 \
  --ws10-n-timepoints 300 \
  --ws10-n-regressors 10 \
  --ws10-n-voxels 2500 \
  --ws10-noise-sd 1.0 \
  --ws10-phi 0.45 \
  --ws10-seed 6050 \
  --ws10-repeats 4 \
  --ws10-warmup 1 \
  --ws10-design-n-scans 240 \
  --ws10-design-tr 0.9 \
  --ws10-run-combine-runs 6 \
  --require-ws01-ws02 \
  --require-ws03 \
  --require-ws04 \
  --require-ws05 \
  --require-ws06 \
  --require-ws07 \
  --require-ws08 \
  --require-ws09 \
  --require-ws10
```

## Fitlins Parity Contract

The fitlins parity harness is implemented in:

- `cross_testing/fitlins_parity.py`
- `cross_testing/test_fitlins_parity.py`
- `cross_testing/benchmark_fitlins.py`
- `cross_testing/FITLINS_PARITY_CONTRACT.md`

Current gate focuses on matched first-level OLS GLM outputs (`betas`, `sigma2`, `t`, `p`) and sign-flip rate. This gate should pass before any speed optimization is accepted.

## Test Structure

### `utils.py`
Core utilities including the `REquivalenceTester` base class that handles:
- R package loading
- Array comparison with proper tolerance
- R function calling from Python
- Object attribute comparison

### `test_hrf_equivalence.py`
Tests all HRF types:
- SPM canonical (SPMG1)
- SPM with derivatives (SPMG2, SPMG3)
- Gamma HRF with parameters
- Gaussian HRF
- Mexican hat wavelet
- B-spline basis

### `test_regressor_equivalence.py`
Tests regressor functionality:
- Basic regressor creation
- Event amplitudes and durations
- Summation behavior
- Regressor sets with conditions
- Multi-block designs
- Sparse output

### `test_complex_scenarios.py`
Real-world usage patterns:
- Complete GLM pipelines
- Mixed event types
- HRF libraries
- Multi-run experiments
- Regularized estimation

### `test_performance.py`
Performance comparisons:
- Large-scale evaluations
- Design matrix construction
- HRF evaluation speed
- Sparse vs dense operations

## Adding New Tests

1. Create test method in appropriate test class
2. Use `r_tester` fixture for R interaction
3. Follow the pattern:
   ```python
   def test_new_feature(self, r_tester):
       # Python implementation
       py_result = python_function(args)
       
       # R implementation
       r_vars = r_tester.run_r_code("""
       result <- r_function(args)
       """)
       
       # Compare
       r_tester.compare_arrays(r_vars['result'], py_result)
   ```

## Tolerance Guidelines

Different test types use different numerical tolerances:

- **Default**: rtol=1e-10, atol=1e-12
- **Matrix operations**: rtol=1e-8, atol=1e-10
- **Sparse operations**: rtol=1e-6, atol=1e-8
- **Large computations**: rtol=1e-5, atol=1e-7

## CI/CD Integration

The cross-tests run automatically on:
- Every push to main branch
- Every pull request
- Multiple Python versions (3.8-3.11)
- Multiple R versions (4.1-4.3)

Results are posted as PR comments and artifacts are uploaded for inspection.

AR/ARMA performance gating is tracked via dedicated workflows:
- `.github/workflows/arma-benchmark.yml` (native backend enabled)
- `.github/workflows/arma-benchmark-fallback.yml` (`FMRIMOD_DISABLE_C_ARMA=1`)

## Troubleshooting

### rpy2 Installation Issues

On macOS:
```bash
# Ensure R is in PATH
export PATH="/usr/local/bin:$PATH"

# Install with proper flags
pip install rpy2 --no-binary rpy2
```

On Linux:
```bash
# Install R development headers
sudo apt-get install r-base-dev

pip install rpy2
```

### R Package Not Found

```R
# Install from GitHub
remotes::install_github("bbuchsbaum/fmrihrf")

# Or from local directory
remotes::install_local("/path/to/fmrihrf")
```

### Numerical Differences

Small numerical differences are expected due to:
- Different linear algebra libraries (BLAS/LAPACK)
- Floating point precision
- Algorithm implementations

Adjust tolerances if needed but document why.

## Future Enhancements

1. **Automated Equivalence Discovery**: Scan both codebases to find equivalent functions
2. **Visual Comparison Tools**: Plot R vs Python results side-by-side
3. **Continuous Monitoring**: Track numerical drift over time
4. **Edge Case Generation**: Automatically generate edge cases for testing
5. **Documentation Generation**: Auto-generate equivalence tables from tests
