# Cross-Testing Infrastructure

This directory contains the infrastructure for automated cross-testing between the R `fmrihrf` package and the Python `fmrimod` implementation.

## Overview

The cross-testing framework ensures that both implementations produce identical results for equivalent operations. This is critical for:

1. **Validation**: Ensuring the Python port is accurate
2. **Regression Testing**: Catching unintended changes
3. **Documentation**: Providing clear equivalence mappings
4. **Performance**: Comparing execution speed

## Requirements

- Python 3.8+
- R 4.1+
- `rpy2` Python package
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
```

### Generate Report

```bash
python cross_testing/generate_report.py
```

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