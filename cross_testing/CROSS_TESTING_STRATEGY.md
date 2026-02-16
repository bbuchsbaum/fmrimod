# R-Python Cross-Testing Strategy for fmrimod

## Overview
Automated cross-testing ensures that R `fmrihrf` and Python `fmrimod` produce identical results for equivalent operations.

## Architecture Options

### 1. Python-Driven with rpy2 (Recommended)
Use Python as the main test runner, calling R code through `rpy2`:

**Pros:**
- Single test framework (pytest)
- Easy comparison of numpy arrays and R matrices
- Better error handling and debugging
- Can reuse existing Python test infrastructure

**Cons:**
- Requires rpy2 installation
- Some R edge cases might be harder to test

### 2. R-Driven with reticulate
Use R as the main test runner, calling Python through `reticulate`:

**Pros:**
- Natural for R users
- Direct access to R's testing tools
- Good for R-specific edge cases

**Cons:**
- Less familiar Python error messages
- Harder to integrate with Python CI/CD

### 3. Separate Scripts with File Exchange
Run R and Python separately, exchange data via files:

**Pros:**
- No inter-language dependencies
- Can run on different machines
- Clear separation of concerns

**Cons:**
- Slower (file I/O)
- More complex orchestration
- Harder to debug failures

### 4. Docker-Based Testing
Use containers with both R and Python:

**Pros:**
- Reproducible environment
- Version control for both languages
- Good for CI/CD

**Cons:**
- More complex setup
- Slower test execution

## Recommended Implementation (Python + rpy2)

### Directory Structure
```
cross_testing/
├── __init__.py
├── conftest.py              # Shared fixtures
├── utils.py                 # Helper functions
├── test_hrf_equivalence.py  # HRF cross-tests
├── test_regressor_equivalence.py
├── test_sampling_equivalence.py
├── data/                    # Shared test data
│   ├── generate_test_data.R
│   └── test_cases.json
└── reports/                 # Comparison reports
```

### Core Testing Utilities

```python
# cross_testing/utils.py
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr

# Auto-convert between R and Python types
pandas2ri.activate()
numpy2ri.activate()

class REquivalenceTester:
    """Base class for R-Python equivalence testing."""
    
    def __init__(self):
        # Import R packages
        self.base = importr('base')
        self.fmrihrf = importr('fmrihrf')
        
    def compare_arrays(self, r_result, py_result, rtol=1e-10, atol=1e-12):
        """Compare R and Python arrays."""
        # Convert R object to numpy
        r_array = np.array(r_result)
        
        # Handle different shapes (R is column-major, Python is row-major)
        if r_array.ndim == 2 and py_result.ndim == 2:
            if r_array.shape[0] == py_result.shape[1] and r_array.shape[1] == py_result.shape[0]:
                r_array = r_array.T
        
        np.testing.assert_allclose(r_array, py_result, rtol=rtol, atol=atol)
        
    def compare_objects(self, r_obj, py_obj, attributes):
        """Compare R and Python objects by attributes."""
        for attr in attributes:
            r_val = self.get_r_attribute(r_obj, attr)
            py_val = getattr(py_obj, attr)
            self.compare_arrays(r_val, py_val)
    
    def get_r_attribute(self, r_obj, attr):
        """Extract attribute from R object."""
        return ro.r[f"attr({r_obj}, '{attr}')"]
```

### Example Cross-Tests

```python
# cross_testing/test_hrf_equivalence.py
import pytest
import numpy as np
from fmrimod import get_hrf, gamma_hrf, spm_canonical
from .utils import REquivalenceTester

class TestHRFEquivalence(REquivalenceTester):
    """Test HRF equivalence between R and Python."""
    
    def test_spm_canonical(self):
        """Test SPM canonical HRF."""
        # Python version
        t = np.linspace(0, 30, 100)
        py_result = spm_canonical(t)
        
        # R version
        r_code = """
        t <- seq(0, 30, length.out=100)
        hrf <- hrf_spmg1(t)
        """
        ro.r(r_code)
        r_result = ro.r['hrf']
        
        # Compare
        self.compare_arrays(r_result, py_result)
    
    def test_gamma_hrf_parameters(self):
        """Test gamma HRF with various parameters."""
        test_cases = [
            {'shape': 6, 'rate': 1},
            {'shape': 4, 'rate': 0.9},
            {'shape': 8, 'rate': 1.2},
        ]
        
        t = np.linspace(0, 30, 100)
        
        for params in test_cases:
            # Python
            py_result = gamma_hrf(t, **params)
            
            # R
            ro.r.assign('t', t)
            ro.r.assign('shape', params['shape'])
            ro.r.assign('rate', params['rate'])
            r_result = ro.r('hrf_gamma(t, shape=shape, rate=rate)')
            
            # Compare
            self.compare_arrays(r_result, py_result, rtol=1e-8)
    
    @pytest.mark.parametrize("hrf_name", [
        "spmg1", "spmg2", "spmg3", "gamma", "gaussian"
    ])
    def test_hrf_library(self, hrf_name):
        """Test all HRF types in library."""
        # Python
        py_hrf = get_hrf(hrf_name)
        t = np.linspace(0, py_hrf.span, 100)
        py_result = py_hrf(t)
        
        # R
        r_hrf = self.fmrihrf.get_hrf(hrf_name.upper())
        r_result = ro.r.evaluate(r_hrf, t)
        
        # Compare
        self.compare_arrays(r_result, py_result)
```

### Regressor Cross-Tests

```python
# cross_testing/test_regressor_equivalence.py
class TestRegressorEquivalence(REquivalenceTester):
    """Test regressor equivalence."""
    
    def test_basic_regressor(self):
        """Test basic regressor creation and evaluation."""
        onsets = [10, 30, 50]
        duration = 2.0
        
        # Python
        from fmrimod import regressor, SamplingFrame
        py_reg = regressor(
            onsets=onsets,
            hrf="spmg1",
            duration=duration
        )
        sf = SamplingFrame(blocklens=100, tr=2.0)
        py_result = py_reg.evaluate(sf.samples)
        
        # R
        ro.r.assign('onsets', onsets)
        ro.r.assign('duration', duration)
        r_code = """
        reg <- regressor(
            onsets = onsets,
            hrf = HRF_SPMG1,
            duration = duration
        )
        sf <- sampling_frame(100, TR = 2.0)
        result <- evaluate(reg, samples(sf))
        """
        ro.r(r_code)
        r_result = ro.r['result']
        
        # Compare
        self.compare_arrays(r_result, py_result, rtol=1e-10)
```

### Complex Scenario Testing

```python
# cross_testing/test_complex_scenarios.py
class TestComplexScenarios(REquivalenceTester):
    """Test complex real-world scenarios."""
    
    def test_full_glm_pipeline(self):
        """Test complete GLM pipeline."""
        # Generate synthetic fMRI data
        n_scans = 200
        tr = 2.0
        
        # Create events
        event_times = np.array([10, 30, 50, 70, 90, 110, 130, 150])
        conditions = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Python pipeline
        from fmrimod import regressor_set, SamplingFrame
        
        sf = SamplingFrame(blocklens=n_scans, tr=tr)
        py_rset = regressor_set(
            onsets=event_times,
            fac=conditions,
            hrf="spmg1"
        )
        py_design = py_rset.evaluate(sf.samples)
        
        # R pipeline
        ro.r.assign('event_times', event_times)
        ro.r.assign('conditions', conditions)
        ro.r.assign('n_scans', n_scans)
        ro.r.assign('tr', tr)
        
        r_code = """
        sf <- sampling_frame(n_scans, TR = tr)
        rset <- regressor_set(
            onsets = event_times,
            fac = as.factor(conditions),
            hrf = HRF_SPMG1
        )
        design <- regressor_design(rset, sf)
        """
        ro.r(r_code)
        r_design = ro.r['design']
        
        # Compare design matrices
        self.compare_arrays(r_design, py_design, rtol=1e-10)
```

### Data Generation for Testing

```r
# cross_testing/data/generate_test_data.R
library(fmrihrf)
library(jsonlite)

# Generate test cases
generate_test_cases <- function() {
  test_cases <- list()
  
  # HRF evaluations
  t <- seq(0, 30, length.out = 100)
  test_cases$hrf_evaluations <- list(
    spmg1 = list(
      t = t,
      result = hrf_spmg1(t)
    ),
    gamma = list(
      t = t,
      params = list(shape = 6, rate = 1),
      result = hrf_gamma(t, shape = 6, rate = 1)
    )
  )
  
  # Regressor examples
  sf <- sampling_frame(100, TR = 2.0)
  reg <- regressor(
    onsets = c(10, 30, 50),
    hrf = HRF_SPMG1,
    duration = 2
  )
  
  test_cases$regressor_example <- list(
    sampling_frame = list(
      blocklens = 100,
      TR = 2.0
    ),
    regressor = list(
      onsets = c(10, 30, 50),
      duration = 2
    ),
    result = evaluate(reg, samples(sf))
  )
  
  # Save as JSON
  write_json(test_cases, "test_cases.json", pretty = TRUE, auto_unbox = TRUE)
}

generate_test_cases()
```

### CI/CD Integration

```yaml
# .github/workflows/cross_test.yml
name: Cross-Language Tests

on: [push, pull_request]

jobs:
  cross-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Set up R
      uses: r-lib/actions/setup-r@v2
      with:
        r-version: '4.1'
    
    - name: Install R dependencies
      run: |
        install.packages('remotes')
        remotes::install_github('bbuchsbaum/fmrihrf')
      shell: Rscript {0}
    
    - name: Install Python dependencies
      run: |
        pip install -e .
        pip install rpy2 pytest
    
    - name: Run cross-tests
      run: |
        pytest cross_testing/ -v --tb=short
    
    - name: Generate comparison report
      if: always()
      run: |
        python cross_testing/generate_report.py
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v2
      if: always()
      with:
        name: cross-test-report
        path: cross_testing/reports/
```

### Tolerance and Numerical Precision

```python
# cross_testing/conftest.py
import pytest

@pytest.fixture
def numerical_tolerance():
    """Define acceptable numerical tolerances."""
    return {
        'rtol': 1e-10,  # Relative tolerance
        'atol': 1e-12,  # Absolute tolerance
        'matrix_rtol': 1e-8,  # For larger matrices
        'sparse_rtol': 1e-6,  # For sparse operations
    }

@pytest.fixture(scope='session')
def r_session():
    """Create persistent R session."""
    import rpy2.robjects as ro
    ro.r('library(fmrihrf)')
    return ro.r
```

### Performance Comparison

```python
# cross_testing/test_performance.py
import time
import pytest
from .utils import REquivalenceTester

class TestPerformance(REquivalenceTester):
    """Compare performance between implementations."""
    
    @pytest.mark.benchmark
    def test_large_scale_evaluation(self):
        """Test performance with large datasets."""
        n_events = 1000
        n_timepoints = 10000
        
        # Generate data
        onsets = np.sort(np.random.uniform(0, 5000, n_events))
        times = np.linspace(0, 5000, n_timepoints)
        
        # Python timing
        py_start = time.time()
        py_reg = regressor(onsets=onsets, hrf="spmg1")
        py_result = py_reg.evaluate(times)
        py_time = time.time() - py_start
        
        # R timing
        ro.r.assign('onsets', onsets)
        ro.r.assign('times', times)
        r_start = time.time()
        ro.r("""
        reg <- regressor(onsets = onsets, hrf = HRF_SPMG1)
        result <- evaluate(reg, times)
        """)
        r_time = time.time() - r_start
        
        # Report
        print(f"\nPerformance Comparison:")
        print(f"Python: {py_time:.3f}s")
        print(f"R: {r_time:.3f}s")
        print(f"Ratio: {r_time/py_time:.2f}x")
        
        # Still verify equivalence
        r_result = ro.r['result']
        self.compare_arrays(r_result, py_result, rtol=1e-8)
```

## Benefits of This Approach

1. **Automated**: Runs on every commit
2. **Comprehensive**: Tests all major functions
3. **Maintainable**: Single test framework
4. **Debuggable**: Clear error messages
5. **Extensible**: Easy to add new test cases
6. **Performance Aware**: Tracks relative performance

## Next Steps

1. Set up rpy2 environment
2. Create initial test cases for core functions
3. Add to CI/CD pipeline
4. Generate regular compatibility reports
5. Monitor for numerical drift over time