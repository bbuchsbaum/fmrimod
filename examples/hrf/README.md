# PyFMRIHRF Examples

This directory contains Python tutorials that replicate the functionality of the R `fmrihrf` package vignettes. These examples demonstrate how to use `fmrimod` for fMRI hemodynamic response modeling.

## Tutorials

### 1. [Hemodynamic Response Functions](01_hemodynamic_response.py)
Learn the basics of working with HRFs:
- Using pre-defined HRFs (SPM canonical, Gaussian, etc.)
- Modifying HRF parameters
- Modeling event durations with `block_hrf`
- Controlling normalization and summation
- Creating custom HRF functions

### 2. [Building fMRI Regressors](02_building_regressors.py)
Create and manipulate fMRI regressors:
- Creating regressors from event onsets
- Handling varying event durations and amplitudes
- Comparing different HRF types
- Building multi-condition designs
- Working with multi-block experiments
- Using sparse matrices for efficiency

### 3. [HRF Generators](03_hrf_generators.py)
Work with flexible HRF basis sets:
- Understanding HRF generators
- Creating B-spline and FIR bases
- Comparing basis representations
- Using basis sets in regression models

### 4. [Advanced HRF Modeling](04_advanced_modeling.py)
Explore advanced modeling features:
- Creating HRF libraries for parameter exploration
- Using reconstruction matrices
- Applying regularization with penalty matrices
- Managing complex experimental designs
- Handling multi-run experiments

## Running the Examples

Each example is a standalone Python script that can be run directly:

```bash
python 01_hemodynamic_response.py
```

The scripts include inline comments and print informative output to help you understand each concept.

## Interactive Notebooks

For an interactive experience, you can also use the Jupyter notebook versions:
- `01_hemodynamic_response.ipynb`
- (Additional notebooks can be created as needed)

## Requirements

The examples require the following packages:
- `fmrimod` (this package)
- `numpy`
- `matplotlib`
- `pandas`
- `scipy`

Install all requirements:
```bash
pip install -e .. # Install fmrimod from parent directory
pip install matplotlib pandas
```

## Comparison with R Vignettes

These Python examples provide equivalent functionality to the R vignettes:

| R Vignette | Python Example | Key Topics |
|------------|----------------|------------|
| `a_01_hemodynamic_response.Rmd` | `01_hemodynamic_response.py` | HRF basics, modifications, block designs |
| `a_02_regressor.Rmd` | `02_building_regressors.py` | Event modeling, design matrices |
| `a_03_hrf_generators.Rmd` | `03_hrf_generators.py` | Basis functions, B-splines, FIR |
| `a_04_advanced_modeling.Rmd` | `04_advanced_modeling.py` | HRF libraries, regularization |

## Key Differences from R

While the functionality is equivalent, there are some Pythonic differences:

1. **Function naming**: Snake_case instead of camelCase
   - R: `regressor()` → Python: `regressor()`
   - R: `HRF_SPMG1` → Python: `get_hrf("spmg1")`

2. **Data structures**: NumPy arrays instead of R vectors
   - More explicit array operations
   - Better performance for large datasets

3. **Plotting**: Matplotlib instead of ggplot2
   - Similar capabilities, different syntax
   - More programmatic control

4. **Object-oriented design**: Classes and methods
   - HRF objects have methods and properties
   - Cleaner API for complex operations

## Getting Help

For more information:
- Package documentation: See the main README
- API reference: Use `help(function_name)` in Python
- Issues: Report bugs on GitHub

## Contributing

Feel free to contribute additional examples or improve existing ones!