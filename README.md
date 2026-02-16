# fmrimod

fMRI Signal Modeling: HRFs, Design Matrices, and Regression.

A unified Python library for fMRI experimental design and signal modeling,
combining hemodynamic response function (HRF) specification, design matrix
construction, and regression tools.

## Installation

```bash
pip install -e .
```

## Subpackages

- `fmrimod.hrf` - HRF basis functions, registry, decorators, and generators
- `fmrimod.regressor` - Event-related regressors and convolution
- `fmrimod.events` - Event representations (factor, variable, matrix, basis)
- `fmrimod.formula` - R-style formula parsing and DSL
- `fmrimod.basis` - Parametric basis functions (polynomial, spline, transforms)
- `fmrimod.contrast` - Contrast specification and F-tests
- `fmrimod.baseline` - Baseline and nuisance models
- `fmrimod.design` - Design matrix assembly and EventModel
- `fmrimod.visualization` - Design matrix plotting

## Quick Start

```python
import fmrimod

# Create an HRF
hrf = fmrimod.SPM_CANONICAL

# Build a regressor
reg = fmrimod.regressor(
    onsets=[1.0, 5.0, 10.0],
    hrf=hrf,
)

# Evaluate on a sampling frame
sf = fmrimod.SamplingFrame(blocklens=100, TR=2.0)
signal = reg.eval_regressor(sf)
```

## Lineage

This package unifies two Python ports of R neuroimaging packages:

- **fmrihrf** (R) -> `fmrimod.hrf` + `fmrimod.regressor`
- **fmridesign** (R) -> `fmrimod.events` + `fmrimod.design` + `fmrimod.contrast` + ...

## License

GPL-3.0-or-later
