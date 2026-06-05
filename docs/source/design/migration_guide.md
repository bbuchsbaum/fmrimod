# Migration Guide: From fmridesign (R) to fmrimod (Python)

This guide helps users transition from the R `fmridesign` package to its Python equivalent `fmrimod`. While the packages share similar functionality, there are important differences in syntax and conventions.

## Overview

The `fmrimod` package provides a Python implementation of the R `fmridesign` package for creating design matrices for fMRI analysis. The core functionality remains the same, but the implementation leverages Python's strengths and ecosystem.

## Key Differences

### 1. Formula Syntax

**R (NSE - Non-Standard Evaluation):**
```r
model <- event_model(onset ~ condition * block, 
                     data = events_df,
                     sampling_frame = sframe)
```

**Python (String formulas):**
```python
model = event_model("onset ~ condition * block", 
                    data=events_df,
                    sampling_frame=sframe)
```

### 2. Factor Handling

**R (automatic factor conversion):**
```r
events$condition <- factor(c("A", "B", "A", "B"))
```

**Python (explicit categorical):**
```python
events['condition'] = pd.Categorical(["A", "B", "A", "B"])
# Or use EventFactor directly
factor = EventFactor(
    onsets=[1, 2, 3, 4],
    durations=[2, 2, 2, 2],
    values=["A", "B", "A", "B"],
    levels=["A", "B"],
    name="condition"
)
```

### 3. S3 Generics vs singledispatch

**R (S3 generics):**
```r
# Generic function
design_matrix <- function(x, ...) UseMethod("design_matrix")

# Method for EventModel
design_matrix.EventModel <- function(x, ...) {
  # implementation
}
```

**Python (singledispatch):**
```python
from functools import singledispatch

@singledispatch
def design_matrix(x, **kwargs):
    raise NotImplementedError(f"design_matrix not implemented for {type(x)}")

@design_matrix.register(EventModel)
def _design_matrix_event_model(model, **kwargs):
    # implementation
```

### 4. NA/Missing Value Handling

**R (NA values):**
```r
values <- c(1.5, NA, 2.0, 3.5)
# NA propagation is automatic
```

**Python (pandas NA or numpy nan):**
```python
import pandas as pd
import numpy as np

values = pd.Series([1.5, pd.NA, 2.0, 3.5])
# or
values = np.array([1.5, np.nan, 2.0, 3.5])
```

## Common Function Mappings

### Event Creation

| R Function | Python Equivalent | Notes |
|------------|-------------------|-------|
| `event_factor()` | `EventFactor()` | Requires explicit `levels` argument |
| `event_variable()` | `EventVariable()` | Auto-centers by default (`center=True`) |
| `event_matrix()` | `EventMatrix()` | Column names required |
| `event_basis()` | `EventBasis()` | Similar usage |
| `event_table()` | `event_table()` | Returns DataFrame instead of data.table |

### Parametric modulators

The typed-spec equivalent of R `fmrireg`'s `trial_type + trial_type:rt`
parametric-modulator pattern is the `modulators=` keyword on `hrf(...)`:

```python
from fmrimod.spec import hrf
spec = hrf("trial_type", modulators=("rt",))
```

**Centering default differs from R fmrireg**: fmrimod centers each
modulator at the input-variable level by default
(`center_modulators=True`), matching `EventVariable(center=True)` and
the post-Mumford modern-correct workflow. R `fmrireg` historically
did not center modulators. For exact R-output reproducibility on
legacy fixtures, pass `center_modulators=False`:

```python
spec_r_compat = hrf("trial_type", modulators=("rt",),
                    center_modulators=False)
```

See [Parametric modulators](../../tutorials/parametric-modulators.qmd)
for the full discussion (semantics, the four entry points, why no
automatic orthogonalization).

### Model Building

| R Function | Python Equivalent | Notes |
|------------|-------------------|-------|
| `event_model()` | `event_model()` | String formulas instead of NSE |
| `baseline_model()` | `baseline_model()` | Similar usage |
| `design_matrix()` | `design_matrix()` | Returns numpy array |

### Contrasts

| R Function | Python Equivalent | Notes |
|------------|-------------------|-------|
| `contrast()` | `contrast()` | String formulas for expressions |
| `pair_contrast()` | `pair_contrast()` | Same syntax |
| `oneway_contrast()` | `oneway_contrast()` | Same behavior |
| `Fcontrasts()` | `Fcontrasts()` | Returns dict instead of list |

### Utilities

| R Function | Python Equivalent | Notes |
|------------|-------------------|-------|
| `cells()` | `cells()` | Returns pandas DataFrame |
| `conditions()` | `conditions()` | Uses dots instead of brackets in names |
| `onsets()` | `onsets()` | Returns numpy array |
| `durations()` | `durations()` | Returns numpy array |
| `is.categorical()` | `is_categorical()` | Python naming convention |
| `is.continuous()` | `is_continuous()` | Python naming convention |

## Example Workflows

### 1. Basic Event Model

**R:**
```r
library(fmridesign)

# Create events
events <- data.frame(
  onset = c(1, 5, 10, 15),
  duration = c(2, 2, 2, 2),
  condition = factor(c("A", "B", "A", "B"))
)

# Create sampling frame
sframe <- sampling_frame(blocklens = c(100), tr = 2.0)

# Build model
model <- event_model(onset ~ condition,
                     data = events,
                     sampling_frame = sframe)

# Get design matrix
X <- design_matrix(model)
```

**Python:**
```python
import fmrimod as pfd
import pandas as pd

# Create events
events = pd.DataFrame({
    'onset': [1, 5, 10, 15],
    'duration': [2, 2, 2, 2],
    'condition': pd.Categorical(['A', 'B', 'A', 'B'])
})

# Create sampling frame
sframe = pfd.sampling_frame(blocklens=[100], tr=2.0)

# Build model
model = pfd.event_model("onset ~ condition",
                        data=events,
                        sampling_frame=sframe)

# Get design matrix
X = pfd.design_matrix(model)
```

### 2. Contrasts

**R:**
```r
# Create contrasts
c1 <- contrast(model, condition[A] - condition[B])
weights <- contrast_weights(c1)

# F-contrasts
fcons <- Fcontrasts(model)
```

**Python:**
```python
# Create contrasts
c1 = pfd.contrast(model, "condition[A] - condition[B]")
weights = pfd.contrast_weights(c1)

# F-contrasts
fcons = pfd.Fcontrasts(model)
```

### 3. Working with Event Terms

**R:**
```r
# Extract information
event_cells <- cells(model)
event_conditions <- conditions(model)
event_onsets <- onsets(model)
```

**Python:**
```python
# Extract information
event_cells = pfd.cells(model)
event_conditions = pfd.conditions(model)
event_onsets = pfd.onsets(model)
```

## Tips for Migration

1. **Import Style**: Use `import fmrimod as pfd` for cleaner code
2. **DataFrames**: Use pandas DataFrames instead of R data.frames
3. **Arrays**: Numpy arrays replace R matrices/vectors
4. **Naming**: Python uses snake_case instead of camelCase
5. **Indexing**: Python uses 0-based indexing vs R's 1-based
6. **String Formulas**: Always quote formula expressions in Python

## Feature Parity Status

As of January 2025, fmrimod implements approximately 80% of fmridesign's functionality:

### Fully Implemented
- Core event types (EventFactor, EventVariable, EventMatrix, EventBasis)
- Event models and design matrix generation
- Baseline models
- Contrast specifications and weights
- HRF convolution (via fmrimod)
- Generic extraction functions (cells, conditions, onsets, etc.)
- AFNI integration

### Not Yet Implemented
- Some specialized extraction functions (labels, levels, columns)
- Direct HRF function (use get_hrf instead)
- Some basis prediction methods
- Condition-specific basis lists

### Python-Specific Features
- Better integration with scientific Python ecosystem (numpy, pandas, matplotlib)
- Type hints for better IDE support
- More flexible plotting with matplotlib
- Easier integration with nilearn and other neuroimaging tools

## Getting Help

- **Documentation**: https://fmrimod.readthedocs.io
- **Examples**: See the `examples/` directory
- **Issues**: https://github.com/bbuchsbaum/fmrimod/issues

For specific migration questions, please open an issue with the "migration" label.