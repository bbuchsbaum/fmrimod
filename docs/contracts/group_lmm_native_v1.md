# Native Group LMM Contract v1

Status: first native implementation slice.

## Scope

`fmrimod.group` now exposes native repeated-measures LMM reducers for the
voxelwise-theta subset:

- `lmm_ri(..., theta_mode="voxelwise")`
- `lmm_ri_slope1(..., theta_mode="voxelwise", slope=...)`

The implementation uses `statsmodels.MixedLM` as the correctness-first native
engine. It is not a port of fmrigds `lmm_core.cpp`, and it does not claim pooled
theta parity.

## Supported Modes

`lmm:ri` supports:

- one-sided fixed-effects formulas such as `~ condition`
- subject grouping from the `GroupDataset.subjects` axis
- repeated-measure predictors from `contrast_data`
- subject-level predictors from `col_data`
- `fit="REML"` and `fit="ML"`
- `theta_mode="voxelwise"`

`lmm:ri_slope1` additionally supports:

- one numeric `slope` column from repeated-measure observation data
- `covariance="diag"` or `covariance="full"`
- optional `center_slope=True`

## Explicit Gaps

These configurations remain unsupported natively and must fail explicitly:

- `theta_mode="pooled"`
- lmer-style random-effects formula syntax such as `~ x + (1 | subject)`
- missing or nonnumeric random-slope columns

Use `backend="fmrigds-r"` as the explicit R oracle or fallback when fmrigds
compiled pooled-theta behavior is required.

## Output Contract

Both reducers return `GroupDataset` with:

- `subjects=("meta",)`
- `contrasts=("model",)`
- scalar assays shaped `sample x 1 x 1`
- fixed-effect assays named `coef:<term>`, `se_coef:<term>`, `t_coef:<term>`,
  and `p_coef:<term>`
- variance-component assays such as `sigma2`, `vc_intercept`, `vc_resid`, and
  random-slope variance/correlation assays where applicable

The LMM milestone is not fully closed until optional R/fmrigds compiled-path
parity fixtures cover the supported voxelwise subset.
