# BIDS Parametric Group Follow-through

Status: `pass`

## Ergonomics

BIDS JSON -> translate_run_node -> fmri_dataset -> StatsModelContrast.apply -> condition(term='trial_type:rt_z') -> group_model('behavior') -> ols_voxelwise

UX status: `typed_contrast_full_group_manual`; E2E UX status: `partial`.

## Group Check

Median behavior coefficient: `0.4`; median t: `18724.4`.
