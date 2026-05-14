# BIDS Parametric Group Follow-through

Status: `pass`

## Ergonomics

BIDS JSON -> translate_run_node -> fmri_dataset -> StatsModelContrast.apply -> modulator('rt_z').within('trial_type') -> group_model('behavior') -> ols_voxelwise

UX status: `typed_contrast_full_group`; E2E UX status: `full`.

## Group Check

Median behavior coefficient: `0.4`; median t: `18724.4`.
