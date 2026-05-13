# Tier A: F-Contrast, Confounds, And Drift

This benchmark directory contains two evidence levels for the same overlap
slice: first-level OLS inference with nuisance/confound regressors,
polynomial drift, a t-contrast, and a joint F-contrast.

- `workflow.py` is a numerical canary. It uses fmrimod's low-level solver and
  contrast functions directly against Nilearn's low-level GLM reference. Its
  claim is algebraic parity only.
- `public_workflow.py` is the public-seam companion. The fmrimod path uses
  typed spec terms, `fmri_dataset`, `fmri_lm`, and `fit.contrast(...)`. Its
  claim is workflow parity for the first-level fit and contrast seam.

This is not yet a flagship workflow because it remains matrix-backed and does
not pressure-test neuroim-python image/mask/contrast-output routing.
