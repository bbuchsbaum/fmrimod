# fmriAR Parity Audit v1

Status: Complete local audit for `fmriAR` 0.3.2

Date: 2026-05-12
Source package: `/Users/bbuchsbaum/code/fmriAR`
Target package: `/Users/bbuchsbaum/code/pycode/fmrimod`

## Objective

Check whether the public R package `fmriAR` has parity coverage in `fmrimod`
in addition to the `fmrihrf`, `fmridesign`, and `fmrireg` trio port.

## R Export Inventory

The installed/local `fmriAR` public export set was inspected with:

```sh
Rscript -e "library(fmriAR); cat(paste(sort(getNamespaceExports('fmriAR')), collapse='\n'))"
```

Exports:

| R export | Python surface | Evidence |
| --- | --- | --- |
| `acorr_diagnostics` | `fmrimod.acorr_diagnostics`; `fmrimod.ar.acorr_diagnostics` | `tests/test_ar/test_diagnostics.py`; `tests/test_ar/test_rpy2_parity.py`; `tests/test_ar/test_fmriar_export_compat.py` |
| `afni_restricted_plan` | `fmrimod.afni_restricted_plan`; `fmrimod.ar.afni_restricted_plan` | `tests/test_ar/test_afni.py`; `tests/test_ar/test_fmriar_export_compat.py` |
| `compat` | `fmrimod.compat`; `fmrimod.ar.compat` | `fmrimod/ar/compat.py`; `tests/test_ar/test_fmriar_export_compat.py` |
| `fit_noise` | `fmrimod.fit_noise`; `fmrimod.ar.fit_noise` | `tests/test_ar/test_estimation.py`; `tests/test_ar/test_integration.py`; `tests/test_ar/test_rpy2_parity.py`; `tests/test_ar/test_fmriar_export_compat.py` |
| `sandwich_from_whitened_resid` | `fmrimod.sandwich_from_whitened_resid`; `fmrimod.ar.sandwich_from_whitened_resid` | `tests/test_ar/test_diagnostics.py`; `tests/test_ar/test_rpy2_parity.py`; `tests/test_ar/test_fmriar_export_compat.py` |
| `whiten` | `fmrimod.whiten`; `fmrimod.ar.whiten` | `tests/test_ar/test_whitening.py`; `tests/test_ar/test_fmriar_export_compat.py` |
| `whiten_apply` | `fmrimod.whiten_apply`; `fmrimod.ar.whiten_apply` | `tests/test_ar/test_whitening.py`; `tests/test_ar/test_integration.py`; `tests/test_ar/test_rpy2_parity.py`; `tests/test_ar/test_fmriar_export_compat.py` |

## Scope Notes

- The R package has non-exported helper functions for PACF transforms,
  stationarity enforcement, Hannan-Rissanen ARMA fitting, multiscale pooling,
  and AFNI root parameterization. Python exposes many of these as lower-level
  `fmrimod.ar` helpers, but this audit treats the installed R namespace exports
  as the public API contract.
- `compat` is represented by the Python module `fmrimod.ar.compat`, re-exported
  as top-level `fmrimod.compat` for migration parity.
- The R `print.fmriAR_plan` method maps to Python `WhiteningPlan.__repr__`.

## Verification Transcript

Commands run locally:

| Check | Command | Result |
| --- | --- | --- |
| R export set | `Rscript -e "library(fmriAR); cat(paste(sort(getNamespaceExports('fmriAR')), collapse='\n'))"` | Listed the seven exports above. |
| Top-level/API compatibility | `python -m pytest tests/test_ar/test_fmriar_export_compat.py -q` | `4 passed in 0.88s`. |
| R parity tests | `python -m pytest tests/test_ar/test_rpy2_parity.py -q` | `30 passed in 1.49s`. |
| Core AR/import tests | `python -m pytest tests/test_ar tests/test_import_cycles.py tests/test_type_hints_resolution.py -q` | `155 passed in 3.07s`. |
| Top-level export probe | Python probe checking each `fmriAR` export is present on `fmrimod` and in `fmrimod.__all__` | All seven exports present; `len(fmrimod.__all__) == 148`. |

## Conclusion

`fmriAR` parity is covered for the installed public export surface. The
implementation has Python module coverage, top-level migration aliases,
focused unit tests, and live-R rpy2 parity tests for the numerical core.
