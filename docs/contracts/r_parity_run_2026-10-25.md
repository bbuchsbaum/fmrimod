# R parity run, 2026-10-25

The first full R-backed `cross_testing/` run. Until now the suite had only
skipped (`R PARITY: NOT CHECKED`). Steward decision D2 asks for the
`AGENTS.md` note to be re-baselined once R is available.

## Environment

| Component | Version |
| --- | --- |
| fmrimod | `main` at `19af704` |
| R | 4.3.3 (Ubuntu 24.04 `r-base-core`) |
| rpy2 | 3.5.17 (3.6 needs R >= 4.4) |
| fmrihrf | 0.4.0.9000, `bbuchsbaum/fmrihrf@5b63d6a` |
| fmriAR | 0.4.2, `bbuchsbaum/fmriAR@bb0789c` |
| fmrilss | 0.2.0, local clone |

CRAN was unreachable from the container. Dependencies came from Ubuntu
`r-cran-*` packages, with `roptim` taken from the `cran/roptim` GitHub
mirror, and every package was installed with `R CMD INSTALL`.

## Result

```
FMRIMOD_REQUIRE_R=1 pytest cross_testing/
R PARITY: 54 R-backed test(s) ran against the R reference.
22 failed, 90 passed, 6 skipped
```

| Cause | Tests | Evidence |
| --- | --- | --- |
| **D2, SPMG1 shape.** fmrimod `spmg1` is exactly `gamma(6) - 0.167 * gamma(16)` with span 32 s; R `HRF_SPMG1` is exactly `gamma(6) - gamma(16) / 6` with span 24 s. | 15: `test_hrf_library[spmg1]`, `test_spm_canonical`, `test_regressor_equivalence` (6), `test_complex_scenarios` (4), `test_performance` (3) | With `SPMG*_HRF.ratio = 1/6` and `span = 24` (plugin [`r_parity_receipts/spm_r_conventions_plugin.py`](r_parity_receipts/spm_r_conventions_plugin.py)), 14 of these pass. The 15th, `test_spm_canonical`, also passes once the `spm_hrf` function default ratio is 1/6. A regressor built on R's exact formula with a 24 s span matches R to 2e-16 for durations 0, 2 and 5 s. |
| **D2, SPMG2/SPMG3 derivative column.** fmrimod uses an onset-shift derivative (SPM, Nilearn); R uses analytic dh/dt. | 4: `test_spmg2`, `test_spmg3`, `test_hrf_library[spmg2]`, `test_hrf_library[spmg3]` | Under the plugin, the canonical and dispersion columns match R to 2.5e-16 and 2.5e-14. Only column 2 differs (corr 0.956, norm ratio 0.91). |
| **Test bug.** `test_gaussian_hrf`'s second R block used `t` without defining it, so it picked up a leftover global. | 1 | `gaussian_hrf` matches R `hrf_gaussian` to 3e-17. Fixed in the same commit as this receipt. |
| Non-R, known | 2: `test_tier_d_showcase`, `test_fitlins_cli_derivative_tree_parity` | They fail with R absent too. |

The R comparisons that pass include fmriAR ARMA (20 cases, to 2.5e-11), the
fmrilss AR-aware LSS oracle, and every regressor or design case that does
not use SPMG.

## What it means

fmrimod's convolution, regressor and design engines agree with R at
floating-point level. Every remaining R difference is a convention choice
D2 already lists: SPMG1's undershoot ratio and span, and SPMG2's second
column. The old `AGENTS.md` note attributed the failures to a factor of 10
in normalization. That describes the `*_Legacy` SPMG classes
(`a1 = 0.0833`), not current `spmg1`. D2 should decide the conventions and
whether the Legacy classes stay. After that, the `spmg1-x10-scale` caveat
can be retired with its tolerance scaffold.

## Reproduce

```bash
apt-get install r-base-core r-base-dev r-cran-rcpp r-cran-rcpparmadillo \
  r-cran-{assertthat,purrr,matrix,cli,jsonlite,memoise,cachem,numderiv,pracma} \
  r-cran-{mass,bigmemory,ggplot2,glmnet,bh,rlang,dplyr,tibble}
git clone https://github.com/cran/roptim && R CMD INSTALL roptim
R CMD INSTALL path/to/fmrihrf path/to/fmriAR path/to/fmrilss
uv pip install --python .venv/bin/python "rpy2==3.5.17"
FMRIMOD_REQUIRE_R=1 .venv/bin/python -m pytest cross_testing/
```
