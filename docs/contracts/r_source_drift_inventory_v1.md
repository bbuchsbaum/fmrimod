# R-source drift inventory (v1)

**Board source:** `r-source-drift/post-01M0DQRQMYYVEXFD9AQHY5N17G`
**Epic:** `bd-01M0DQSE2QBF4TZ6G9DKYY8G2W`
**Wave beads:** 1a AR `bd-01M0DQSF0AXMMT91WD0H4W5XQR`; 1b DL `bd-01M0DQSF9N141KXMK1C174HY77`; 1c design `bd-01M0DQSFJ9K4C6W9CWRYARH0YD`; 2 LSS/perm `bd-01M0DQSFWS9QSXYVW0E3NFMYTH`; 3 harden `bd-01M0DQSG8K2G4CVRTFJYBNSKB5`; 4 HRF `bd-01M0DQSGMSZWFQVKAJ504F4PQM`; 5 S2 `bd-01M0DQSH4XRHW0Z4G1BXVT710P`.
**Steward request:** 2026-08-19 audit of 2026 updates to `fmriAR`,
`fmridesign`, `fmrihrf`, `fmrilss`, `fmrireg`, and `fmrigds` against
fmrimod HEAD.
**R clones used:** `fmrihrf@69be541`, `fmridesign@a94ade7`,
`fmrireg origin/master@063c8ef`, `fmrilss@f851fb0`,
`fmriAR origin/main@c7f370a`, `fmrigds origin/main@67cc585`.

The R behavior is the spec; the Python surface is not a mechanical
translation. Rows marked **done** were already ported (or never buggy
because of redesign). Rows marked **wave N** are owned by this plan.
Rows marked **S2** are tracked, not implemented, until steward
approval (GOVERNANCE §10).

Cheap-pass disqualifier: a stub, no-op, or equal-weight special case
where old and new formulas agree cannot close a row.

## Status legend

- **done** — ported or N/A by redesign; residual nits listed.
- **wave1** — silent-wrong math.
- **wave2** — LSS contracts + perm FWER.
- **wave3** — hardening / docs.
- **wave4** — `estimate_hrf` redesign + image accessors.
- **S2** — architecture / product; beads only.

## fmrihrf 0.4.0

| Item | R | fmrimod | Wave |
| --- | --- | --- | --- |
| Epoch quadrature / no `1/precision` scaling (#45) | 0.4.0 breaking | Ported `tests/hrf/test_issue45_port.py` | done |
| `summate=FALSE` on conv path | 0.4.0 | Ported; docs now say duration-average | done |
| Duration-only impulse/block branch | 0.4.0 | Ported (`HRF.evaluate`); `block_hrf` factory still `width <= precision` | done |
| Hat-basis onsets, loop span+duration | 0.4.0 | Ported | done |
| sine/fourier scalar t, bspline knots, Daguerre grid, causal kernels | 0.4.0 | Ported | done |
| `hrf_boxcar` / `hrf_weighted` / trial-varying / `hrf_norm` | 0.2–0.4 | Ported | done |
| SPMG1 ×10 scale vs R `HRF_SPMG1` | convention | Indexed as `spmg1-x10-scale` | done |
| `SamplingFrame.precision` 0.1 vs evaluate 0.33 | R evaluate default 0.33 | Documented split | done |

## fmridesign 0.6.0

| Item | R | fmrimod | Wave |
| --- | --- | --- | --- |
| Nuisance diagnostics / `nuisance_check` / `na_action` | 0.6.0 / #7 | Ported | done |
| Onset bounds backstop (#5) | Jun 2026 | Ported | done |
| `convolve_design` data.frame `[[i]]` | 0.6.0 | N/A (vectorized) | done |
| Degenerate parametric-modulator warnings (#8) | 1486585 | Missing | **wave1** |
| Interaction contrast keys `:` vs `_` (#9) | 1486585 | Redesign (term-name keys) | done |
| Zero-event subset keeps canonical names | 0.6.0 | Python raises | S2 document |
| Shared-HRF C++ / skip-zero-column conv | 0.6.0 perf | N/A unless benches demand | S2 |

## fmrireg 0.2.0

| Item | R | fmrimod | Wave |
| --- | --- | --- | --- |
| AR residual df no longer deflated | 0.2.0 | Live path `n-rank`; `effective_df` no longer subtracts `ar_order` | done |
| Sandwich meat `e²` not `e⁴` | 0.2.0 | Ported | done |
| Robust / cfg / AR shorthand precedence | 0.2.0 | Typed config redesign | done |
| `na_action` for non-finite Y | Jun 2026 | Ported | done |
| Even-ISI simulate `n_events+1` | `c82918a` | Still `/ n_events` | **wave1** |
| `estimate_hrf` smooth FIR | 0.2.0 / `6fafaac` | Ported (`tests/test_hrf/test_estimate_hrf.py`) | done |
| `coef_images()` / contrast `coef_image` | 0.2.0 | Ported (`coef_images` + `type=`) | done |
| Model templates / `collect_results` | Jun 2026 | `StudyDataset` + native group | S2 |
| Mixed-model fail → NA not zeros | 0.2.0 | Ported (NaN + warn) | done |
| RRR bootstrap wrap when `block_size>=n` | 0.2.0 | One contiguous identity block | done |

## fmriAR 0.3.2 / 0.3.3

| Item | R | fmrimod | Wave |
| --- | --- | --- | --- |
| Per-run mean, fragment lag products | 0.3.3 | Per-fragment centering | **wave1** |
| Parcel honours `censor` / `runs` / fixed `p` | 0.3.3 / 0.3.2 | Ignored | **wave1** |
| PSD projection + relative margin | 0.3.3 | Missing | **wave1** |
| Global weights by uncensored n | 0.3.3 | Original run length | **wave1** |
| BIC `floor(n_eff/5)` selection cap | 0.3.3 | `n-1` only | **wave1** |
| `enforce_stationary_ar` shrink loop | 0.3.3 | PACF clip only | **wave1** |
| `noise_acvf`, plan `gamma`/`sigma2`, bias correction | 0.3.3 | Absent | **wave1** |
| `acorr_diagnostics(runs=)` | 0.3.3 | Ported (per-run centre, no cross-run lags) | done |
| Parcel labels / finite resid / arma+censor warn | 0.3.3 | Partial / missing | **wave1** |
| `censor=` API / `afni_restricted_plan` | 0.3.2 | Present | done (math in wave1) |

Red check: `tests/test_ar/test_estimator_correctness.py` transcribed from
`fmriAR/tests/testthat/test-estimator-correctness.R`. Cheap pass:
`plan.censor` stored while φ is still the fragment-centered estimate.

## fmrilss 0.2.0 / `f851fb0`

| Item | R | fmrimod | Wave |
| --- | --- | --- | --- |
| Top-level `prewhiten` + plan on result | 0.2.0 | Ported as `whitening_plan` | done |
| Forward `design` / `acvf_correction` / `correction_max_lag` | `f851fb0` | Ported (`tests/test_single/test_prewhiten_acvf.py`) | done |
| Multi-basis trial-major inference / OASIS-only `K>1` | `f851fb0` | Ported (`tests/test_single/test_trial_major.py`) | done |
| OASIS ridge defaults fractional 0.05 | 0.2.0 | Ported (`OasisConfig` defaults + `test_oasis.py`) | done |
| `create_lwu_grid` ∘ SBHM | `f851fb0` | Ported (`tests/test_single/test_lwu_grid.py`) | done |
| SBHM `alpha_source` / `whiten_power` | 0.2.0 | Ported (`SbhmConfig` + `sbhm_match`) | done |

## fmrigds (through `67cc585`)

| Item | R | fmrimod | Wave |
| --- | --- | --- | --- |
| DL meta-reg `tr(P)=sum(w)-sum(w² xAx)` | `reducers-core.R:188-194` | `sum(w_i xAx)` | **wave1** |
| Synthetic unit-var guard (#5) | Jun 2026 | Missing | **wave1** |
| `ols:voxelwise` listwise NaN (#7) | Jun 2026 | Ported with `n_obs` assay | done |
| Intercept-only `~ 1` without col_data (#1) | Jun 2026 | Ported | done |
| Perm FWER `alternative` + `p_fwer >= p_perm` (#22) | NEWS | Ported (`tests/test_group/test_perm_fwer_tails.py`) | done |
| Perm onesample weights 1/var, n_eff, custom (#21) | NEWS | Ported (`perm_onesample(weights=...)`) | done |
| NIfTI affine + `scl_slope=1` (#6) | Jun 2026 | Affine preserved; `scl_slope=1` on write | done |
| `lmm:ri` / `lmm:ri_slope1` | sprint 10 | Native voxelwise | done |
| `lmm:*_knownvar` | Aug 2026 | Missing | wave4 or R backend |
| `examine_group()` | `0de6114` | Missing | **S2** |
| Result frames / GDS projections | `67cc585` | Missing | **S2** |

Red check for DL: unequal-weight covariate fixture where `sum(w h)` and
`sum(w² h)` diverge; `tests/test_stats/test_meta_re_reg_parity.py` currently
transcribes the **wrong** formula as its independent reference and must
be rewritten.

## Wave map

| Wave | Closes | Depends on |
| --- | --- | --- |
| 0 | This inventory + board + beads | — |
| 1 | AR ACVF/censor/parcel, DL C, synthetic-var, degenerate modulators, even-ISI | 0 |
| 2 | LSS `f851fb0` contracts, perm FWER/weights | 1a |
| 3 | n_obs, scl_slope, acorr runs, effective_df, docs/CAVEATS, mixed NaN, RRR bootstrap | 1 |
| 4 | `estimate_hrf` FIR, `coef_images`, SBHM knobs | 2 |
| 5 | examine_group, frames, templates (beads only) | steward |
