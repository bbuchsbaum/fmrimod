# FitLins/Nilearn Overlap Parity Contract v1

Status: Draft

## Objective

Establish that `fmrimod` can do the same statistical modeling work as FitLins
and Nilearn in the domains where the projects overlap. The program has three
separate deliverables:

- **Functional parity**: `fmrimod` can produce the same classes of outputs from
  the same inputs: design matrices, beta estimates, contrast statistics,
  statistical maps, group summaries, and reports.
- **Numerical reproducibility**: for matched data and matched modeling choices,
  `fmrimod` agrees with FitLins/Nilearn within documented tolerances, or the
  divergence is named, justified, and tested as a caveat.
- **Performance and ergonomics**: `fmrimod` is competitive on runtime and memory
  while exposing a less painful API for the same modeling task.

Each accepted parity example must produce:

1. an executable `fmrimod` workflow;
2. a parity report in JSON plus Markdown;
3. a benchmark row or explicit note that the case is correctness-only.

## Scope

The overlap surface is intentionally limited.

### In Scope

- First-level GLM: SPM canonical and Glover HRFs, time/dispersion derivatives,
  block and event designs, parametric modulators, AR(1) prewhitening, t/F
  contrasts, confound regression, polynomial/cosine drift, and high-pass terms.
- BIDS-style integration: BIDS Stats Model JSON consumption, `*_events.tsv`
  ingestion, sidecar metadata needed for timing, and fMRIPrep-style confound TSV
  joins.
- Group and second level: one-sample t, paired t, simple regression on
  first-level contrast maps, intercept-only models, and contrast-of-effects.
- Reporting: design matrix figures, contrast matrices, design correlation,
  statistical map summaries, and compact HTML/Quarto reports.

### Out of Scope

- Surface analysis beyond what the existing image adapters can carry.
- MVPA and decoding; Nilearn owns that domain.
- Atlas, parcellation, and general image manipulation utilities; delegate to
  Nilearn where useful.
- Preprocessing; `fmrimod` consumes fMRIPrep-style outputs and does not replace
  fMRIPrep.

## Reference Workflow Tiers

| Tier | Workflow | Reference | fmrimod target | Required evidence |
| --- | --- | --- | --- | --- |
| Canary | Synthetic OLS GLM | `nilearn.glm.first_level.run_glm` | low-level `fmrimod` OLS solver | Harness JSON/Markdown report; no network or dataset fetch. |
| A1 | SPM auditory block design | Nilearn first-level example | fmrimod event/baseline design + OLS solver | Implemented in `benchmarks/parity/tier_a_spm_auditory/` and `cross_testing/test_spm_auditory_parity.py`; beta/effect and listening t-map parity now pass without the retired `spm-auditory-hrf-grid-scale` caveat because `hrf(..., norm="spm")` matches Nilearn's reference-grid scale. |
| A2 | FIAC event-related contrasts | Nilearn first-level example | fmrimod OLS solver over Nilearn design matrices | Implemented in `benchmarks/parity/tier_a_fiac/`; sentence and speaker contrast effect/t maps match Nilearn fixed-effects outputs. |
| A3 | Localizer audio-vs-visual contrast | Nilearn first-level example | fmrimod OLS solver over Nilearn design matrix | Implemented in `benchmarks/parity/tier_a_localizer_fixed_effects/`; localizer audio computation vs visual computation effect maps match Nilearn, and t maps pass rank/MAE thresholds with the declared `localizer-tstat-variance-outliers` caveat. |
| A4 | Synthetic confound/drift F-contrast | `nilearn.glm.first_level.run_glm` + `compute_contrast` | fmrimod low-level OLS solver plus public `fmri_dataset → fmri_lm → contrast` seam | Implemented in `benchmarks/parity/tier_a_f_confound_drift/`; `workflow.py` is a numerical canary, and `public_workflow.py` verifies the same joint F-test over two task regressors through typed spec terms, nuisance/confound columns, polynomial drift, and public contrast calls. |
| B1 | FitLins-style BIDS workflow | FitLins CLI | BIDS Stats Model translator + `fmrimod` fit | Translator v1 implemented in `fmrimod.bids.stats_model`; `benchmarks/parity/tier_b_fitlins_bids/` verifies the constrained FitLins-style JSON subset and runs a real local FitLins CLI derivative-tree comparison on a tiny BIDS/fMRIPrep fixture. Effect maps pass tight voxel-wise thresholds; t/variance maps are reported with the declared `fitlins-nilearn-ar1-stat-covariance` caveat. |
| B2 | fMRIPrep confound regression | Nilearn/FitLins confound API | confound TSV to baseline/nuisance model | design and fitted-stat parity. |
| C1 | One-sample group t-test | `SecondLevelModel` | `group_fit` | Implemented in `benchmarks/parity/tier_c_second_level/`; synthetic one-sample effect and t maps match Nilearn. |
| C2 | Regression second level | `SecondLevelModel` | `group_fit` | Implemented in `benchmarks/parity/tier_c_second_level/`; synthetic covariate effects match Nilearn. The statistic is rank/correlation checked with the declared `second-level-normal-vs-t-pvalues` caveat because fmrimod meta-regression uses supplied effect-size SEs while Nilearn estimates second-level OLS residual variance. |
| D1 | AR(p) and robust IRLS | fmrimod-distinctive | AR(2)+ and robust estimators | Implemented in `benchmarks/parity/tier_d_showcase/`; AR(2) recovery and Huber outlier downweighting pass executable thresholds. |
| D2 | Sketched high-voxel GLM | fmrimod-distinctive | sketch/chunk engines | Implemented in `benchmarks/parity/tier_d_showcase/`; sketched high-voxel GLM benchmark reports beta correlation against exact OLS. |
| D3 | LSS trial-wise betas | Nilearn recipe | `fmrimod.single.lss` | Implemented in `benchmarks/parity/tier_d_showcase/`; vectorized LSS matches a per-trial reference recipe and reports timing rows. |

## Harness Policy

The reusable harness lives under `cross_testing/harness/` and must support:

- named parity cases with candidate/reference pipelines;
- per-output tolerance settings;
- array deltas with max absolute error, MAE, Pearson correlation, Spearman
  correlation, and rank diagnostics;
- first-class caveats with stable IDs;
- JSON and Markdown report rendering.

Default tolerance policy:

- Design matrices: `rtol=1e-6`, `atol=1e-9`, exact rank unless caveated.
- Betas: `rtol=1e-4`, `atol=1e-6`, Pearson/Spearman correlations reported.
- t-statistics: `rtol=1e-3`, `atol=1e-5`, Spearman rank correlation reported
  because thresholding depends on rank stability.
- p-values: report MAE and correlation; use case-specific thresholds near zero.

Caveats are allowed only when checked into the case definition. Each caveat must
include:

- `caveat_id`;
- affected quantity;
- reason;
- expected direction or magnitude when known;
- linked issue or contract section.

Undeclared drift fails the parity test.

## Repository Layout

Use existing `cross_testing` infrastructure for shared code and fast parity
tests. Heavy workflow cases can live under `benchmarks/parity/` once dataset
fetching and derivative-tree outputs are needed.

Planned layout:

```text
cross_testing/
  harness/
    compare.py
    fixtures.py
    report.py
  test_parity_harness_canary.py
benchmarks/
  parity/
    tier_a_spm_auditory/
    tier_a_fiac/
    tier_a_localizer_fixed_effects/
    tier_b_fitlins_bids/
    tier_c_second_level/
    tier_d_showcase/
docs/tutorials/parity/
```

## Phases

### P1. Harness and Canary

- Build the shared harness.
- Add a synthetic OLS canary against Nilearn's GLM path.
- Add the first public Nilearn dataset case only after the no-I/O canary is
  green.
- File motes for every API gap exposed.

### P2. Tier A Nilearn First-Level Workflows

- Port SPM auditory, FIAC, and localizer fixed-effects cases.
- Bring AR(1), HRF derivatives, confounds, and run combination into scope.
- Document all caveats.

### P3. FitLins-Style BIDS Workflow

- Implement a thin `fmrimod.bids.stats_model` translator from BIDS Stats Model
  JSON to `FmriModel`.
- Run one ds000003 or ds000114 workflow through FitLins and `fmrimod`.
- Compare derivative trees voxel-wise.
- Current executable compromise avoids dataset downloads by generating a tiny
  BIDS/fMRIPrep-style fixture and running the real local FitLins CLI through
  `uv`; report: `benchmarks/parity/tier_b_fitlins_bids/reports/FITLINS_CLI_DERIVATIVES.md`.

### P4. Second Level

- Compare `group_fit` to Nilearn `SecondLevelModel` on one-sample and
  covariate/regression cases.
- Keep the scoped contract in `docs/contracts/second_level_parity_v1.md` ahead
  of any implementation change.

### P5. fmrimod-Distinctive Showcases

- Add AR(p)/robust IRLS, sketched high-voxel GLM, and LSS examples.
- These cases demonstrate value beyond parity and are not required to match a
  reference package.
- Current executable report: `benchmarks/parity/tier_d_showcase/reports/SHOWCASE.md`.

### P6. Performance Tracking

- Add nightly benchmark trend artifacts.
- Gate PRs on correctness; track performance as a trend unless a protected
  performance path has an explicit threshold.
- Current trend artifact: `benchmarks/performance/results/parity_performance_trends.json`.

## Phase 0 Success Criteria

- Three Tier A first-level workflows reproduce Nilearn within declared
  tolerance, with caveats documented and tested.
- One BIDS Stats Model workflow reproduces FitLins with only documented
  derivative-tree deltas.
- One second-level workflow reproduces Nilearn.
- Three showcase examples demonstrate features FitLins/Nilearn do not provide.
- Public parity table and tutorial chapters are generated from checked-in
  reports.
- Nightly performance trend artifacts exist.
- Every API gap discovered is fixed or filed in mote with an explicit decision.

## Known Pain Points to Track

- Naming friction between Nilearn result labels and `fmrimod` result accessors.
- HRF derivative conventions and sampling-grid differences.
- fMRIPrep confound TSV ingestion ergonomics.
- BIDS Stats Model JSON coverage.
- AR estimator differences between pooled AR(1) and voxelwise AR(p).
- Mask handling convenience around NIfTI inputs.
- Missing FitLins/Nilearn-style GLM report output.
