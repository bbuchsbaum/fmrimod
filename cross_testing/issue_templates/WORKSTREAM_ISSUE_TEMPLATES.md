# Workstream Issue Templates

Use one template per tracked issue. Keep one issue per workstream item.

## WS01 Design matrix construction parity

Title: `WS01: Design matrix construction parity (fmrimod vs fitlins/nilearn)`
Body:
- Problem: Design terms are not yet fully parity-validated across HRF, derivatives, drift, confounds.
- Scope: Build fixture matrix suite and parity comparator for design matrices.
- Acceptance criteria:
- Column-wise correlation >= 0.999 or documented transformed equivalence.
- Max absolute deviation within workstream tolerance table.
- Artifacts:
- JSON parity report path.
- Failing-case fixtures (if any).
- CI gate:
- Add/extend PR gate for WS01.

## WS02 Contrast engine parity

Title: `WS02: Contrast engine parity (t/F, effect, var, t, z, p)`
Body:
- Problem: Contrast outputs need broader parity coverage for multi-column contrasts.
- Scope: Add parity suites for t/F contrasts and output statistics.
- Acceptance criteria:
- t-corr >= 0.995 and p-MAE <= 0.035 on canonical fixtures.
- Artifacts:
- JSON parity report path.
- CI gate:
- Extend parity gate with WS02 metrics.

## WS03 Variance + degrees-of-freedom parity

Title: `WS03: Variance + df parity (OLS and AR-corrected)`
Body:
- Problem: Variance/df logic must be parity-verified for all supported paths.
- Scope: Add dedicated tests for sigma2 and df accounting.
- Acceptance criteria:
- sigma2-corr >= 0.995, sigma2-MAE <= 0.02, df contract tests pass.
- Artifacts:
- JSON report path + df contract logs.
- CI gate:
- Add sigma2/df threshold checks.

## WS04 Run-combination parity (fixed effects)

Title: `WS04: Run-combination parity for fixed-effects pooling`
Body:
- Problem: Multi-run pooling behavior needs explicit parity checks.
- Scope: Validate weighting and pooled inference outputs vs reference.
- Acceptance criteria:
- Combined betas/t-stats within parity thresholds on multi-run fixtures.
- Artifacts:
- Multi-run parity artifact.
- CI gate:
- Add WS04 fixture suite to PR gate.

## WS05 Censor/sample-mask parity

Title: `WS05: Censor/sample-mask parity with boundary safety`
Body:
- Problem: Censoring behavior around run boundaries and gaps must be proven.
- Scope: Add censor-sensitive fixtures and boundary leakage tests.
- Acceptance criteria:
- All censor fixtures pass with zero boundary leakage regressions.
- Artifacts:
- Censor parity artifact + boundary checks.
- CI gate:
- Add censor suite to PR gate.

## WS06 LSA/LSS parity + performance

Title: `WS06: LSA/LSS parity and performance program`
Body:
- Problem: Single-trial methods require both parity and speed superiority evidence.
- Scope: Add LSA/LSS parity fixtures and end-to-end benchmark harness.
- Acceptance criteria:
- Similarity thresholds pass.
- Median runtime speedup >= 1.5x vs reference baseline.
- Artifacts:
- LSA/LSS parity JSON + benchmark JSON.
- CI gate:
- Add LSA/LSS quick gate on PR and strict gate nightly.

## WS07 Rank-deficient design behavior

Title: `WS07: Rank-deficient design parity and robustness`
Body:
- Problem: Collinearity/singularity behavior must be deterministic and parity-validated.
- Scope: Add rank-deficient fixtures, dropped-column checks, contrast stability checks.
- Acceptance criteria:
- All rank-deficiency fixtures pass contract + parity tests.
- Artifacts:
- Rank-deficiency artifact set.
- CI gate:
- Add WS07 fixtures to PR gate.

## WS08 Numeric precision parity

Title: `WS08: Numeric precision parity (float64 vs float32)`
Body:
- Problem: Precision mode drift is not fully bounded across kernels.
- Scope: Add float32/float64 parity and drift envelope checks.
- Acceptance criteria:
- Precision drift stays within documented tolerances by stage.
- Artifacts:
- Precision drift artifact.
- CI gate:
- Add precision envelope checks.

## WS09 Residual diagnostic parity

Title: `WS09: Residual diagnostic parity (whiteness/autocorrelation)`
Body:
- Problem: Coefficient parity alone is insufficient for noise model correctness.
- Scope: Add residual diagnostic parity suites and whiteness thresholds.
- Acceptance criteria:
- Whiteness deltas and residual autocorrelation metrics within bounds.
- Artifacts:
- Residual diagnostics artifact.
- CI gate:
- Add WS09 diagnostic checks.

## WS10 Performance decomposition parity

Title: `WS10: Stage-level performance decomposition and regression gates`
Body:
- Problem: Performance regressions are hard to localize without stage decomposition.
- Scope: Benchmark design build, fit, contrast, whitening, and run-combine stages.
- Acceptance criteria:
- Stage medians published in JSON artifact.
- Stage-level regression gates active in CI.
- Artifacts:
- Stage benchmark JSON + trend snapshots.
- CI gate:
- Add PR quick threshold and nightly strict thresholds.
