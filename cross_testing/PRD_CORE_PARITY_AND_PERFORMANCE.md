# PRD: Core Computation Parity and Performance Program

Status: Draft
Owner: fmrimod core team
Date: 2026-02-18

## 1. Problem

`fmrimod` must be demonstrably superior to `fitlins`/`nilearn` in speed and implementation quality while staying numerically trustworthy and not bloated. Today, parity and performance checks exist, but they are not yet complete across all core computations.

## 2. Goals

1. Prove parity (or explicitly better behavior) for core first-level computations.
2. Beat reference runtimes with transparent, repeatable benchmarks.
3. Prevent regressions using CI gates and stable benchmark artifacts.

## 3. Non-Goals

1. Reproducing every private internal detail of reference implementations.
2. Adding broad second-level/group inference in this PRD.
3. Optimizing I/O pipelines beyond what is needed to benchmark core math kernels.

## 4. Scope Matrix (Required Workstreams)

## 4.1 Design matrix construction parity
Requirements:
- Match event expansion, durations, amplitudes, HRF basis handling, derivatives, drift/confounds.
- Compare matrix values and column semantics against reference.
Acceptance:
- Correlation per design column >= 0.999 (or justified transformed equivalence).
- Max absolute deviation within predefined tolerance envelope by feature class.

## 4.2 Contrast engine parity
Requirements:
- Validate t/F contrasts for single-column and multi-column contrasts.
- Validate effect, variance, t, z, p outputs.
Acceptance:
- t-corr >= 0.995, p-MAE <= 0.035 on canonical synthetic suites.

## 4.3 Variance + degrees-of-freedom parity
Requirements:
- Verify residual variance and df accounting for OLS and AR-corrected runs.
Acceptance:
- sigma2-corr >= 0.995, sigma2-MAE <= 0.02, df logic matches contract tests.

## 4.4 Run-combination parity (fixed effects)
Requirements:
- Verify run pooling and weighting formulas.
Acceptance:
- Combined betas/t-stats within parity thresholds on multi-run fixtures.

## 4.5 Censor/sample-mask parity
Requirements:
- Verify handling of dropped frames, run boundaries, and gap segmentation.
Acceptance:
- Censor-sensitive fixtures pass with no boundary leakage.

## 4.6 LSA/LSS parity + performance
Requirements:
- Compare trial-wise estimates and summary contrasts.
- Benchmark end-to-end single-trial workloads.
Acceptance:
- Similarity thresholds met; median runtime speedup >= 1.5x vs reference baseline.

## 4.7 Rank-deficient design behavior
Requirements:
- Verify singular/collinear designs, dropped columns, stable inference behavior.
Acceptance:
- All rank-deficiency fixtures pass contract + parity checks.

## 4.8 Numeric precision parity
Requirements:
- Compare float64 vs float32 behavior for key kernels.
Acceptance:
- Precision drift stays within feature-specific tolerance envelopes.

## 4.9 Residual diagnostic parity
Requirements:
- Compare whiteness/autocorrelation diagnostics, not only coefficient outputs.
Acceptance:
- Whiteness deltas within bounds on shared synthetic AR datasets.

## 4.10 Performance decomposition parity
Requirements:
- Separate benchmarks for design build, fit, contrast, whitening, and run-combine.
Acceptance:
- Per-stage medians tracked in JSON artifacts; stage regressions gate on CI.

## 5. AR Reference Policy

1. AR1 first-level parity reference: `fitlins`/`nilearn`.
2. ARMA/ARp accuracy reference: `fmriAR`.
3. Where references disagree, document expected behavior and choose explicit contract.

## 6. Deliverables

1. Unified parity matrix runner producing one JSON report per run.
2. Dedicated benchmark scripts per workstream (including ARMA path benchmark).
3. CI workflows:
- PR-level quick gate (reduced repeats, looser thresholds).
- Nightly gate (higher repeats, tighter thresholds).
4. Human-readable summary report generator from JSON artifacts.

## 7. Metrics and Gates

Core parity metrics:
- beta_corr, beta_mae, beta_max_abs
- t_corr, t_mae, t_max_abs
- p_mae, sigma2_corr, sigma2_mae
- sign_flip_rate

Core speed metrics:
- median runtime per stage
- speedup_vs_reference (overall and by stage)

Initial gates:
- Preserve current AR1 parity thresholds already used in benchmark harness.
- Require speedup_vs_reference >= 1.0 for protected paths.
- Require ARMA benchmark threshold checks (run/global) in CI.

## 8. Execution Plan

Phase 1: Coverage completion
- Implement missing parity suites for items 1-10.
- Standardize fixture generators and JSON schema.

Phase 2: Optimization pass
- Profile stage bottlenecks and optimize highest-leverage kernels.
- Keep regression tests + parity gates green during all optimization PRs.

Phase 3: Hardening
- Tighten thresholds based on stable historical distributions.
- Add trend monitoring and regression alerting.

## 9. Risks and Mitigations

Risk: Benchmark noise causes flaky CI.
Mitigation: PR quick mode + nightly strict mode; median-based thresholds.

Risk: Overfitting to one reference implementation quirk.
Mitigation: Contract tests + dual-reference policy (`fitlins/nilearn` and `fmriAR`).

Risk: Speed improvements degrade numerical behavior.
Mitigation: parity + contract + residual-diagnostic gates are required in same PR.

## 10. Definition of Done

This PRD is complete when:
1. All 10 workstreams have automated parity tests and benchmark coverage.
2. CI enforces parity and performance thresholds with artifacts attached.
3. AR1 and ARMA/ARp reference policies are codified and passing continuously.
4. Performance claims are reproducible via checked-in scripts and reports.

## 11. Execution Checklist (Workstream-to-Issue Mapping)

Execution checklist:
- [ ] WS01 open and assigned: Design matrix construction parity.
- [ ] WS02 open and assigned: Contrast engine parity.
- [ ] WS03 open and assigned: Variance + degrees-of-freedom parity.
- [ ] WS04 open and assigned: Run-combination parity.
- [ ] WS05 open and assigned: Censor/sample-mask parity.
- [ ] WS06 open and assigned: LSA/LSS parity + performance.
- [ ] WS07 open and assigned: Rank-deficient design behavior.
- [ ] WS08 open and assigned: Numeric precision parity.
- [ ] WS09 open and assigned: Residual diagnostic parity.
- [ ] WS10 open and assigned: Performance decomposition parity.
- [ ] PR quick CI gate enabled for parity/perf artifacts.
- [ ] Nightly CI gate enabled for strict thresholds and trend checks.

Issue template mapping:
1. WS01 -> `.github/ISSUE_TEMPLATE/ws01-design-matrix-construction-parity.md`
2. WS02 -> `.github/ISSUE_TEMPLATE/ws02-contrast-engine-parity.md`
3. WS03 -> `.github/ISSUE_TEMPLATE/ws03-variance-and-df-parity.md`
4. WS04 -> `.github/ISSUE_TEMPLATE/ws04-run-combination-parity.md`
5. WS05 -> `.github/ISSUE_TEMPLATE/ws05-censor-sample-mask-parity.md`
6. WS06 -> `.github/ISSUE_TEMPLATE/ws06-lsa-lss-parity-performance.md`
7. WS07 -> `.github/ISSUE_TEMPLATE/ws07-rank-deficient-design-behavior.md`
8. WS08 -> `.github/ISSUE_TEMPLATE/ws08-numeric-precision-parity.md`
9. WS09 -> `.github/ISSUE_TEMPLATE/ws09-residual-diagnostic-parity.md`
10. WS10 -> `.github/ISSUE_TEMPLATE/ws10-performance-decomposition-parity.md`

## 12. Benchmark Artifact Schema

Canonical JSON schema for benchmark artifacts is defined at:
- `cross_testing/schemas/core_parity_benchmark.schema.json`

This schema supports:
1. Fitlins AR1 parity/speed benchmark artifacts.
2. AR/ARMA path benchmark artifacts.
3. Future unified core parity matrix artifact.

All new benchmark scripts should emit JSON that validates against one of the schema variants.
