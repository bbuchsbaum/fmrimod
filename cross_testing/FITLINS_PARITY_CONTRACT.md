# Fitlins Parity Contract

This document defines the first gate in the fmrimod-vs-fitlins superiority plan:

1. Match first-level OLS GLM outputs at high numerical agreement.
2. Use that parity gate as a fixed correctness guard before speed tuning.

## Scope (Gate A)

- Model class: first-level GLM, OLS.
- Reference implementation: nilearn GLM path used by fitlins nistats estimator.
- Inputs: matched synthetic `X` (design), `Y` (voxel time series), and a fixed t-contrast.
- Outputs compared:
  - `betas`
  - `sigma2`
  - `t` statistics
  - `p` values
  - t-stat sign flips

## Thresholds

The current contract constants are implemented in `cross_testing/fitlins_parity.py` (`ParityThresholds`):

- `beta_corr >= 0.999`
- `beta_mae <= 5e-4`
- `beta_max_abs <= 5e-2`
- `sigma2_corr >= 0.999`
- `sigma2_mae <= 1e-3`
- `t_corr >= 0.995`
- `t_mae <= 5e-3`
- `t_max_abs <= 2.5e-1`
- `p_mae <= 1e-2`
- `sign_flip_rate <= 1e-3`

## Speed Gate (Gate B)

Speed is tracked but kept separate from parity:

- Benchmark measures median runtime over repeated matched runs.
- Initial speed threshold is conservative:
  - `speedup_vs_reference >= 1.0`
- The strategic target remains `>=3x` end-to-end on representative first-level jobs.

## Reproducible Commands

Run parity tests:

```bash
pytest cross_testing/test_fitlins_parity.py -v
```

Generate parity + benchmark report JSON:

```bash
python cross_testing/benchmark_fitlins.py \
  --output cross_testing/reports/fitlins_parity_benchmark.json
```

