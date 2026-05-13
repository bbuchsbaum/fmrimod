# Adversarial Benchmark Report Schema v1

Adversarial parity benchmarks report failure boundaries, not only array
allclose deltas. A report with `schema_version =
"adversarial-gauntlet/v1"` must contain:

- `name`, `status`, `summary`, and a nonempty `cases` list.
- case fields: `case_id`, `purpose`, `status`, `expected_boundary`,
  `design_shape`, `design_rank`, `design_condition`, `fmrimod`, `nilearn`,
  `comparisons`, and `verdict`.
- engine fields for both `fmrimod` and the reference path: `status`, `rank`,
  `df_residual`, `is_full_rank`, `ill_conditioned`, `aliased_columns`,
  finite/NaN fractions, `warning_messages`, `undefined_t_policy`,
  `exception_type`, and `exception_message`.

Allowed report status values are `pass` and `fail`. Allowed case status values
are `pass`, `fail`, `changed`, and `boundary_observed`. Allowed engine status
values are `ok` and `exception`.

Fractions such as `finite_stat_fraction` and `nan_p_fraction` must be numeric
values in `[0, 1]`. Numeric diagnostic values in `comparisons` may be `null`
when a comparison is undefined, for example when every statistic is `NaN`.

The schema is enforced by `benchmarks.parity.adversarial_schema`.
