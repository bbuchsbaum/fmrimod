# Retired Parity Caveat Receipts

This receipt lists parity caveats that once appeared in
[`CAVEATS.md`](CAVEATS.md) and were retired by checked-in evidence. It
keeps the active caveat index focused on live divergences while making
an empty or short active table auditable instead of silent.

| Caveat ID | Active | Original owner | Retirement commit | Report path | Red check |
| --- | --- | --- | --- | --- | --- |
| `spm-auditory-hrf-grid-scale` | `false` | `bd-01KRFKYRKX6C4DMXEEEZSBESXY` | `a6c3af9` | `benchmarks/parity/tier_a_spm_auditory/reports/parity_report.json` | `python3.9 -m pytest cross_testing/test_spm_auditory_parity.py -q` |
| `localizer-tstat-variance-outliers` | `false` | `bd-01KRGRF4X588F56QT21CPBB7NJ` | `db5fcce` | `benchmarks/parity/tier_a_localizer_fixed_effects/reports/parity_report.json` | `python3.9 -m pytest tests/test_benchmarks/test_tier_a_localizer_diagnostics.py -q` |
| `fitlins-ar1-coefficient-binning` | `false` | `bd-01KRGRF4X588F56QT21CPBB7NJ` | `2b3660b` | `benchmarks/parity/tier_b_fitlins_bids/reports/fitlins_cli_derivative_report.json` | `python3.9 -m pytest tests/test_ar/test_ar1_nilearn.py cross_testing/test_fitlins_cli_derivative_parity.py -q` |
| `second-level-normal-vs-t-pvalues` | `false` | `bd-01KRGRF4WRPCKEK7FBR8BXJ4AY` | `64fd3fb` | `benchmarks/parity/tier_c_second_level/reports/parity_report.json` | `python3.9 -m pytest tests/test_benchmarks/test_tier_c_second_level_workflow.py -q` |
| `typed-re-meta-regression-unimplemented` | `false` | `bd-01KRRJVG3YM80X0FX9C8KY5VG4` | `100ae17` | `tests/test_stats/test_meta_re_reg_parity.py` (no benchmark workflow exercises meta-regression, so by design there is no `parity_report.json`; fabricating a canary was disqualified as dishonest in the root-cause trace on `bd-01KRRMM5AEAKVQWAYWKTZRF263` — the checked-in parity test is the evidence, tying the typed path to the R-oracle-validated native `meta_re_reg` kernel) | `python3.9 -m pytest tests/test_stats/test_meta_re_reg_parity.py tests/test_benchmarks/test_parity_proof_artifacts.py::test_caveats_index_matches_generated_report_caveat_ids -q` |

## Maintenance Rule

When an active caveat row is removed from `CAVEATS.md`, add or update a
receipt row here in the same commit. The receipt row must name the
retired `caveat_id`, set `Active` to `false`, preserve the owning bead,
cite the retirement commit, point to a checked-in report path, and name
a runnable red-check command.
