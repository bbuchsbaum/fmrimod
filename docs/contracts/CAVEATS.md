# Parity Caveats Index

This index lists every active parity caveat that appears in generated reports.
A caveat is not a tolerance escape hatch: it must name the affected quantity,
the current reason for divergence, an owning work item, and a concrete exit
criterion.

| Caveat ID | First appearance | Affected tiers | Owner | Exit criterion |
| --- | --- | --- | --- | --- |
| `localizer-tstat-variance-outliers` | `benchmarks/parity/tier_a_localizer_fixed_effects/` | Tier A localizer fixed effects | `bd-01KRGRF4X588F56QT21CPBB7NJ` | Resolve the remaining near-zero-dispersion sparse-mask voxels where Nilearn `run_glm` and fmrimod's X/Y-aware OLS path produce different variance-derived t statistics, then restore the normal allclose gate. |
| `fitlins-ar1-coefficient-binning` | `benchmarks/parity/tier_b_fitlins_bids/` | Tier B FitLins CLI derivative parity | `bd-01KRGRF4X588F56QT21CPBB7NJ` | Narrow or remove the AR/statistic caveat once t and variance maps meet the standard gate without undocumented tolerance bypasses. |
| `second-level-normal-vs-t-pvalues` | `benchmarks/parity/tier_c_second_level/` | Tier C second-level regression | `bd-01KRGRF4WRPCKEK7FBR8BXJ4AY` | Add the requested OLS second-level backend and switch the Nilearn-parity case to that backend so effect and statistic rows pass without the meta-regression caveat. |

## Maintenance Rule

Generated reports may introduce a new caveat only after this file is updated in
the same change. The test suite checks that every structured `caveat_id` in the
checked-in parity reports appears in this index.
Rows in this index must also have live owning work items unless they explicitly
name a closed replacement owner in the exit criterion.
