# Parity Caveats Index

This index lists every active parity caveat that appears in generated reports.
A caveat is not a tolerance escape hatch: it must name the affected quantity,
the current reason for divergence, an owning work item, and a concrete exit
criterion.

| Caveat ID | First appearance | Affected tiers | Owner | Exit criterion |
| --- | --- | --- | --- | --- |
| `spm-auditory-hrf-grid-scale` | `benchmarks/parity/tier_a_spm_auditory/` | Tier A SPM auditory first level | `bd-01KRFKYRKX6C4DMXEEEZSBESXY` | Add an exact reference HRF sampling/normalization mode, or update the case to compare a documented fmrimod-native design where design, effect, and t-stat maps pass without rescaling. |
| `localizer-tstat-variance-outliers` | `benchmarks/parity/tier_a_localizer_fixed_effects/` | Tier A localizer fixed effects | `bd-01KRFKZ0JG6FMC29VKRGWX5NMF` | Replace the variance-derived t-stat caveat with an explicit selectable covariance/whitening mode whose t map passes the normal allclose or strict absolute-error gate. |
| `fitlins-ar1-coefficient-binning` | `benchmarks/parity/tier_b_fitlins_bids/` | Tier B FitLins CLI derivative parity | `bd-01KRFKZ0JG6FMC29VKRGWX5NMF` | Factor the binned voxelwise AR(1) compatibility path into a generic parity configuration and narrow or remove the caveat once t and variance maps meet the standard gate. |
| `second-level-normal-vs-t-pvalues` | `benchmarks/parity/tier_c_second_level/` | Tier C second-level regression | `bd-01KRFGWV85C0EVQPFDCYE189C4` | Add the requested OLS second-level backend and switch the Nilearn-parity case to that backend so effect and statistic rows pass without the meta-regression caveat. |

## Maintenance Rule

Generated reports may introduce a new caveat only after this file is updated in
the same change. The test suite checks that every structured `caveat_id` in the
checked-in parity reports appears in this index.
