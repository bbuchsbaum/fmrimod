# Parity Caveats Index

This index lists every active parity caveat that appears in generated reports.
A caveat is not a tolerance escape hatch: it must name the affected quantity,
the current reason for divergence, an owning work item, and a concrete exit
criterion.

| Caveat ID | First appearance | Affected tiers | Owner | Exit criterion |
| --- | --- | --- | --- | --- |
| `localizer-tstat-variance-outliers` | `benchmarks/parity/tier_a_localizer_fixed_effects/` | Tier A localizer fixed effects | `bd-01KRGRF4X588F56QT21CPBB7NJ` | Resolve the remaining near-zero-dispersion sparse-mask voxels where Nilearn `run_glm` and fmrimod's X/Y-aware OLS path produce different variance-derived t statistics, then restore the normal allclose gate. |
| `fitlins-ar1-coefficient-binning` | `benchmarks/parity/tier_b_fitlins_bids/` | Tier B FitLins CLI derivative parity | `bd-01KRGRF4X588F56QT21CPBB7NJ` | Narrow or remove the AR/statistic caveat once t and variance maps meet the standard gate without undocumented tolerance bypasses. |
| `spm-canonical-r-parity-divergence` | Tier A R parity suite (early 2026) | `tests/design/test_r_parity.py` SPMG-realised value tests | bbuchsbaum/fmrihrf SPM-canonical alignment ticket | fmrimod's default `basis="spm"` was aligned to the SPM/Nilearn standard parameterization (`delay=6, undershoot=16, dispersion=1, u_dispersion=1, ratio=0.167`); the R `fmrireg` package still uses the legacy `(p1=5, p2=15, a1=0.0833)` form. The R-side fixtures pin the legacy shape, so the SPMG-realised R parity tests are marked `xfail(strict=True)`. Exit: bbuchsbaum/fmrihrf adopts the SPM parameterization, fmrireg follows, R fixtures are regenerated, xfails removed. The legacy fmrimod path remains available via `basis="spm_legacy"`. |
| `spm-realized-column-sampling-convention` | Tier A SPM parity probes (early 2026) | Pattern A SPM-canonical workflows when frame_times conventions diverge | informational — no code fix planned | fmrimod samples at the *midpoint* of each TR window (``t = TR/2 + k·TR``) while Nilearn uses frame start (``t = k·TR``). Both are valid choices for representing slice-timed BOLD; neither is uniquely "correct" without slice-timing metadata. With matched grids fmrimod's and Nilearn's realised columns agree at correlation 0.9997 / max abs diff 1.3e-4 (the limit of finite-precision discrete convolution). Pattern A workflows that compare against Nilearn must pass `frame_times = arange(n) * TR + TR/2` to Nilearn, or accept the convention difference as expected. Exit: this caveat is informational; not slated for code change unless we add a configurable `slice_timing_offset` argument. |

## Maintenance Rule

Generated reports may introduce a new caveat only after this file is updated in
the same change. The test suite checks that every structured `caveat_id` in the
checked-in parity reports appears in this index.
Rows in this index must also have live owning work items unless they explicitly
name a closed replacement owner in the exit criterion.
