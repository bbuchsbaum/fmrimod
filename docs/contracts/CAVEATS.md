# Parity Caveats Index

This index lists every active parity caveat that appears in generated reports.
A caveat is not a tolerance escape hatch: it must name the affected quantity,
the current reason for divergence, an owning work item, and a concrete exit
criterion.

| Caveat ID | First appearance | Affected tiers | Owner | Exit criterion |
| --- | --- | --- | --- | --- |
| `fitlins-ar1-coefficient-binning` | `benchmarks/parity/tier_b_fitlins_bids/` | Tier B FitLins CLI derivative parity | `bd-01KRGRF4X588F56QT21CPBB7NJ` | Narrow or remove the AR/statistic caveat once t and variance maps meet the standard gate without undocumented tolerance bypasses. |

## Maintenance Rule

Generated reports may introduce a new caveat only after this file is updated in
the same change. The test suite checks that every structured `caveat_id` in the
checked-in parity reports appears in this index.
Rows in this index must also have live owning work items unless they explicitly
name a closed replacement owner in the exit criterion.
