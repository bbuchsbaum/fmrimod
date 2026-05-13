# Parity Caveats Index

This index lists every active parity caveat that appears in generated reports.
A caveat is not a tolerance escape hatch: it must name the affected quantity,
the current reason for divergence, an owning work item, and a concrete exit
criterion.

| Caveat ID | First appearance | Affected tiers | Owner | Exit criterion |
| --- | --- | --- | --- | --- |
| `dfres-n-minus-rank` | `benchmarks/parity/tier_a_multicollinear_baseline/` | Tier A multicollinear baseline; any rank-deficient parity surface | `bd-01KRHTASRWPA5ZQNGV55BS6XFE` | Nilearn's `run_glm` adopts `dfres = n - rank` for rank-deficient designs, or this divergence is recorded as permanent under the parity-as-exit-door rule. fmrimod's `fast_preproject` (`fmrimod/glm/solver.py:272-278`) uses the textbook `n - rank`; Nilearn uses `n - p` regardless of rank. fmrimod will not regress to match Nilearn — the residual-DoF math for rank-deficient X requires `n - rank`. |

## Maintenance Rule

Generated reports may introduce a new caveat only after this file is updated in
the same change. The test suite checks that every structured `caveat_id` in the
checked-in parity reports appears in this index.
Rows in this index must also have live owning work items unless they explicitly
name a closed replacement owner in the exit criterion.
