# Parity Caveats Index

This index lists every active parity caveat that appears in generated reports.
A caveat is not a tolerance escape hatch: it must name the affected quantity,
the current reason for divergence, an owning work item, and a concrete exit
criterion.

Retired caveats are not deleted from history. They live in
[`CAVEAT_RETIREMENTS.md`](CAVEAT_RETIREMENTS.md), which records the retirement
commit, report path, red check, and owner/closing bead.

| Caveat ID | First appearance | Affected tiers | Owner | Exit criterion |
| --- | --- | --- | --- | --- |
| `dfres-n-minus-rank` | `benchmarks/parity/tier_a_multicollinear_baseline/` | Tier A multicollinear baseline; any rank-deficient parity surface | `bd-01KRHTASRWPA5ZQNGV55BS6XFE` | Nilearn's `run_glm` adopts `dfres = n - rank` for rank-deficient designs, or this divergence is recorded as permanent under the parity-as-exit-door rule. fmrimod's `fast_preproject` (`fmrimod/glm/solver.py:272-278`) uses the textbook `n - rank`; Nilearn uses `n - p` regardless of rank. fmrimod will not regress to match Nilearn — the residual-DoF math for rank-deficient X requires `n - rank`. |
| `typed-re-meta-regression-unimplemented` | `fmrimod/stats/meta.py:244` (no parity report; typed path raises before any benchmark runs) | Group-level random-effects meta-regression with subject covariates | `bd-01KRRJVG3YM80X0FX9C8KY5VG4` | Typed `fmri_meta` / `group_fit` accept a covariate formula under random effects (`method`/`tau2` in `pm`/`dl`/`reml`) instead of raising `NotImplementedError` (`fmrimod/stats/meta.py:244` `_solve_meta_wls`). Reference implementation exists in the R oracle: `~/code/fmrigds/R/reducers-core.R:348` `meta:re_reg` computes DerSimonian–Laird tau² for the regression case (`C = sum w - tr(H)`, `df = n - p`). Minimum exit: port the DL regression path so RE+covariate returns coefficients/SE/tau² matching `meta:re_reg`; the `fmri_meta_fit(Y, V, X, …)` matrix compat helper remains the documented interim path. |

## Maintenance Rule

Generated reports may introduce a new caveat only after this file is updated in
the same change. The test suite checks that every structured `caveat_id` in the
checked-in parity reports appears in this index.
Rows in this index must also have live owning work items unless they explicitly
name a closed replacement owner in the exit criterion.

Retired caveats are intentionally kept out of this active table.
