# 0005 — SPMG bases follow R fmrihrf

Status: Accepted (maintainer, 2026-10-25)
Date: 2026-10-25
Decides: R-source drift v2 steward decision **D2** (SPMG conventions);
row HRF-09; the `spmg1-x10-scale` caveat.

## Context

fmrihrf `daca7ba` ("fix: correct SPMG basis semantics") made R's SPMG
family the SPM double gamma it was always meant to be:

- `HRF_SPMG1` is `exp(-t) * (t^5 / 120 - t^15 / (6 * 15!))`, i.e.
  `gamma(6) - gamma(16) / 6`, over a 24 s span. Before, `A1 = 0.0833`
  made it ten times too large; that was the `spmg1-x10-scale` caveat.
- `HRF_SPMG2` adds the analytic time derivative `dh/dt`.
- `HRF_SPMG3` adds SPM's dispersion difference `(h(d) - h(d + 0.01)) / 0.01`
  (it had been the second time derivative).

fmrimod had meanwhile aligned its defaults with Nilearn: undershoot ratio
0.167, a 32 s span and, for SPMG2/3, SPM's finite difference in the delay
parameter as column 2. It kept the pre-`daca7ba` R form on
`SPMG{1,2,3}_HRF_Legacy` (`basis="spm_legacy"`, `"spmg1_legacy"`, ...).
The first full R parity run (`r_parity_run_2026-10-25.md`) traced all 19
R-backed failures to these choices: 15 to the ratio and span, 4 to column 2.

## Options

- **(a)** Follow R: ratio exactly 1/6, span 24 s, analytic `dh/dt`.
- **(b)** Keep the Nilearn conventions and record a permanent caveat.
- **(c)** Split: R's ratio and span, SPM's finite-difference column 2.

## Decision

**(a)**, and remove the `*_Legacy` classes.

- `SPMG1_HRF`, `SPMG2_HRF`, `SPMG3_HRF` default to `ratio=1/6`, `span=24`.
  The canonical keeps its typed SPM parameters (`delay`, `undershoot`,
  `dispersion`, `u_dispersion`, `ratio`); at the defaults it is R's closed
  form.
- Column 2 is `spmg1_derivative`, the analytic time derivative of the
  canonical for any parameter values. Column 3 of SPMG3 was already SPM's
  dispersion difference, which is what R now uses.
- `deriv()` is analytic for all three, matching R's `deriv.SPMG1_HRF`,
  `deriv.SPMG2_HRF` and `deriv.SPMG3_HRF` (the third column of the last is
  the time derivative of the dispersion column).
- The legacy classes, `spm_canonical_legacy`, the legacy-parameterized
  derivative helpers and the `*_legacy` basis names are removed. The
  legacy form was R's pre-fix ×10 bug, so nothing correct depended on it.

## Consequences

- All 65 R-backed `cross_testing/` tests pass with `FMRIMOD_REQUIRE_R=1`
  (19 failed before). The three bases and their `deriv()` agree with R to
  about 1e-14.
- The R fmridesign fixtures in `tests/design/fixtures/` are regenerated
  against fmridesign `4c02760` / fmrihrf `5b63d6a`. Six strict xfails that
  waited on this are gone; the remaining one is the modulator-centring
  difference noted in `r_source_drift_v2/fmridesign.md`, unrelated to HRFs.
- The `spmg1-x10-scale` caveat is retired (`CAVEAT_RETIREMENTS.md`).
- Nilearn parity changes slightly:
  - the canonical differs from Nilearn's only in ratio (1/6 vs 0.167) and
    kernel length (24 vs 32 s), and still correlates > 0.999;
  - the analytic column 2 correlates 0.999 with Nilearn's
    `spm_time_derivative` after rescaling (max residual 4.6% of its peak),
    inside the `tier_a_spm_derivative_basis` gate, which already allows
    rescaling for this column.
- Behaviour change for users: default SPM designs change slightly (span
  and ratio), SPMG2/3 column 2 changes shape, and `basis="spm_legacy"`
  (and the other `*_legacy` names) now raise as unknown bases.
