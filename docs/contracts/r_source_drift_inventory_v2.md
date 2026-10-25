# R-source drift inventory (v2)

**Request:** 2026-10-24, direct user request: keep fmrimod in step with the
bug fixes, functionality and correctness work in the R sources, without
turning it into a mechanical port.
**Predecessor:** [`r_source_drift_inventory_v1.md`](r_source_drift_inventory_v1.md)
(2026-08-19; waves 1 to 3 ported in `3c6fc84`).
**fmrimod HEAD audited:** `4f14a19`.
**Epic:** `bd-01M5PS5JM4J8A2JKZX9WHN96QS`, filed from the v1 epic
`bd-01M0DQSE2QBF4TZ6G9DKYY8G2W`. `mote` was not available in the container
that produced this audit and wave 1, so the epic and its 40 child beads are
declared with fixed ids in
[`r_source_drift_v2/file_beads.py`](r_source_drift_v2/file_beads.py); run it
once from the repository root on a machine with mote (`--dry-run` first).
No board post exists.

The R behaviour is the spec; the Python surface is not a mechanical
translation. Rows record what changed upstream, what fmrimod does today, and
a suggested Python shape where it is not obvious. Some suggestions say to skip
an R feature because it does not fit fmrimod's design.

## Ranges audited

| Package | Range | Version at HEAD | Non-merge commits | R/src files | Detail |
| --- | --- | --- | --- | --- | --- |
| fmrihrf | `69be541`..`5b63d6a` | 0.4.0.9000 | 82 | 25 | [fmrihrf.md](r_source_drift_v2/fmrihrf.md) |
| fmridesign | `a94ade7`..`4c02760` | 0.6.1.9000 | 113 | 31 | [fmridesign.md](r_source_drift_v2/fmridesign.md) |
| fmrireg (fit core) | `063c8ef`..`0321e77` | 0.3.0.9000 | 84 of 166 | 50 of 92 | [fmrireg_fit_core.md](r_source_drift_v2/fmrireg_fit_core.md) |
| fmrireg (group, meta, output) | `063c8ef`..`0321e77` | 0.3.0.9000 | rest of 166 | 39 of 92 | [fmrireg_group_output.md](r_source_drift_v2/fmrireg_group_output.md) |
| fmriAR | `c7f370a`..`bb0789c` | 0.4.2 | 9 | 19 | [fmriar.md](r_source_drift_v2/fmriar.md) |
| fmrilss | `f851fb0`..`188d0dd` | 0.2.0 (dev) | 36 | 32 | [fmrilss.md](r_source_drift_v2/fmrilss.md) |
| fmrigds | `67cc585`..`cb5f44b` | 0.1.0.9002 | 32 | 68 | [fmrigds.md](r_source_drift_v2/fmrigds.md) |
| fmridataset | `a8bf021`..`58c9f4c` | 0.11.0.9000 | 140 | 78 | [fmridataset.md](r_source_drift_v2/fmridataset.md) |

fmridataset was not in v1. Its baseline (`a8bf021`, 2026-03-29) is inferred
from the features `fmrimod.dataset` carries; see the Baseline section of its
detail file.

## Method

- Each package range was indexed from `git log`, the `NEWS.md` diff and the
  changed testthat files. New testthat expectations were treated as the
  strongest signal of a behaviour change. Refactor-only commits (simplify
  series, Rcpp plumbing, CRAN housekeeping) are summarised in one paragraph
  per package and get no rows.
- Every status claim about fmrimod cites the `path:line` that was read at
  `4f14a19`.
- R was not installed. A row marked *verified* means a script transcribed the
  fixed R formula or R test into NumPy and showed the divergence against
  fmrimod HEAD. The 83 scripts are in
  [`r_source_drift_v2/repro/`](r_source_drift_v2/repro/); run them with
  `.venv/bin/python <script>`. They are one-shot evidence, not tests: each
  red check below names the test that should replace its script. All 83 ran
  cleanly at `4f14a19`.
- The R-backed `cross_testing/` suite was not run (`R PARITY: NOT CHECKED`).

Cheap-pass disqualifier (unchanged from v1): a stub, a no-op, or a fixture on
which the old and new formulas agree cannot close a row. Each detail file
states the disqualifier per red check.

### Classes

- **wrong-result**: fmrimod returns a different number or inference than the
  fixed R on valid input, and the R fix is correct.
- **contract**: validation, error or warning, default value, NA or edge-case
  handling, ordering or naming.
- **feature**: new capability with no fmrimod counterpart.
- **done**: already present at fmrimod HEAD.
- **n/a**: R-only plumbing, or moot under fmrimod's design.
- **S2** (tag): needs steward approval under `GOVERNANCE.md` § 10 because it
  changes a public signature, promotes a name, adds a caveat, or changes a
  seam.

## Counts

| Package | wrong-result | contract | feature | done | n/a | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| fmrihrf | 8 | 8 | 1 | 3 | 2 | 22 |
| fmridesign | 10 | 11 | 3 | 7 | 5 | 36 |
| fmrireg (fit core) | 13 | 6 | 2 | 6 | 4 | 31 |
| fmrireg (group, meta, output) | 7 | 5 | 5 | 6 | 4 | 27 |
| fmriAR | 12 | 10 | 5 | 3 | 4 | 34 |
| fmrilss | 3 | 1 | 5 | 4 | 2 | 15 |
| fmrigds | 7 | 5 | 3 | 5 | 3 | 23 |
| fmridataset | 6 | 8 | 6 | 4 | 3 | 27 |
| **Total** | **66** | **54** | **30** | **38** | **27** | **215** |

Five wrong-result rows restate a defect that another package's row already
records ([§ Cross-package merges](#cross-package-merges)), so there are about
61 distinct wrong results.

## Headline findings

1. **Most wrong results are fmrimod defects that new R tests expose, not
   only new R behaviour.** Upstream ran large review cycles in this range
   (fmrireg's "review 2026-10" streams, the fmrihrf and fmridesign simplify
   series, fmriAR 0.4.x) and added tests that pin behaviour fmrimod never
   had. Several defects predate the v1 pins: v1 inventoried NEWS-level items
   and did not read the AR, robust and contrast code at this depth. Each row
   says when the R behaviour arrived.
2. **Four formulas that wave 1 ported faithfully were corrected upstream
   afterwards**: the AR order BIC penalty (AR-01), parcel pooling (AR-04),
   the residual-bias correction solve (AR-05) and global pooling across runs
   (AR-12). A fifth, the unpivoted QR in the bias matrix (AR-08), was a
   wave 1 port defect. The v1 "`block_hrf` factory nit" is now a wrong result
   (HRF-01), and the B-spline basis construction changed again (HRF-02).
3. **R fmridataset deleted the surface fmrimod ports.** `f305b44`
   (2026-09-06) removed the pre-frame dataset architecture (constructors,
   backends, sampling-frame accessors, chunking, `fmri_series`, `fmri_group`,
   BIDS-HDF5 wrappers; about 37k lines). `fmri_frame` is now the only R
   container. No R code at HEAD corresponds to `fmrimod.dataset`; its R spec
   stops at `3ae565e`. See decision D1.
4. **The `spmg1-x10-scale` caveat no longer describes R.** fmrihrf `daca7ba`
   (2026-09-13) set SPMG1's A1 to 1/120 before CRAN 0.4.0, which matches
   fmrimod's scale. What remains is smaller: the undershoot ratio (0.167 vs
   1/6), span (32 vs 24) and SPMG2's second column (HRF-09). The AGENTS.md
   note about "twenty-two `cross_testing/` failures against fmrihrf 0.4.0"
   describes the pre-release 0.4.0. The caveat is not touched here: retiring
   it needs a replacement row or a regenerated receipt (AGENTS.md,
   retirement-without-tightening rule).
5. **fmrireg's group layer moved into fmrigds**: spatial FDR (`fdr:sabha`),
   the Welch t-test, `coef_contrast` and NIfTI t/df input. fmrimod's native
   `group/` layer is the matching home, and the REG-B and GDS rows that hit
   the same fmrimod code are merged below.
6. **Two structural roots explain many fit-core rows.**
   - `fmri_lm`'s AR path (`fmrimod/ar/integration.py:25-144`) bypasses the
     canonical `fmrimod.ar.fit_noise` / `whiten_apply`, which the
     single-trial path does use (`fmrimod/single/_prewhiten.py:311-335`). So
     the fmriAR fixes in lane 1a would not reach first-level fits (REG-A-02,
     REG-A-03, REG-A-12).
   - `iterative_gls` and `robust_refit` re-read raw run data instead of
     composing on the engine's preprocessed run matrices (REG-A-01,
     REG-A-04, REG-A-16).

   AGENTS.md asks that compatibility layers stay thin facades over one
   canonical implementation. Fixing the two roots closes the symptom rows
   and keeps them closed.

## Cross-package merges

These rows describe the same fmrimod code from two R packages. File one bead
for each group.

| Group | Rows | Shared fmrimod code |
| --- | --- | --- |
| HRF smoothing kernel | REG-A-10, REG-B-06 | `fmrimod/glm/compat.py:639-647` |
| t-to-z/p derivation | GDS-02, REG-B-05 | `fmrimod/group/ops.py:131-156` |
| Spatial FDR | REG-B-03, REG-B-04, GDS-06 | `fmrimod/stats/spatial_fdr.py` |
| NaN-safe BH | GDS-07, NaN part of REG-B-03 | `fmrimod/stats/inference.py:165`, `fmrimod/stats/spatial_fdr.py:265-266` |
| Absolute tolerances | REG-A-07 (covers REG-B-07); same theme as AR-10(a) | `glm/contrasts.py:726-731`, `glm/compat.py:358,458,479`, `glm/strategies.py:874-888` and others |
| Continuously sampled feature regressor | HRF-17, DES-23 | none yet (S2 only if promoted to `fmrimod.__all__`) |

Dependencies across packages:

- The residual-bias correction (AR-05, AR-08) feeds both REG-A-03 (first
  level) and LSS-04 (`residual_model="corrected"`).
- HRF-02 (B-spline basis) should land before the wave 4 `estimate_hrf`
  port, which uses the same basis.
- LSS-01 should land before the wave 4 SBHM `alpha_source` / `whiten_power`
  work.
- GDS-05's df rule applies to any `lmm:*_knownvar` port (wave 4).

## Wave map

### Wave 1: wrong results

Lanes run in parallel unless a dependency is listed.

| Lane | Rows | Depends on | Notes |
| --- | --- | --- | --- |
| 1a Noise model | AR-01 to AR-12 | none | Land first: first-level fits and single-trial prewhitening consume it. |
| 1b First-level AR routing | REG-A-02, REG-A-03, REG-A-12; censor crash in REG-A-04 | 1a | Route `fmri_lm` through `fit_noise` / `whiten_apply` (root 1 above). |
| 1c First-level composition and numerics | REG-A-01, REG-A-04, REG-A-05, REG-A-06, REG-A-07 (with REG-B-07), REG-A-08, REG-A-09, REG-A-10 (with REG-B-06), REG-A-11 | REG-A-04 pairs with 1b | REG-A-13 waits on D4. |
| 1d HRF | HRF-01 to HRF-05, HRF-07, HRF-08 | none | HRF-06 affects the CLI only and moves to wave 2. |
| 1e Design and contrasts | DES-01 to DES-10 | 1d for basis-dependent fixtures | DES-01: the silent drop of spec `contrasts=` becomes an error whatever D3 decides. |
| 1f Single-trial | LSS-01, LSS-02, LSS-03; unwhitened intercept (pre-existing bug P-1) | 1a | |
| 1g Group and meta | REG-B-01, REG-B-02, GDS-01, GDS-02 (with REG-B-05), GDS-03, GDS-04, GDS-05, spatial FDR group, GDS-07 | none | |
| 1h Dataset | DS-01, DS-03, DS-04, DS-05, DS-06; DS-02 routing fix | none | DS-02 can route through `group/space.py` `common_mask` without the S2 `VoxelSpace` change. |

Suggested starting order, by reach and size:

1. **REG-B-01**: with `weights="equal"` or `"custom"`, the meta standard
   error is 1/sqrt(n) whatever the data (`fmrimod/stats/meta.py:316,340`).
   It reaches `fmri_ttest(engine="meta")` and `group_fit`. The fix is small
   and local.
2. **GDS-05 and GDS-02**: between-subject LMM terms use N·K − p df (p
   values 7 to 10 times too small), and `derive` turns t into p with the
   normal distribution, ignoring df. Both make group maps too liberal.
3. **Lane 1a, then 1b**: the noise model is upstream of every prewhitened
   first-level and single-trial fit.
4. **DES-01 to DES-03**: term contrasts land on the wrong basis columns,
   `covariate(x, y)` builds one product column, and `where=` is ignored.
5. **DS-03 and DS-01**: per-run `start_time` collapses to run 1 (a silent
   grid shift), and multi-session BIDS runs share block ids.

### Wave 2: contracts

| Package | Rows |
| --- | --- |
| fmrihrf | HRF-06, HRF-11 to HRF-16: done, see "After wave 2"; HRF-10 waits on D8 |
| fmridesign | DES-11, DES-12, DES-13, DES-15 to DES-21: done, see "After wave 2" |
| fmrireg | REG-A-14, REG-A-16 to REG-A-19, REG-B-09, REG-B-10, REG-B-12: done, see "After wave 2" |
| fmriAR | AR-14 to AR-16, AR-18 to AR-22: done, see "After wave 2" (AR-13 done in wave 1) |
| fmrilss | LSS-04 (with LSS-01) |
| fmrigds | GDS-08 to GDS-12: done, see "After wave 2" |
| fmridataset | DS-07 to DS-14: done, see "After wave 1" |

### Wave 3: features without an S2 tag

HRF-17 / DES-23 (feature regressor, if kept out of `fmrimod.__all__`),
DES-22, DES-24, REG-A-20, REG-A-21, REG-B-14, REG-B-17 (plots, low
priority), AR-24, AR-25, AR-27, LSS-06, LSS-07, GDS-13, GDS-14, GDS-15,
DS-20 (S1). Each detail row gives a Python shape or a reason to skip.

### Wave 4: v1 items still open

| v1 item | Status at `4f14a19` | Note |
| --- | --- | --- |
| `estimate_hrf` smooth FIR | Open (`fmrimod/single/voxel_hrf.py:160-215`) | Land HRF-02 first. |
| `coef_images()` / contrast `coef_image` | Partly covered (`fmrimod/accessors.py:277-310`; `ContrastResult.to_nifti()`) | An image accessor on `FmriMetaResult` would fit with REG-B-13. |
| SBHM `alpha_source` / `whiten_power` | Open (`fmrimod/single/_types.py:263-291`) | Land LSS-01 first. |
| `lmm:*_knownvar` | Open; the R bridge runs only fe/dl | Use GDS-05's df rule. |
| Zero-event subset keeps canonical names | Open (`fmrimod/design/event_model.py:332-335`) | Decide with D3. |
| Shared-HRF C++ fast path | Close | R removed the path (`5ca6b9a`). |

### S2: steward

HRF-09, DES-14, DES-23 (and HRF-17 if promoted), REG-A-13, REG-A-15,
REG-A-28, the `cov_tri` order part of REG-B-02, REG-B-08 (if real REML is
kept), REG-B-11, REG-B-13, REG-B-15, REG-B-16, AR-17, AR-26, LSS-05,
LSS-08, LSS-09, the `VoxelSpace` part of DS-02, DS-15 to DS-19, DS-25.
The decisions they need are grouped below.

## Wave 1 status

Branch `claude/r-source-drift-v2`, 2026-10-24. Every fix below started
from its row's red check, failing before the fix and passing after; the
commit bodies cite the R commit and say which existing tests had pinned
the old behaviour and were updated. The detail files remain the audit
snapshot of `4f14a19`, so their status column is not edited.

| Lane | Rows fixed | Commits | Left open |
| --- | --- | --- | --- |
| 1a Noise model | AR-01 to AR-12; the AR part of REG-A-07 | `2cc2f60`..`6f1bff0` | none |
| 1b First-level AR routing | REG-A-02, REG-A-03, REG-A-04, REG-A-12; `fmri_lm` AR now runs through `fit_noise` / `whiten_apply` | `fc83a9b`..`abd5113` | REG-A-14 (ARMA, wave 2) |
| 1c First-level composition and numerics | REG-A-01, REG-A-05 to REG-A-11, REG-B-06, REG-B-07 | `a75b60f`..`8054393` | REG-A-13 (D4) |
| 1d HRF | HRF-01 to HRF-05, HRF-07, HRF-08; also the FIR basis end-of-span defect (same root as HRF-02) | `dbdabea`..`71a3e3c` | none |
| 1e Design and contrasts | DES-01 to DES-09 | `f7562ce`..`d932541` | DES-10 (D3) |
| 1f Single-trial | LSS-01 to LSS-04, P-1 | `537cbcb`..`6a147d0` | none |
| 1g Group and meta | REG-B-01 to REG-B-05, GDS-01 to GDS-07 | `fbe6cee`..`b197d6f` | see below |
| 1h Dataset | DS-01 to DS-06 | `5bc0bbd`..`3913893` | see below |

Partly fixed rows:

- **REG-B-02**: tau2 and Q_fe use the regression forms. The packed
  covariance order (`cov_tri` lower vs R's row-major upper) is unchanged
  because it alters a public output layout (S2).
- **REG-B-05**: t-only Fisher and Stouffer now use the t distribution.
  The Fisher statistic stays chi2 rather than R's signed z, which would
  change `group_fit`'s result contract.
- **GDS-06**: the wrong-result parts are fixed (missing p, weight
  normalisation, smoothing formula, `min_pi0` placement). z input with an
  empirical null and the `fdr:sabha` post-hoc registration are wave 3.
- **DS-01**: BIDS-HDF5 events now carry the scan position as `run` and
  the BIDS label as `bids_run`. A subject whose first scan has no events
  still shifts later runs, through P-2 (decision D10).
- **DS-02**: subjects are routed by voxel id through each subject's mask;
  the emitted space stays `SampleLabelSpace` (the `VoxelSpace` part is S2).

First-level AR (lane 1b) keeps fmrimod's scrub semantics for censored
volumes (decision D4) and differs from R in three places, recorded in its
commits: zero-weight volumes are left out of the AR residual df (R uses
n - rank), global phi is estimated from the preprocessed residuals (R uses
raw OLS residuals), and `reestimate_phi` with global pooling is refused
(R ignores it). The older public `fmrimod.ar.iterative_ar_gls` is
untouched apart from following the new `AROptions.pooling` default; it
still re-reads raw run data (decision D9, and REG-A-14 proposes routing
ARMA through it).

Also fixed while integrating: the top-level `fmrimod.matrix_dataset`
facade dropped `censor`, `start_time` and `slice_timing_offset`
(`5e75ccf`), and v1 is registered in both contract audits (`97c769c`).

Behaviour changes users will notice are listed in each commit body. The
largest are: equal and custom meta weights are relative (tau2 is NaN);
prewhitened single-trial fits use aggregate-trial residuals by default;
voxel HRFs have unit positive peaks; B-spline, tent and FIR bases are
zero at both ends of the span; `p="auto"` AR order selection counts
effective voxels and selects higher orders on large independent-voxel
data; first-level AR phi is estimated with fmriAR's residual-bias
correction (larger phi, so betas and t move), `AROptions.pooling`
defaults to R's `"run"` (what `fmri_lm` already did), `iter_gls` above 1
gives the same fit as 1, `pooling="parcel"` and voxelwise AR outside
R's narrow contract raise; factorial interaction contrast rows follow R's order; a zero SE
gives t = 0, p = 1 everywhere; and BIDS-HDF5 event `run` is an integer
scan position.

New follow-ups found during wave 1 (not inventory rows):

- `get_hrf("spmg1", n_basis=6)` reports `nbasis=6` but evaluates one
  column; a lagged HRF is truncated at the unlagged span; `trialwise()`
  ignores durations.
- `golden_tests/specs/core/hrf_bspline_basis.xml` still describes the
  old B-spline basis (no test runs it).
- For parcel pooling with `p="auto"`, fmriAR defaults to
  `pacf_weighted` multiscale pooling; fmrimod defaults to none.
- `cross_testing/test_tier_d_showcase` expects case ids that no longer
  exist (fails at `4f14a19` too).

## After wave 1

- **AR-03 R parity** (PR #22): the pooled Hannan-Rissanen ARMA agrees with
  fmriAR 0.4.2 to 2.5e-11 in `cross_testing/test_fmriar_arma_parity.py`.
- **AR-13 and AR-23** (`c50c542`): `fit_noise(method="arma")` selects its
  orders by BIC over `0..p_max` x `0..q_max` under `p="auto"` and the new
  `q="auto"` (`q_max=2`). AR-23's signature change was approved by the
  maintainer on 2026-10-24 (D5). Twenty cases agree with fmriAR, seven of
  them with automatic orders, including the selected order.
- **DS-07 to DS-14 and D1** (branch `claude/dataset-correctness-and-space`).
  The six standalone fixes:
  - DS-07: runs and a mask must share run 0's space (`9b7b795`);
  - DS-08 and DS-09: finite timing, non-negative dataset `start_time`,
    seconds compared at `sqrt(eps)` (`3b6218f`);
  - DS-12: an empty mask plans zero chunks (`939828e`);
  - DS-13: canonical subject ids (`d5d007e`);
  - DS-14: subsetting keeps parcel membership and basis projectors
    (`404a2c1`).

  D1 is decided as (c), recorded in
  [`decisions/0004-dataset-storage-and-space.md`](decisions/0004-dataset-storage-and-space.md).
  `FmriDataset` now reads through one `StorageBackend` with a typed
  `VolumeSpace` (`84708b4`), which fixes DS-10 (per-file run lengths) and
  DS-11 (NIfTI-backed datasets keep their space). Voxel-routed study
  estimates of image data lower to the common `VoxelSpace` (`1118e19`),
  closing DS-02's S2 part.

## After wave 2

All wave-2 rows except HRF-10 (waits on D8) landed on branch
`claude/wave2-drift-rows`, one commit per row. On that branch the
Python suite has only the three network/history-gated failures, and
`FMRIMOD_REQUIRE_R=1 pytest cross_testing/` runs 65 R-backed tests whose
failures are exactly the D2 baseline in
[`r_parity_run_2026-10-25.md`](r_parity_run_2026-10-25.md).

| Package | Rows and commits | Checked against R |
| --- | --- | --- |
| fmrigds | GDS-08 `efca281`, GDS-09 `8370463`, GDS-10 `3ef1b99`, GDS-11 `29892b4`, GDS-12 `2acd93f` | GDS-08's bounds read off `R/reducers-lmm.R` |
| fmrihrf | HRF-13 `ab38ff2`, HRF-11 `3a039de`, HRF-12 `870cac3`, HRF-16 `0c78a36`, HRF-14 `5c50407`, HRF-15 `be8d181`, HRF-06 `3380a25` | regressors agree once D2 is set aside |
| fmrireg | REG-A-19 `bf34b7a`, REG-A-18 `e7ed219`, REG-A-16 `108c951`, REG-B-10 `2190ac4`, REG-B-12 `da182e4`, REG-B-09 `d522b18`, REG-A-14 `2fcaa9d`, REG-A-17 `f43bb8c` | no |
| fmriAR | AR-14 `a09714a`, AR-15 `c1964b1`, AR-16 `fe51ad1`, AR-18 `cc5f895`, AR-19 `35ef65e`, AR-20 `ba95577`, AR-21 `2c9cb78` (test only), AR-22 `8fa630a` | AR-18, 19, 21, 22 against fmriAR 0.4.2 to 1e-10 (31 parity tests) |
| fmridesign | DES-11 `d71b6ed`, DES-12 `f339fa3`, DES-13 `61669de`, DES-15 `ab77d09`, DES-16 `d5878c5`, DES-17 `2eabce8`, DES-18 `db31cee`, DES-19 `5319e59`, DES-20 `11c11a6`, DES-21 `31c3f4c` | ports R's tests |

User-visible behaviour changes:
- `"betas"` is the canonical accessor target; `"estimates"` is deprecated.
- Uncomputable requested contrasts raise `ContrastError` (a `ValueError`)
  instead of warning and dropping the contrast. This includes
  `column_contrast` patterns that select different columns in the
  design-name and term-level namespaces (DES-12 met REG-A-18 at
  integration).
- `fmri_lm` with `q > 0`, `struct="arma"` or `method="arma"` fits a runwise
  ARMA GLS. It used to fit AR(p) silently.
- Covariate columns are named `<tag>_<variable>` (default tag `cov`).
- `basis_name` in column facts is the lower-case basis family.
- Term-level `column_contrast` patterns on multi-basis terms spread weight
  over the basis columns, as R does.
- Malformed censor indices, AFNI specs, plan/run-count mismatches,
  non-orthogonal design residuals and fractional block lengths now raise.

Judgement calls:
- DES-19: the maintainer decided (2026-10-25) that duplicate contrast
  names raise `ContrastError` (a `ValueError`) naming each repeated name
  and its positions, in `contrast_set()` after flattening and among the
  contrasts attached to one model term. This diverges from R, whose test
  keeps them (`contrast_set(a, contrast_set(a))` equals
  `contrast_set(a, a)`) and whose name-keyed results then overwrite; the
  R side should get the same fix.
- DES-20 includes the model's F-contrasts, as R does.
- HRF-12's named frame is opt-in (`as_frame=`), so the default return type
  is unchanged.
- REG-A-14 does not port R's 64-voxel estimation subset; it uses all
  finite voxels.

Follow-ups found during wave 2 (not inventory rows):
- `hr_arma`'s default `p_big`: fmriAR's C++ path uses `max(8, p + q + 5)`,
  its R fallback and fmrimod add `ceil(10 log10 n)`.
- `iterative_ar_gls` is public, unused and keeps the old per-run plan and
  censor handling; `fmri_lm` now uses `iterative_gls`.
- `AROptions.p_max` and `convergence_tol` are still ignored on the AR path
  (`censor` belongs to REG-A-15).
- `single/_prewhiten.py` forms residuals from an intercept-augmented design
  but passes `design=config.design` to `fit_noise`.
- Poly/BSpline event bases are refit per run even without `subset=`
  (`_subset_event_term`); belongs with D3.
- `!=` in pair-contrast formulas matched nothing, so
  `one_against_all_contrast` returned level-vs-baseline weights with a
  "Mask B is empty" warning. Fixed 2026-10-25: `!=` selects the other
  levels and other predicates (`in`, `and`, ...) are evaluated on the
  term's cells, as R does; one-vs-all now gives R's `c(1, -1/2, -1/2)`.
- `design_colmap` reports `modulation_type="amplitude"` for covariate
  columns; R reports `covariate`.

## Steward decisions

- **D1. fmridataset direction** (DS-15 to DS-19, DS-25, DS-02). The R spec
  for `fmrimod.dataset` is frozen at `3ae565e`. Options: (a) treat
  `3ae565e` as the permanent spec and port only correctness fixes; (b) adopt
  R's frame, array-source, feature-space and entity architecture; (c) keep
  the `fmrimod.dataset` API and adopt individual frame ideas where they fix
  real defects, such as spatial identity for DS-02 and DS-07. Recommendation:
  (c). Wave 1h and wave 2 do not depend on the choice. **Decided (c)** on
  2026-10-24; see
  [`decisions/0004-dataset-storage-and-space.md`](decisions/0004-dataset-storage-and-space.md).
- **D2. SPMG conventions** (HRF-09). Replace or retire `spmg1-x10-scale`;
  choose the undershoot ratio (Nilearn 0.167 or exact 1/6), the span (32 or
  24) and SPMG2's second column (onset-shift derivative like Nilearn and SPM,
  or analytic dh/dt like R); decide whether the `*_Legacy` SPMG classes stay.
  Re-baseline the AGENTS.md `cross_testing/` note once R is available.
  **Decided (a), follow R, and remove the legacy classes** on 2026-10-25;
  see [`decisions/0005-spmg-conventions.md`](decisions/0005-spmg-conventions.md).
  HRF-09 is done and `spmg1-x10-scale` is retired.
- **D3. Contrast semantics** (DES-01, DES-10, DES-14, v1 zero-event
  subset). Port R's embedding of term-attached contrasts, or retire them in
  favour of the fit-level contrast path. Decide whether subsetted
  `Poly`/`BSpline` bases are refit (fmrimod) or evaluated on all events (R),
  and whether excluded levels keep all-zero columns (fmrimod) or are dropped
  (R).
- **D4. First-level inference semantics** (REG-A-13, REG-A-15, REG-A-07).
  Runwise multi-run contrasts: per-run fixed-effects pooling as R does, or
  refuse. Censoring: scrub volumes (fmrimod, like AFNI `-censor`) or feed
  censoring to the AR model only (R). The t statistic when se is exactly 0:
  NaN or 0. Also pre-existing: fmrireg's default baseline is a B-spline drift
  basis, while fmrimod defaults to a run intercept, so default fits differ.
- **D5. Public surface** (REG-A-28, AR-17, AR-26, REG-B-11,
  REG-B-13, REG-B-15, REG-B-16, DS-20; AR-23 was approved and landed, see
  "After wave 1"). R deleted its unreachable bootstrap
  and effective-df helpers, but fmrimod still exports `bootstrap_glm` at
  top level with no R spec. The other rows add keywords or result fields to
  public functions.
- **D6. Group semantics** (REG-B-08, REG-B-02, GDS open questions). Keep
  fmrimod's real REML (more than R offers, which needs a caveat) or follow R
  and run Paule-Mandel with a message. Decide whether fmrimod's weighted
  `perm_onesample` (no R counterpart at HEAD) is intended, and whether to
  adopt R's packed-covariance order.
- **D7. Single-trial scope** (LSS-02, LSS-05, LSS-08, LSS-09). Port the
  voxel-HRF positive-peak contract in full or keep unit-L2 with a caveat;
  accept a k-means caveat for binned voxel AR; R1-GLM; GLMsingle as a native
  port, an adapter around the upstream Python package, or out of scope.
- **D8. `neural_input` semantics** (HRF-10). R now returns the stimulus
  density the convolution engine uses, which reverses the premise of
  fmrimod's docstring (`fmrimod/regressor/neural_input.py:45-55`).
- **D9. First-level AR routing** (lane 1b). Routing `fmri_lm` through
  `fit_noise` is internal but spans `ar/` and `glm/`. Grade it as S1 with a
  co-sign, or S2 under the multi-module refactor escalator.
- **D10. Run numbering for event-free runs** (P-2, rest of DS-01).
  `event_model` numbers block labels by first appearance among the
  events, so when a run has no events the later runs' events are
  convolved in the wrong run's window. fmridesign does the same
  (`canonicalize_blockids`, `match(x, unique(x))`), so this is a shared
  hazard, not drift. Keeping integer labels as given is ambiguous: labels
  `[1, 2]` could be 1-based runs 1 and 2 or 0-based runs 2 and 3, and a
  wrong guess misplaces events the other way. Options: map labels against
  the sampling frame's run ids when the caller supplies them; warn when
  the events name fewer runs than the frame has; or report it upstream
  and keep R's rule. Recommendation: the first two together, raised with
  the fmridesign maintainer.

## Pre-existing fmrimod bugs found (not drift)

The auditors found these while checking drift rows. None comes from an
upstream change, so they are listed apart from the waves. P-1 to P-3 and
P-12 were verified by script; the rest come from reading the code.

| ID | Bug | Where |
| --- | --- | --- |
| P-1 | With prewhitening, `include_intercept=True` adds an unwhitened intercept after whitening; betas differ by up to 0.999 (true betas about 1). | `fmrimod/single/__init__.py:143-154`, `fmrimod/single/lss.py:362-367`; `repro/fmrilss_intercept_whiten.py` |
| P-2 | Block ids are numbered by first appearance among events, so events of a run that follows an event-free run land in the wrong run. R does the same, but R rejects decreasing ids and fmrimod accepts them. Blocks a clean DS-01 fix. | `fmrimod/design/event_model.py:2473-2481`; `repro/fmridesign_blockids.py` |
| P-3 | Index normalisation is repeated per module and wrong in three places: `np.asarray(mask, dtype=np.intp)` turns bool masks into 0/1 indices in AR censoring (AR-06) and dataset selectors (DS-05), and `np.where(mask)[0]` on a 3-D mask gives x-coordinates as chunk voxel indices (DS-06). `fmrimod/dataset/backends/nifti_backend.py:26-48` and `fmrimod/dataset/series.py:219` already do it right; one shared helper would close all three. | rows AR-06, DS-05, DS-06 |
| P-4 | The dataset path of single-trial estimation adds no run intercepts for multi-run data. | `fmrimod/single/__init__.py:425,443` |
| P-5 | `io/results.py` writes an all-zero SE map when `ContrastResult.se is None` (for example F contrasts). | `fmrimod/io/results.py:593-608` |
| P-6 | `fmri_rlm` forces `robust.type="huber"` even when the config asks for bisquare. | `fmrimod/glm/compat.py:497` |
| P-7 | `compress_bids_study` silently discards `encoding=` and `template=`. | `fmrimod/dataset/bids_h5_write.py:427` |
| P-8 | Mask binarisation differs by path: `> 0` in `NiftiBackend`, `astype(bool)` (NaN becomes True) in `NeuroVecAdapter`. | `fmrimod/dataset/backends/nifti_backend.py:97`, `fmrimod/dataset/adapters/neuroim_adapter.py:257-258` |
| P-9 | `SamplingFrame.from_dict` defaults `start_time` to 0.0, not TR/2. | `fmrimod/sampling.py:347` |
| P-10 | `hr_arma(step1="burg")` is accepted and ignored. | `fmrimod/ar/hr_arma.py:132-143` |
| P-11 | Fit-level `ContrastSpec` resolution matches `Formula("A")` but not `Formula("cond == 'A'")` against realised names. | `fmrimod/glm/fmri_lm.py:770` |
| P-12 | SPMG2's second column correlates only 0.965 with Nilearn's time derivative, and the benchmark comment describing it is stale. | `fmrimod/hrf/derivatives.py:263-294`; `benchmarks/parity/tier_a_spm_derivative_basis/workflow.py:284-290` |

The detail files also list naming and ordering differences from R that are
deliberate or harmless (interaction cell order, factor-by-modulator column
names, centring of interaction modulators). Those need a decision, not a fix.

## Baseline test state

At `4f14a19` in this container, `pytest tests/ -k "not rpy2"` gives 3312
passed and 7 failed. None of the failures comes from this audit.

- Two doc-audit tests fail because v1 was not registered in
  `contract_doc_audit.md` and `CONTRACT_AUDIT.md`. Fixed in the same branch
  as this document, which leaves 4 failures.
- `test_internal_audit_counts_track_live_probe`: `internal_any_audit.json`
  was not regenerated after `3c6fc84` added about 40 functions.
- `test_retirement_commits_exist` needs commits that a shallow clone does
  not have.
- Two Tier A benchmarks download data from osf.io, which the container's
  network policy blocks. The SPM-auditory one deletes its checked-in report
  when the download fails, so `test_retirement_receipt_has_expected_rows_and_shape`
  also fails when it runs afterwards (order-dependent). Restore the report
  with `git checkout -- benchmarks/parity/` after a run without network.

## Proposed beads

This was the plan at audit time. The beads actually filed, with their ids,
priorities and bodies, are in
[`r_source_drift_v2/file_beads.py`](r_source_drift_v2/file_beads.py): the
epic hangs off the v1 epic, every other bead off the v2 epic, and the
wave 1 lane beads are filed closed with their commits. Priorities follow
mote (0 highest).

| Proposed title | Pri | Rows | Red check home |
| --- | --- | --- | --- |
| Epic: R-source drift v2 | 1 | this document | none |
| Noise model: port fmriAR 0.4.x estimator fixes | 1 | AR-01 to AR-12 | `tests/test_ar/` |
| Route fmri_lm AR through fit_noise and whiten_apply | 1 | REG-A-02, REG-A-03, REG-A-12, REG-A-04 (censor) | `tests/test_glm/`, `tests/test_ar/` |
| First-level composition: weights, projection, robust on preprocessed runs | 1 | REG-A-01, REG-A-04, REG-A-08, REG-A-11 | `tests/test_glm/`, `tests/test_robust/` |
| First-level numerics: scale invariance, QR solve, sketch inference, factorial rows, smoothing kernel | 2 | REG-A-05, REG-A-06, REG-A-07, REG-A-09, REG-A-10, REG-B-06, REG-B-07 | `tests/test_glm/`, `tests/test_lowrank/` |
| HRF basis and convolution fixes | 1 | HRF-01 to HRF-05, HRF-07, HRF-08 | `tests/hrf/` |
| Design and contrast placement fixes | 1 | DES-01 to DES-10 | `tests/design/`, `tests/test_contrast/` |
| Single-trial prewhitening and voxel-HRF fixes | 1 | LSS-01 to LSS-04, P-1 | `tests/test_single/` |
| Group p-values: t-to-p df, LMM df, Lancaster tail, permutation count | 1 | GDS-01, GDS-02, GDS-04, GDS-05, REG-B-05 | `tests/test_group/` |
| Meta weights and validity | 1 | REG-B-01, REG-B-02, GDS-03 | `tests/test_stats/`, `tests/test_group/` |
| NaN-safe multiple comparisons and spatial FDR | 1 | GDS-06, GDS-07, REG-B-03, REG-B-04 | `tests/test_stats/` |
| Dataset timing, run identity and selectors | 1 | DS-01 to DS-06, P-2, P-3 | `tests/test_dataset/` |
| Contracts, one bead per package | 2 | wave 2 table | per package |
| Features, one bead each | 3 | wave 3 list | per row |
| Steward decision note D1 to D10 | 1 | S2 list | `docs/contracts/decisions/` |
| Pre-existing bugs P-4 to P-12 | 2 | pre-existing table | per row |
| SPM-auditory benchmark deletes its checked-in report when the download fails | 3 | baseline failure | `tests/test_benchmarks/test_parity_receipts.py` |
| Regenerate `internal_any_audit.json` | 3 | baseline failure | `tests/test_public_api/test_api_inventory.py` |

## Not checked

- No R code was run for the audit or wave 1; `cross_testing/` was not run
  against R. One exception came after wave 1 merged: the ARMA path (AR-03,
  pooled Hannan-Rissanen with censoring, both poolings, `hr_iter` and the
  order grid) agrees with fmriAR 0.4.2 (`bb0789c`, R 4.3.3) to 2.5e-11 in
  `cross_testing/test_fmriar_arma_parity.py`. The rest of fmriAR and the
  other six packages are still unchecked against R.
- GitHub issues and pull requests of the R packages were not read, only
  their git history, NEWS and tests.
- `pybids` is not installed, so the BIDS-directory `space=` filter path is
  unverified (DS open questions).
- HDF5 files written by R `write_gds_h5` were not read into fmrimod; the
  axis order through hdf5r and h5py is unverified (GDS open questions).
- REG-A-03 uses fmrimod's own wave 1 `fit_noise(design=)` port as the
  reference for R's corrected AR estimator, and the LSS-01 red check uses
  fmrimod's `fit_noise` as its AR estimator. Both rest on v1's
  `tests/test_ar/test_estimator_correctness.py`.
