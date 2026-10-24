# Ergonomics & speed vs Nilearn

`benchmarks/parity/` asks *"does fmrimod compute the same numbers?"*. This
directory asks the critic's question: *"is fmrimod easier to write, faster,
and better documented for the same analysis?"* — and, where it is not, says
exactly what to fix.

```bash
.venv/bin/python -m benchmarks.ergonomics.workflow --large   # writes reports/
.venv/bin/python -m pytest tests/test_benchmarks/test_ergonomics_vs_nilearn.py
```

Results: [`reports/REPORT.md`](reports/REPORT.md) (human) and
`reports/ergonomics_report.json` (machine). `reports/round_N_report.json`
keeps each earlier round; the report's "fmrimod since ..." section compares
against the latest one (`--baseline` picks another).

## Rounds so far

- **Round 1** found fmrimod 26% longer than Nilearn in total (976 vs 775
  tokens), losing code size on 5 of 9 tasks, plus one crash bug (per-run event
  lists).
- **Round 2** closed those gaps: `fmri_lm` takes images directly, string
  level contrasts (`"face - house"`), `drift()` defaults to 128 s, per-run
  event lists and confound tables, `hrf(..., span=)`, typed `trialwise()`
  for single trial, `fmri_group_lm` on contrast results,
  `fm.design_matrix(spec, events=..., n_scans=..., tr=...)`, cached design
  columns. fmrimod is now 607 tokens against Nilearn's 775 and faster in
  every small-input row; the ~25k-voxel AR(1) row is close enough that
  repeated runs score it a tie or a narrow win.
- **Round 3** added four critic scenarios fmrimod had never been tested on
  — z-map to NIfTI, a batch of contrasts, a second level from effect maps
  on disk with group + age, and first level straight from a BIDS/fMRIPrep
  tree. On the round-2 API they put fmrimod back behind (1199 vs 1109
  tokens; 153 vs 66 for the z-map, 129 vs 81 for maps-on-disk, 242 vs 122
  for BIDS). Fixes: `ContrastResult.z` / `to_nifti(kind="z")`,
  `fmri_group_lm(paths, formula=, covariates=, mask=)` with
  `GroupLmResult.to_nifti`, `fm.bids_datasets`, confound tables carried
  on the dataset (`fmri_dataset(confounds=[...])` + `confounds("trans_x",
  ...)`), and the image shorthand on single-trial estimation. Now 854 vs
  1109 tokens, faster in all 15 timed rows.
- **Round 4** worked through the open notes without changing any default
  (OLS noise, TR/2 grid, opt-in drift, R column names and the 24 s FIR span
  are recorded as "By design" notes). Additions: `ar="ar1_voxelwise"`,
  `hrf(..., pool_modulators=True)` (Nilearn's pooled-modulator model),
  `names="short"` design-matrix labels, group accessors taking a bare level
  (`t_stat("patient")`), `covariates=` as a `participants.tsv` path, and a
  BIDS-to-group scenario (s14). Using Nilearn's AR(1) model in s06 exposed
  voxelwise AR 10-33x slower than Nilearn; a vectorised Yule-Walker plus a
  batched GLS solve (identical to ~1e-12) made it faster on small inputs and
  2.2x slower at ~25k voxels (Nilearn bins coefficients; fmrimod stays
  exact). fmrimod 965 vs Nilearn 1316 tokens over 14 scenarios.

## How a scenario is built

`scenarios.py` holds one function per library per task. Both start from the
same NIfTI images, mask, events and confounds (`inputs.py`, signal generated
with a plain-numpy HRF so neither library's code makes its own test data) and
stop at that library's natural result object.

Rules, so the comparison stays honest:

- Write each body the way a competent user of *that* library would. Use the
  shortest idiom the library actually supports; if you find a shorter
  fmrimod spelling, use it and regenerate.
- Imports go inside the body. Knowing what to import is part of an API's cost.
- No helpers inside a measured body. If the fmrimod body is clumsy, that is
  the finding — fix fmrimod, then shorten the body.
- A line ending `# parity-knob` exists only to align conventions so the two
  outputs can be compared (noise model, high-pass cutoff, sampling-grid
  origin, signal scaling). Knob lines are not charged to either side.
- Each scenario has an agreement gate (correlation of the output maps). The
  comparison only counts if both paths computed the same analysis.

## What is measured

| axis | how | winner when |
| --- | --- | --- |
| authoring | tokens in the body (statements, calls, keywords, imports also recorded); formatting-independent | ≥5% fewer tokens |
| speed | best of N warm calls (first call recorded separately) | ≥10% faster |
| noise | warnings emitted at the user | fewer |
| docs | for each library callable the body actually invoked (captured by a profiler hook): docstring, `Parameters`, `Examples` | ≥5 points higher |

The docs axis only sees docstrings. Nilearn keeps most usage documentation in
its example gallery, so its score here understates its docs; fmrimod's
equivalent is `docs/tutorials/`.

## Acting on the results

The report ends with a ranked **Deficiencies** list: every axis fmrimod
loses, plus curated friction notes (in each `Scenario.notes`) that name the
API gap and a candidate fix. Confirmed gaps that are mechanically checkable
should be pinned as **strict xfail** probes in
`tests/test_benchmarks/test_ergonomics_vs_nilearn.py`. When you close a gap:

1. the probe XPASSes and fails the test run;
2. shorten the matching scenario body to the new idiom and drop its note;
3. regenerate the report (`test_checked_in_report_matches_scenario_bodies`
   fails until you do);
4. turn the probe into a regression test asserting the short form equals
   the long one (sections 3-5 of that file hold rounds 2-4).

## Adding a scenario

Add the two functions and a `Scenario(...)` row to `SCENARIOS`, give it an
agreement threshold you have observed with margin, add `large_inputs` if the
scaling behaviour matters, then regenerate the report.
