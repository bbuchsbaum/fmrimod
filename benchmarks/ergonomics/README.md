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
`reports/ergonomics_report.json` (machine).

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
are also pinned as **strict xfail** probes in
`tests/test_benchmarks/test_ergonomics_vs_nilearn.py`. When you close a gap:

1. the probe XPASSes and fails the test run;
2. shorten the matching scenario body to the new idiom and drop its note;
3. regenerate the report (`test_checked_in_report_matches_scenario_bodies`
   fails until you do);
4. delete the probe.

## Adding a scenario

Add the two functions and a `Scenario(...)` row to `SCENARIOS`, give it an
agreement threshold you have observed with margin, add `large_inputs` if the
scaling behaviour matters, then regenerate the report.
