# Mission

## What fmrimod is

**fmrimod is a typed, composable Python library for fMRI experimental
design and statistical modeling.** It is the unified Python home of the
seven sibling R packages that together cover the fMRI modeling stack:

| R package | Domain | Python home in fmrimod |
| --- | --- | --- |
| [`fmrihrf`](https://github.com/bbuchsbaum/fmrihrf) | HRFs, regressors | `fmrimod.hrf`, `fmrimod.regressor` |
| [`fmridesign`](https://github.com/bbuchsbaum/fmridesign) | Event/baseline design | `fmrimod.events`, `fmrimod.design`, `fmrimod.contrast`, `fmrimod.baseline` |
| [`fmrireg`](https://github.com/bbuchsbaum/fmrireg) | First-level GLM | `fmrimod.model`, `fmrimod.glm`, `fmrimod.stats` |
| [`fmrilss`](https://github.com/bbuchsbaum/fmrilss) | Single-trial (LSS) | `fmrimod.single`, `fmrimod.betas` |
| [`fmriAR`](https://github.com/bbuchsbaum/fmriAR) | AR modeling | `fmrimod.ar` |
| [`fmrigds`](https://github.com/bbuchsbaum/fmrigds) | Group stats & organization | `fmrimod.group`, `fmrimod.stats` |
| [`fmridataset`](https://github.com/bbuchsbaum/fmridataset) | Dataset containers | `fmrimod.dataset` |

The R behavior is the specification; the Python surface is not a
mechanical translation of it. We redesign types, composition, naming,
and ergonomics for Python, then *prove* parity in
[`./benchmarks/`](benchmarks/) and `cross_testing/` against the R sources
and Nilearn/FitLins.

## What we commit to deliver

1. **First-class design objects.** Events, baselines, HRFs, regressors,
   contrasts, AR specifications, single-trial estimators, and group
   reductions are dataclass-shaped Python values with explicit types
   and validation. No string-typed pipelines, no hidden global state.
2. **End-to-end first- and second-level analysis.**
   `fmri_dataset → fmri_lm → contrast → group_fit` is a single coherent
   path. Every node is independently testable, serializable, and
   inspectable.
3. **Native group inference.** `fmrimod.group` provides `GroupDataset`,
   `GroupSpace`, and the canonical reducer set (`meta_fe/re`,
   `ols_voxelwise`, `lmm_ri*`, Stouffer/Fisher/Lancaster, permutation
   tests) without requiring R. `fmrigds-r` exists only as a parity
   oracle and a transitional fallback.
4. **Use-case parity, validated in `./benchmarks/`.** Every workflow a
   user would actually run — SPM auditory, the FIAC localizer, a
   FitLins BIDS pipeline, a fmrireg-style second-level comparison —
   has a benchmark in `./benchmarks/parity/` that compares fmrimod
   output to the relevant R or Nilearn reference under typed
   tolerances. Drift in a use case is a regression.
5. **Parity with discipline.** Every named divergence from an external
   implementation lives in
   [`docs/contracts/CAVEATS.md`](docs/contracts/CAVEATS.md) with the
   affected quantity, an owning mote bead, and an exit criterion. New
   caveats land in the same commit that introduces them; the test suite
   enforces that.
6. **neuroim-python as the dogfooded substrate.** Volumes, surfaces,
   masks, and contrast outputs route through
   [`neuroim-python`](https://github.com/bbuchsbaum/neuroim-python), the
   Python port of `neuroim2`. We import nibabel only at the I/O boundary
   and treat awkward neuroim ergonomics as bugs in neuroim, not as
   reasons to bypass it.
7. **BIDS Stats Model interoperability.** Constrained BIDS Stats Model
   JSON nodes translate directly into `EventModel` + `BaselineModel` +
   contrast vectors so analyses can be authored and reproduced from a
   BIDS specification.
8. **Performance tracked, not gated.** Stage-level timing is recorded
   per commit and hardware tag via
   `benchmarks/performance/parity_trends.py` and `check_regression.py`.
   Correctness gates pass or fail; performance is reviewed.

## The non-negotiable bar

Parity with Nilearn and FitLins is the floor we ship from, not the
ceiling we aim at. On every workflow that lives in this repo, fmrimod
commits to *exceeding* the current Python state of the art on four
axes simultaneously:

- **Faster** — wall-clock end-to-end, on the same hardware and the same
  data, with the comparison recorded in `./benchmarks/`.
- **Better designed** — typed dataclasses where the alternatives use
  dicts and strings; composable values where the alternatives use
  stateful pipelines.
- **More elegant** — the intent of the analysis is legible at the call
  site, and the natural call sequence falls out of typed composition.
  Brevity is a *consequence*, not a target: we will not chase line
  count by introducing special-case shortcuts that hide real modeling
  decisions.
- **More powerful** — models, contrasts, AR structures, single-trial
  decompositions, and group reductions that the Nilearn/FitLins surface
  cannot express today.

When a comparable workflow in fmrimod is slower, uglier, less
expressive, or less composable than its Nilearn or FitLins counterpart,
that is a regression in fmrimod. We do not ship "matched" workflows.

## How we work

- **One thematic change per commit**, with a "why" body and a
  `Co-Authored-By` trailer when an agent authored the change.
- **Mote (`.mote/`) for issue tracking and coordination.** Every active
  caveat and every roadmap item has a bead. Reservations are taken
  before editing shared paths. See [`AGENTS.md`](AGENTS.md) for the
  workflow.
- **Tests are first-class evidence.** `python3.9 -m pytest tests/ -k
  "not rpy2"` is the always-on gate; `./benchmarks/` and
  `cross_testing/` parity runs are exercised when their inputs change.
- **Documentation is part of the API.** Quarto tutorials and the
  reference site in `docs/` are kept current with the surface; vignettes
  ported from the R sibling packages are the primary onboarding path.

## What is out of scope

- **Preprocessing pipelines.** Motion correction, registration, and
  normalization are upstream of fmrimod. We consume preprocessed BOLD;
  we do not produce it.
- **A general-purpose statistical computing framework.** fmrimod is
  fMRI-specific by design; the algebra it implements is in service of
  the modeling layer above.
- **Reinventing nibabel or neuroim.** fmrimod *uses* those containers;
  it does not duplicate them. When neuroim-python is the friction, we
  fix neuroim-python.

## How to tell if we are succeeding

- A new user reproduces a published Nilearn or FitLins analysis in
  fmrimod with stricter types and a clearer mapping to the underlying
  model, end-to-end, including group inference, without dropping into
  NumPy. Lower line count, when it appears, is a side effect of
  composition — not a forced design goal.
- Every active parity caveat in `docs/contracts/CAVEATS.md` either has
  a closed retirement bead or a dated note explaining why it is still
  open.
- A user-facing workflow in `./benchmarks/parity/` matches its R or
  Nilearn reference under the strict gate, not the caveat-bypassed gate.
- The `fmrigds-r` backend stops being on the critical path for any
  Tier C workflow.

---

See also: [`VISION.md`](VISION.md) for the world-state argument behind
these commitments, [`AGENTS.md`](AGENTS.md) for the operating manual,
and [`./benchmarks/`](benchmarks/) for the parity-validated use-case
workflows.
