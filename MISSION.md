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

## First users

fmrimod is built first for users who already feel the limitations of
the current Python fMRI modeling stack:

- Methods developers who need to express HRFs, AR structure, contrasts,
  single-trial estimators, and group reductions as inspectable objects,
  not notebook-local conventions.
- Labs with repeated GLM and group-analysis workflows that need one
  checked-in analysis specification to survive across people, machines,
  and reruns.
- Python users who like Nilearn or FitLins until the high-level API
  runs out and the analysis drops into hand-shaped NumPy arrays.
- R `fmri*` users who want the same statistical seriousness in a Python
  stack without accepting a mechanical port of the old surface.

These first users do not need fmrimod to cover every historical corner
case on day one. They need the workflows it does cover to be clearer,
more inspectable, and harder to misuse than the alternatives.

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

## How we get there — the strategy

Our lead strategy is **flagship-benchmark-driven design**. We pick a
small, deliberately chosen set of end-to-end use cases that live in
[`./benchmarks/`](benchmarks/) and treat them as the simultaneous design
driver, correctness gate, performance baseline, and marketing artifact.
The API, the dataset/lm/contrast/group seam, and the neuroim-python
integration are all shaped by what those flagship workflows demand —
not by the union of what the R packages happen to export. We build the
library *backwards from the showcase*.

The flagship set covers the modeling stack from end to end:

- A canonical first-level fixed-effects analysis (SPM auditory).
- A multi-condition localizer fixed-effects analysis (FIAC).
- A FitLins-style BIDS Stats Model pipeline driven from JSON.
- A group meta-analytic / second-level workflow.
- A single-trial decoding study (LSS → multivariate readout).

Each flagship workflow does five things at once:

1. **Forces a coherent API** across the seven R sources by making
   incoherent seams impossible to write.
2. **Pressure-tests neuroim-python** as the real data substrate, so
   friction surfaces as bugs in neuroim-python (or in our usage), not
   as workarounds.
3. **Provides the side-by-side comparison** against Nilearn / FitLins /
   the R source on speed, lines, and expressiveness.
4. **Anchors the parity contract** with strict-gate numeric agreement
   recorded in `cross_testing/` and `docs/contracts/CAVEATS.md`.
5. **Doubles as the onboarding artifact** — the Quarto tutorial for
   each flagship is the canonical "how do I do X in fmrimod" answer.

Not every parity benchmark carries the same claim. We keep three
evidence levels distinct:

- **Numerical canaries** prove that a specific solver, contrast,
  transform, or statistical quantity matches a named reference under
  controlled inputs. These may use low-level arrays and exist to catch
  drift in the algebra.
- **Workflow parity cases** prove that fmrimod can reproduce an
  analysis through the intended public seam:
  `fmri_dataset → fmri_lm → contrast → group_fit`.
- **Flagship workflows** prove the full claim: fmrimod matches the
  reference numerically while being clearer, more typed, more
  inspectable, and at least performance-competitive.

Each flagship workflow must carry a proof artifact that a skeptical
reader can inspect without trusting the mission statement. That artifact
includes:

- The reference implementation being compared against.
- The public fmrimod code path used for the same analysis.
- Strict parity status, including every active caveat and its owning
  bead.
- Runtime and stage-level timing on the recorded hardware tag.
- The typed design, contrast, and group objects produced along the way.
- The modeling decision or output that fmrimod expresses more directly
  than the reference surface can.

A low-level canary is correctness evidence, not evidence that the
user-facing workflow is finished. Every benchmark that bypasses the
public typed API must say so in its report and name the public workflow
that should eventually replace or wrap it.

This strategy explicitly **deprioritizes function-by-function coverage**
of the R surface. Niche functions in `fmrireg`, `fmrigds`, or
`fmridataset` that no flagship workflow exercises may never appear in
fmrimod. That is the cost of "exceed, not match": we are not chasing
completeness, we are chasing leverage. Coverage gaps are surfaced
through user requests and parity drift, not preemptively closed.

## The baseline that makes agility compound

Our agility advantage over Nilearn and FitLins is real but conditional:
it only compounds on top of a baseline we are willing to keep building
on for years without rewriting. Agility is a multiplier on the
baseline, not a substitute for it. If we lock in the wrong shapes
early — string-typed dispatch, stateful pipelines, mechanical R-isms,
leaky neuroim seams — we inherit the same ossification we are trying to
exploit in the incumbents, just earlier in our lifecycle.

Four non-negotiables, therefore, on the baseline itself:

- **Types end-to-end, with no string-keyed escape hatches.** Every
  modeling decision (HRF, AR structure, contrast, baseline,
  reduction, correction) is a typed value, not a magic string. The
  dispatch table is the type system, not a registry of names.
- **Composition over monolithic fit functions.** No `do_everything()`
  entry points. `fmri_dataset → fmri_lm → contrast → group_fit` is a
  sequence of independently testable, individually inspectable nodes.
  Convenience wrappers may exist, but only as thin façades on top of
  the composable seam — never as the only way in.
- **neuroim-python as the only substrate.** No `nibabel` back doors in
  modeling code; no ad-hoc NumPy reshuffling that bypasses neuroim's
  typed containers. The boundary lives at I/O, period.
- **The four-stage seam is load-bearing, not convenient.** The
  `dataset → lm → contrast → group_fit` chain is the production
  pathway for every flagship workflow, not just one option among
  several. Alternative entry points must reduce to the same seam.

This is slower right now than transliterating R's API or wrapping
NumPy with thin helpers. That is the cost we pay for a baseline whose
agility compounds rather than decays. We accept it.

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
