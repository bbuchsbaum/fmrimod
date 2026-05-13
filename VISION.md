# Vision

## The world we are building toward

Computational fMRI analysis in Python should feel like writing a thought
about the experiment, not assembling a pipeline. A researcher should be
able to express the *design they ran* — its events, its hemodynamic
assumptions, its AR structure, its baseline, its contrasts, its
single-trial decomposition, its grouping structure — in a few honest
lines of code, and have the system deliver group-level inference that is
correct, reproducible, and explainable down to the specific matrix that
was solved.

We see the same analysis written three ways today: a hand-rolled SPM
batch script, a Nilearn notebook that reaches into NumPy whenever the
high-level API runs out, and an R `fmrireg`/`fmrigds` pipeline that gives
real first-class designs but is unreachable from a Python stack. Each one
is a partial answer. None of them lets a thoughtful researcher *and* a
thoughtful engineer share the same artifact.

**fmrimod's vision is to be that shared artifact.** Specifically:

- **A unified Python ecosystem, not a transliterated R one.** fmrimod
  brings together the seven sibling R packages —
  [`fmrihrf`](https://github.com/bbuchsbaum/fmrihrf),
  [`fmridesign`](https://github.com/bbuchsbaum/fmridesign),
  [`fmrireg`](https://github.com/bbuchsbaum/fmrireg),
  [`fmrilss`](https://github.com/bbuchsbaum/fmrilss) (single-trial),
  [`fmriAR`](https://github.com/bbuchsbaum/fmriAR) (autoregressive
  modeling),
  [`fmrigds`](https://github.com/bbuchsbaum/fmrigds) (group stats and
  data organization), and
  [`fmridataset`](https://github.com/bbuchsbaum/fmridataset) (dataset
  containers) — under one cohesive Python surface. The R behavior is
  the spec; the Python shape is redesigned for Python.
- **Idiomatic re-imagining, parity-validated.** We are not doing a
  mechanical port. Types, composition, naming, and ergonomics are
  redesigned around what is elegant in Python; numerical and semantic
  agreement with the R sources is then proven case-by-case in
  `./benchmarks` use-case workflows and the `cross_testing/` parity
  matrix. Parity is a property of behavior, not of API shape.
- **Parity is the floor, not the ceiling.** Matching Nilearn and
  FitLins is the baseline fmrimod ships from, not the destination. On
  every workflow we touch we commit to *exceeding* the current Python
  state of the art on four axes — **speed, design, elegance, and
  power** — and we treat this as non-negotiable. A comparable
  workflow that is slower, uglier, less expressive, or less composable
  than its Nilearn/FitLins counterpart is a regression, not an
  acceptable trade-off. Brevity, where it appears, is an *emergent
  property* of good types and good composition — never the result of
  special-case shortcuts that hide real modeling choices from the
  user. We will sometimes be more verbose than Nilearn when the
  alternative is papering over a decision the researcher should be
  making explicitly.
- **Dogfooded neuroim-python as the data substrate.** fmrimod is built
  on [`neuroim-python`](https://github.com/bbuchsbaum/neuroim-python) —
  the Python port of `neuroim2` — and we use it the way our users will:
  if neuroim-python is awkward for an fmrimod workflow, we fix
  neuroim-python rather than route around it. Volumes, surfaces, masks,
  and contrast outputs round-trip through neuroim's typed containers so
  the same analysis can move between volumetric, surface, and basis
  spaces without a rewrite.
- **An fMRI analysis is a first-class object, not a script.** Event
  designs, baselines, HRFs, contrasts, run combinations, single-trial
  estimates, and group reductions are typed, composable Python values
  that can be inspected, diffed, serialized, and re-fit without
  rebuilding the world.
- **Group-level inference is native, not an R bridge.** `fmrimod.group`
  carries the canonical group-analysis contract; the R `fmrigds-r`
  backend exists only as an oracle and a transitional fallback while
  parity is being demonstrated.
- **Parity is a contract with an exit door.** Every divergence from a
  named external implementation (Nilearn, FitLins, SPM, the R sources)
  is written down with the affected quantity, an owner, and a concrete
  retirement criterion in
  [`docs/contracts/CAVEATS.md`](docs/contracts/CAVEATS.md). Caveats
  decay; they do not accumulate.
- **Reproducibility is operational, not aspirational.** Designs and fits
  carry the seed, the solver path, the HRF normalization knob, the AR
  configuration, the masking mode, and the version that produced them.
  Two people on two machines should be able to reconstruct the same
  result from a checked-in design specification alone.

If we succeed, the Python fMRI community stops treating "use the
high-level API until it breaks, then drop to NumPy" as the normal mode.
fmrimod is the high-level API that does not break, the place where new
modeling research lands first, and the canonical Python home of the
fmri\* R ecosystem.

---

See also: [`MISSION.md`](MISSION.md) for present-day scope and
commitments, [`AGENTS.md`](AGENTS.md) for the operating manual, and
[`./benchmarks/`](benchmarks/) for the parity-validated use-case
workflows that keep this vision honest.
