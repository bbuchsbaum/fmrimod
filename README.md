# fmrimod

Typed, composable fMRI design and statistical modeling in Python.

A design is a Python value, not a formula string: build it from `hrf(...)`,
`drift(...)` and `intercept(...)` terms, compose with `+`, and inspect every
field before a coefficient is solved. One library covers the whole path from
an event table to a group-level result — HRF specification, design matrices,
first-level GLM (with AR, robust and low-rank variants), single-trial
estimation, contrasts, and group statistics — with `numpy`/`pandas` types at
the boundaries and BIDS in and out.

## Documentation

- **[Documentation site](https://bbuchsbaum.github.io/fmrimod/)** — tutorials
  and API reference.
- [Get started](docs/get-started.qmd) · [Tutorials](docs/tutorials/) —
  the same material as source, if you would rather read it in the repo.
- [The golden path](docs/tutorials/golden-path.qmd) walks one dataset from an
  event table through a first-level fit, a contrast, and into a group
  analysis.

## Installation

```bash
pip install -e .
```

## Subpackages

Grouped by the four-stage spine `dataset -> design -> fit -> contrast/group`:

**Design and timing**

- `fmrimod.hrf` - HRF basis functions, registry, decorators, and generators
- `fmrimod.regressor` - Event-related regressors and convolution
- `fmrimod.events` - Event representations (factor, variable, matrix, basis)
- `fmrimod.basis` - Parametric basis functions (polynomial, spline, transforms)
- `fmrimod.design` - Design matrix assembly and EventModel
- `fmrimod.baseline` - Baseline and nuisance models
- `fmrimod.spec` - Typed, composable design specs (`hrf(...) + drift(...)`)
- `fmrimod.formula` - R-style formula parsing and DSL (legacy sugar)

**Data and fitting**

- `fmrimod.dataset` - Dataset containers and adapters (neuroim, nibabel, BIDS)
- `fmrimod.model` / `fmrimod.glm` - First-level GLM
- `fmrimod.ar` - AR modeling and whitening
- `fmrimod.robust` - Robust fitting
- `fmrimod.lowrank` - Sketch- and Nystrom-based solvers
- `fmrimod.single` / `fmrimod.betas` - Single-trial (LSS/LSA) estimation

**Inference and output**

- `fmrimod.contrast` - Contrast specification and F-tests
- `fmrimod.stats` / `fmrimod.group` - Group-level analysis
- `fmrimod.simulate` - Synthetic datasets with known effects
- `fmrimod.visualization` - Design matrix plotting
- `fmrimod.bids` - BIDS Stats Models translation and export

## Quick Start

A single regressor, convolved and sampled on a scan grid:

```python
import fmrimod

# Create an HRF
hrf = fmrimod.SPM_CANONICAL

# Build a regressor
reg = fmrimod.regressor(
    onsets=[1.0, 5.0, 10.0],
    hrf=hrf,
)

# Evaluate on a sampling frame
sf = fmrimod.SamplingFrame(blocklens=100, TR=2.0)
signal = reg.evaluate(sf.global_scan_times)   # -> (100,)
```

The library's actual spine is a *typed* design spec: build it with `+`,
inspect it before fitting, then name the hypothesis by condition rather than
by column position.

```python
import fmrimod as fm
from fmrimod.spec import hrf, drift
from fmrimod.contrast import condition

# The design is a value you can compose and inspect, not a magic string.
spec = hrf("condition", basis="spmg1") + drift("poly", degree=2)

fit = fm.fmri_lm(spec, dataset)

faces = fit.contrast(
    condition("face", term="condition") - condition("scene", term="condition"),
    name="face_minus_scene",
)
faces.estimate
```

See [the golden path](docs/tutorials/golden-path.qmd) for the same workflow
end to end, including where `dataset` comes from.

## Correctness

Numerical behaviour is checked against established implementations rather
than asserted. `benchmarks/` holds end-to-end parity workflows against
**Nilearn** and **FitLins** — design matrices, effect sizes, t/F statistics
and group results compared case by case, with the tolerances and receipts
checked into the repo. `cross_testing/` adds a workstream matrix covering
contrasts, variance and degrees of freedom, run combination, censoring,
single-trial estimation, rank-deficient designs, numeric precision and
residual diagnostics.

## Lineage

fmrimod consolidates capabilities that in R are spread across seven separate
packages — `fmrihrf`, `fmridesign`, `fmrireg`, `fmrilss`, `fmriAR`,
`fmrigds` and `fmridataset` — into one Python library with a single coherent
API. The R implementations serve as an executable specification for the
statistics, and their behaviour is cross-checked in `cross_testing/`, but the
types, composition and ergonomics here are designed for Python rather than
transliterated. Coming from those packages? See the migration guides below.

Migration guides:

- `docs/source/design/migration_guide.md` maps R `fmridesign` workflows to Python.
- `docs/source/design/fmrireg_migration.rst` maps R `fmrireg` workflows to Python.
- `docs/source/design/fmrilss_migration.rst` maps R `fmrilss` workflows to Python.

## Contributing & coordination

Agents and humans working in this repo share two coordination surfaces:

- [`AGENTS.md`](AGENTS.md) — canonical agent instructions (mote tracker, build
  & test, session protocol, commit conventions). `CLAUDE.md` symlinks to it.
- [`message_board.md`](message_board.md) — `mote discuss` ground rules and
  command crib for the public discussion board (`general-discussion` is the
  catch-all topic).

## License

GPL-3.0-or-later
