# fmrimod

fMRI Signal Modeling: HRFs, Design Matrices, and Regression.

A unified Python library for fMRI experimental design and signal modeling,
combining hemodynamic response function (HRF) specification, design matrix
construction, and regression tools.

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

## Lineage

This package unifies Python ports of seven related R neuroimaging packages:

- **fmrihrf** (R) -> `fmrimod.hrf` + `fmrimod.regressor`
- **fmridesign** (R) -> `fmrimod.events` + `fmrimod.design` + `fmrimod.contrast` + ...
- **fmrireg** (R) -> `fmrimod.model` + `fmrimod.glm` + `fmrimod.robust` + `fmrimod.lowrank` + ...
- **fmrilss** (R) -> `fmrimod.single` + `fmrimod.betas`
- **fmriAR** (R) -> `fmrimod.ar` + `fmrimod.backends`
- **fmrigds** (R) -> `fmrimod.group` + `fmrimod.stats`
- **fmridataset** (R) -> `fmrimod.dataset`

It is not a mechanical port: the R behaviour is the specification, but the
Python types, composition and ergonomics are redesigned for Python. Numerical
and semantic parity is proven case by case in `benchmarks/` and
`cross_testing/`.

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
