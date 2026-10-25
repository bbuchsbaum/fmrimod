# Documentation Source Map

This directory contains the current site, internal records, and legacy sources:

- `*.qmd`, `tutorials/*.qmd`, and `reference/*.qmd` are the canonical Quarto
  site sources rendered by `docs/_quarto.yml`.
- `contracts/*.md` contains project contracts, parity caveats, audits, and
  migration plans. These documents are evidence and planning artifacts, not a
  second product narrative.
- `source/design/` and `source/hrf/` are legacy Sphinx source trees retained
  for migration provenance. They are not rendered by the current Quarto site.
  When editing them, keep their introductions subordinate to `MISSION.md` and
  `VISION.md`; do not describe fmrimod as a standalone mechanical port of one
  R package.
- `_freeze/`, `_site/`, and generated reference artifacts are build outputs.

Current user-facing docs should describe fmrimod as a typed, composable Python
library for the main workflow:

```text
fmri_dataset -> fmri_lm -> contrast -> group_fit
```

The docs should make the same commitments as the mission and vision:

- R behavior is the statistical specification, but Python API shape is
  redesigned around semantic composition.
- Parity with R, Nilearn, and FitLins is the floor, not the ceiling.
- `neuroim-python` is the intended substrate for real image, mask, and contrast
  objects; nibabel compatibility stays inside bounded I/O adapters.
- Compatibility helpers and low-level matrix utilities are valid when clearly
  labeled, but flagship tutorials should prefer the public modeling seam.

## Executable tutorial contract

Tutorial code follows a capability rule, not a package-name slogan:

- `neuroim` owns spatial image objects, reading and writing, masks, smoothing,
  ROI extraction, reconstruction, and export.
- NumPy and pandas remain the ordinary tools for simulation, tabular events,
  design matrices, and non-spatial feature arrays.
- The FIAC tutorial may use Nilearn's `fetch_fiac_first_level` to acquire the
  dataset and `plot_design_matrix` to display the externally owned design.
  That narrow exception does not authorize Nilearn masking or image utilities.
- Operational Python uses executable <code>```{python}</code> cells. A plain
  <code>```python</code> block must be immediately preceded by a
  `docs-schematic` comment that explains why it is not runnable. An
  `eval: false` cell likewise requires a `docs-nonexecuted` reason.

Run the source gate before rendering:

```bash
python scripts/audit_docs_source.py docs/tutorials
```

The gate parses Python fences. It rejects direct tutorial nibabel imports,
Nilearn image/masking APIs, fabricated identity-affine or all-ones-mask paths,
manual spatial reconstruction, and silently unexecuted snippets. Thin package
adapters remain allowed because this policy is scoped to reader-facing
tutorials rather than enforced by repository-wide token count.

## Model inspection and figures

The site teaches inspection alongside fitting: HRF shapes and uncertainty,
design structure and correlations, numerical rank, residuals, contrast weights
and explanations, trial estimates, and subject influence in group models.
Figures must identify simulated data, label quantities and units, and explain
what the display can and cannot establish. The image viewer provides an
enlargement path for narrow screens.

`_plot-setup.qmd` loads `assets/fmrimod.mplstyle` in each tutorial kernel.
The CSS and the two SCSS files own the reading layout and light/night
palettes. Fonts are served locally, with their license in `assets/fonts/`.
`DESIGN.md` at the repository root records the visual system.

## Build and verify

Use the project environment and the single `_quartodoc.yml` configuration:

```bash
uv pip install --python .venv/bin/python -e ".[dev,test,docs]"
cd docs
../.venv/bin/quartodoc build --config _quartodoc.yml
cd ..
.venv/bin/python scripts/normalize_quartodoc.py
```

Check which interpreter the `python3` Jupyter kernel actually starts. A user
kernel with that name can override the virtual environment even when
`QUARTO_PYTHON` is set. A temporary kernel avoids changing the user's setup:

```bash
.venv/bin/python -m ipykernel install \
  --prefix /tmp/fmrimod-docs-kernel --name python3
PATH="$PWD/.venv/bin:$PATH" \
QUARTO_PYTHON="$PWD/.venv/bin/python" \
JUPYTER_PATH=/tmp/fmrimod-docs-kernel/share/jupyter \
quarto render docs --execute --no-cache --execute-daemon 0
.venv/bin/python scripts/audit_docs_artifact.py docs/_site
```

For a fresh qualification run, move both `docs/_freeze` and
`docs/.quarto/_freeze` aside before rendering. `--no-cache` alone does not
disable project freezing. Preserve the old output until the render succeeds.
The post-render step removes Quarto 1.7's unused RequireJS loader from static
table pages while retaining it for actual AMD or Jupyter widget output.

Inspect representative tutorials and the API reference at desktop and 390 px,
including image enlargement, mobile navigation, search, and the theme toggle.
A local render does not publish the site; the Pages workflow deploys `main`.
