# Documentation Source Map

This directory has two documentation layers:

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
