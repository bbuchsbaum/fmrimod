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
library for the load-bearing workflow:

```text
fmri_dataset -> fmri_lm -> contrast -> group_fit
```

The docs should make the same commitments as the mission and vision:

- R behavior is the statistical specification, but Python API shape is
  redesigned around typed composition.
- Parity with R, Nilearn, and FitLins is the floor, not the ceiling.
- `neuroim-python` is the intended substrate for real image, mask, and contrast
  objects; nibabel belongs at I/O boundaries.
- Compatibility helpers and low-level matrix utilities are valid when clearly
  labeled, but flagship tutorials should prefer the public typed seam.
