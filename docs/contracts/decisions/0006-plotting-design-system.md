# 0006 — One plotting design system, ported from the R plot overhaul

Status: Accepted (maintainer request, 2026-10-25)
Date: 2026-10-25
Decides: how fmrimod follows the plotting overhaul in R fmrihrf, fmridesign
and fmrireg (drift v2 rows HRF-21, DES-24, DES-35, REG-B-17, REG-B-27,
previously marked n/a or "skip for now").

## Context

The R packages rebuilt their plots on ggplot2: fmrihrf (`R/plot-style.R`,
`R/plot-methods.R`), fmridesign (`R/plot_theme.R`, `plot_event_model.R`,
`plot_heatmaps.R`, `plot_baseline.R`) and fmrireg (`R/plot-helpers.R`,
`plot-fmri_lm.R`, `fmri_model.R`). They share a design language (left-aligned
bold titles, explanatory captions, recessive grids, direct labels, runs as
hairlines with a "Run k" top axis, diverging fills about zero) but each has
its own palette, ink and theme.

fmrimod had 16 matplotlib functions, each styled ad hoc (default colour
cycle, `viridis`, `RdBu_r`, red dotted onsets), inconsistent return types
(`Axes`, `(Figure, Axes)`, `None`), no plots for fits, `FmriModel`, contrasts
results or groups, and two bugs (`design_map(event_model)` never drew run
separators; it failed on `FmriModel`).

## Decision

1. **One design system for all of fmrimod** (`fmrimod/visualization/style.py`),
   not three. Tokens follow fmridesign's `plot_theme.R`, the most complete
   R system: ink (`#1C232E` text … `#B8BEC8` zero line), a 7-colour
   categorical palette assigned by design position and never cycled, the
   7-stop diverging ramp with a neutral warm-grey midpoint, the magnitude
   ramp, and an ordered ramp past seven series; all interpolated in CIE Lab
   as R does. A condition keeps its colour from design plot to fit plot.
2. **matplotlib**, already a hard dependency, drawn under a theme rc applied
   inside every plot function (`theme_rc`), so figures look the same whatever
   the user's global style. `set_theme()` overrides it (R
   `options(fmrireg.plot_theme)`); `theme()` styles user figures to match.
3. **Shared idioms** in `fmrimod/visualization/_helpers.py`: figure header and
   caption, run axis, m:ss time axis, fmrihrf's tick budget (an exact port,
   tested against R), onset rugs and event bars, haloed direct labels, and
   `column_info()`, one table of display labels (`face × high · temporal
   deriv.`) for any model or fit.
4. **Return type**: every public plot function returns a matplotlib `Figure`;
   single-panel functions accept `ax=` and draw into it.
5. **Numbers in pure functions**: VIF, contrast SE, HRF confidence bands,
   partial R², the residual-ACF null band and portmanteau test, and so on are
   separate functions with numeric tests against independent computations;
   renderers only draw. Plot tests check structure and numerics, not pixels
   (as the R packages do).

## Ported surface

- fmrihrf: `plot_hrf(s)`, `plot_regressor(s)`, `HRF.plot`, `Regressor.plot`,
  `hrf_curves`, `regressor_curves`.
- fmridesign: event-model plot (stacked / overlay / heatmap), `design_map`,
  `correlation_map` with VIF, `plot_contrasts` with sum and SE margins,
  baseline-model and sampling-frame plots.
- fmrireg: `plot_fit` / `FmriLm.plot` (betas, contrasts, hrf, timecourse,
  residuals) and `plot_fmri_model` / `FmriModel.plot`.

## Not adopted

| R feature | Disposition |
| --- | --- |
| Three per-package palettes and themes | One system (above). |
| fmrihrf's base-graphics fallback and `fmrihrf_plot` data-frame return | Python returns the `Figure`; the data are available from the pure builders (`hrf_curves`, `design_correlations`, ...). |
| fmrireg `design_plot()` Shiny/plotly app | Not ported; optional interactive backends can come later behind an extra. |
| vdiffr-style snapshots | None in R either; structure and numeric tests instead. |

## Consequences

- Breaking: plot functions return `Figure` instead of `Axes` or
  `(Figure, Axes)`; `**kwargs` passed straight to matplotlib are gone
  (styling comes from the theme).
- Drift v2 rows HRF-21, DES-24, DES-35, REG-B-17 and REG-B-27 are done.
