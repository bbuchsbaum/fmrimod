# Second-Level Parity Contract v1

Status: Draft (frozen semantics for Python interface phase 1)

## Goal

Define a stable second-level contract for `fmrimod` that:

- preserves core semantics from `fmrireg` / `fmrigds`,
- allows a Python-first API surface,
- minimizes internal drift by keeping parity behavior explicit.

## Canonical Request

`group_fit` accepts a `GroupData`, a pandas DataFrame (validated to a
frozen `GroupData` via the typed `group_data_from_csv`; pass its
`effect_cols` schema), or an explicit `GroupFitRequest`. The default
fit needs no `GroupFitRequest`; it remains the explicit form for
non-default fits (custom `formula`/`model`/`backend`/…) and is still
importable (`from fmrimod.stats import GroupFitRequest`) though it is
no longer advertised in `fmrimod.stats.__all__`.

Core fields:

- `data`: `GroupData`
- `formula`: model formula, default `"~ 1"`
- `model`: `"meta"` or `"ttest"`
- `effects`: `"fixed"` or `"random"`
- `tau2`: `"dl" | "pm" | "reml"` (used when `effects="random"`)
- `method`: optional explicit method override (canonicalized)
- `weights`: `"ivw" | "equal" | "custom"`
- `correction`: `None | "bh" | "by" | "spatial"`
- `group_ids`: required for `"spatial"` correction
- `backend`: `"auto" | "python" | "fmrigds-r"`

## Canonical Result

`GroupFitResult` returns canonical numeric arrays:

- `estimate`, `se`, `statistic`, `p`, optional `q`, optional `tau2`
- `predictor_names`, `feature_names`
- `model`, `method`, `formula`, `backend`
- `metadata` for backend/correction provenance

## Axis Convention

For v1 canonical outputs:

- `estimate`, `se`, `statistic`, `p`, `q` use shape `(feature, parameter)`.
- One-parameter outputs (for example intercept-only t-test) are represented as `(feature, 1)`.

## Parity Mapping

### Method aliases

- `fixed` -> `fe`
- `random` -> `dl`
- `meta:fe` -> `fe`
- `meta:re` -> `dl`
- canonical random-effects methods: `dl`, `pm`, `reml`

### Weight aliases

- `1/var` -> `ivw`
- `inverse_variance` -> `ivw`

### Correction aliases

- `fdr:bh` -> `bh`
- `fdr:by` -> `by`
- `fdr:spatial` -> `spatial`

### Backend aliases

- `r` -> `fmrigds-r`

## Behavior Guarantees (v1)

- Existing `fmri_meta` and `fmri_ttest` behavior remains unchanged.
- `group_fit` delegates to parity implementations after normalization.
- Canonical `GroupFitResult.p` values are two-sided unless a future estimator
  explicitly documents a different alternative.
- Nilearn `SecondLevelModel` reports signed one-sided p-values for t maps.
  Parity workflows that compare this quantity must name it separately (for
  example `age_p_signed_one_sided`) and must not map it onto native
  `p_coef:<term>` assays.
- Correction is applied post-fit on canonical p-values, not on separately
  named Nilearn signed-p parity quantities.
- Spatial correction requires explicit grouping (`group_ids`) in v1.
- `backend="fmrigds-r"` is available as an explicit R fmrigds oracle/fallback with capability checks.
- `backend="fmrigds"` is retired from dispatch to avoid treating the R bridge as the native production path.
- `available_second_level_backends()` lists only production-native names by
  default: `"auto"` and `"python"`. Oracle/fallback names are opt-in via
  `available_second_level_backends(include_oracles=True)`.

## Out of Scope (v1)

- Full parity for all tau2 estimators in fmrigds bridge (`pm`/`reml` are pending).
- New statistical estimators beyond currently implemented Python parity code.
- Behavioral changes to existing wrappers.

## Non-CSV Preparation (v1)

The canonical request contract supports non-CSV `GroupData` formats:

- `h5`: paths/mask/contrast/stat are represented in bridge payloads.
- `nifti`: beta/se/var/t maps and df metadata are represented in bridge payloads.

Bridge execution remains conservative in v1:

- Executed path: CSV meta/ttest parity.
- Prepared path: payload mapping and tests for `h5` and `nifti`.

## Drift Control Rules

- Any new second-level feature must update this contract first.
- Alias changes require explicit contract update and tests.
- Semantics changes require parity fixtures against R references before merge.
