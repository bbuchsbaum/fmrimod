# Compat Module Retirement Inventory (v1)

**Owner:** `bd-01KRHTCQSMV6GKVGV9EXFAHJ9B`
**Board source:** `general-discussion/post-01KRHT4H90S9695V0PWRW32Q6Y` (bullet 2)
**Amended:** 2026-05-13 from
`major-issues-lets-talk/post-01KRHVA1NC98BP4KQ1Z29WYXWG` — reflects
`536677e glm: promote matrix fit helpers` and adds the missing fifth
compat surface (`fmrimod/stats/meta_compat.py`). Live LOC at HEAD: 1897
across five modules.

## Purpose

The five `compat`-named modules in `fmrimod/` total **1897 LOC** of code
whose shared header is some variant of "fmrireg-style compatibility
helpers." The scathing-read post asked an honest question: if
MISSION.md commits to a redesign, what are we backward-compatible with?

This inventory answers the question per export: name the destination,
say whether the module is a migration surface (R-named entry point
deliberately offered to ease porting), an internal bridge (used inside
fmrimod to translate between R-styled and Python-typed paths), or a
unique typed value (the compat label is a misnomer; the export should
be promoted into a typed module).

Rows name the typed replacement target plus a tier estimate. When a
typed value has already been promoted, the row remains as a compat-shim
receipt rather than pretending the old import path disappeared.

## Module summary

| Module | LOC | Public exports | Top-level promoted | Direct external callers | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `fmrimod/glm/compat.py` | 642 | 17 | 15 | 0 (flows via subpackage) | Migration surface; matrix-fit helpers promoted in `536677e` |
| `fmrimod/dataset/compat.py` | 464 | 16 | 13 | 3 | Migration surface; `LatentDataset` promoted to `dataset/latent.py` with compat shim |
| `fmrimod/simulate/compat.py` | 338 | 3 | 3 | 1 | Migration surface |
| `fmrimod/stats/meta_compat.py` | 296 | 5 | 5 | 0 (flows via subpackage) | Low-level R-name meta-analysis helpers |
| `fmrimod/ar/compat.py` | 157 | 2 | 0 (ar-level only) | 1 | Deprecated shim over typed constructors in `ar/plan.py` |
| **Total** | **1897** | **43** | **36** | **5** | |

"Direct external callers" counts files outside the compat module itself
and outside the subpackage's `__init__.py` re-export — i.e. real
end-user-style imports. Most names flow into product code via the
subpackage re-export plus the top-level `fmrimod` re-export, which is
why direct compat-module imports are sparse.

## fmrimod/glm/compat.py — 642 LOC, 17 public exports

**Module docstring (self-classification):** "Compatibility helpers for
fmrireg-style GLM workflows… intentionally thin: the stable Python
surface remains `fmri_lm` plus result methods, while these helpers make
migration code and parity tests less ambiguous."

| Export | Public-API impact | Class | Tier | Typed replacement target |
| --- | --- | --- | --- | --- |
| `SoftProjection` (class) | `fmrimod.glm` | Internal bridge | S1 | Lift to `fmrimod/glm/projection.py` (no R-name parity) |
| `soft_projection` | top-level | Migration surface | S2 | `Projection` from `fmrimod/glm/solver.py` with explicit options |
| `apply_soft_projection` | top-level | Migration surface | S2 | Method on the typed `Projection` |
| `compute_lm_contrasts` | top-level | Migration surface | S2 | `FmriLm.contrast(...)` + `ContrastResult` (already exists) |
| `compute_lm_contrasts_from_suffstats` | top-level | Migration surface | S2 | Typed `SuffstatBundle` consumer in `fmrimod/glm/` |
| `fit_contrasts` | top-level | Migration surface | S2 | `FmriLm.contrast(...)` (alias-only retirement: emit DeprecationWarning) |
| `fit_glm_on_transformed_series` | top-level | Migration surface | S2 | `fmri_lm(spec, dataset, config=FmriLmConfig(...))` with prewhitening |
| `fit_glm_with_config` | top-level | Migration surface | S1 | `fmri_lm(spec_or_model, config=...)` already covers this; redundant alias |
| `fmri_ols_fit` | top-level | Migration surface | S2 | `fmri_lm` with OLS engine (`config=FmriLmConfig()` default) |
| `fmri_rlm` | top-level | Migration surface | S2 | `fmri_lm` with `config=FmriLmConfig(robust=RobustOptions(...))` |
| `LowRankControl` (class) | top-level | Migration surface | S1 | `LowRankConfig` from `fmrimod/lowrank/` is already the typed equivalent |
| `lowrank_control` | top-level | Migration surface | S1 | Constructor for `LowRankConfig` |
| `paired_diff_block` | top-level | Migration surface | S2 | New `fmrimod.contrast` helper or `ContrastSpec` builder |
| `flip_sign` | top-level | Migration surface | S1 | Method on `ContrastResult` or `FmriLm` |
| `t_to_beta_se` | top-level | Migration surface | S1 | Free function in `fmrimod/stats/` |
| `hrf_smoothing_kernel` | top-level | Migration surface | S1 | Move to `fmrimod/hrf/` |
| `estimate` | top-level | Migration surface (no-op stub) | S0 | Delete; the `estimate(...)` call is a no-op that raises — surface confusion only |

**Matrix-fit helpers promoted (2026-05-13).** `fit_glm_from_matrix` and
`fit_glm_from_suffstats` were lifted from `glm/compat.py` to the typed
`fmrimod/glm/matrix.py` module in commit `536677e`. Top-level names
preserved via the re-export chain. The two rows previously listed at
"S0 to promote" in this section are closed. The remaining 17 exports
above are the live retirement surface.

**Representative callers (verified):**

- `tests/test_glm/test_contrast_to_neurovol.py`
- `tests/test_glm/test_fmrireg_compat.py`
- `tests/test_glm/test_combine_runs.py`
- `benchmarks/parity/tier_a_fiac/workflow.py`
- `benchmarks/parity/tier_a_localizer_fixed_effects/workflow.py`

## fmrimod/dataset/compat.py — 464 LOC, 16 public exports

**Module docstring:** "fmrireg-style dataset, IO, and benchmark
compatibility helpers."

| Export | Public-API impact | Class | Tier | Typed replacement target |
| --- | --- | --- | --- | --- |
| `fmri_mem_dataset` | top-level | Migration surface | S2 | `fmri_dataset(numpy_array, tr=..., events=...)` already does this |
| `LatentDataset` (class) | not top-level | Compat shim to typed value | S0 shipped | Canonical home is `fmrimod/dataset/latent.py` |
| `latent_dataset` | top-level | Compat shim to typed constructor | S1 shipped | Canonical constructor is `fmrimod.dataset.latent_dataset` |
| `fmri_latent_lm` | top-level | Migration surface | S2 | `fmri_lm` overloading on `LatentDataset` type |
| `voxel_index_chunks` | not top-level | Internal bridge | S1 | Move to `fmrimod/utils/chunking.py` |
| `extract_csv_data` | not top-level | Migration surface (IO) | S1 | Move to `fmrimod/io/csv.py` |
| `read_h5_full` | top-level | Migration surface (IO) | S2 | Typed `GroupDataReader` in `fmrimod/io/h5.py` |
| `read_nifti_full` | top-level | Migration surface (IO) | S2 | Typed `GroupDataReader` in `fmrimod/io/nifti.py` |
| `read_fmri_config` | top-level | Migration surface (IO) | S2 | Typed `FmriConfig.from_json(...)` — couples to Spec v1 |
| `register_basis` | not top-level | Internal bridge | S1 | Existing `fmrimod.extension_registry` or `hrf.registry` |
| `resolve_basis` | not top-level | Internal bridge | S1 | Same as above |
| `load_benchmark_dataset` | top-level | Migration surface | S2 | `fmrimod.examples.benchmarks` module (new) |
| `list_benchmark_datasets` | top-level | Migration surface | S2 | Same as above |
| `get_benchmark_summary` | not top-level | Migration surface | S1 | Method on the typed benchmark accessor |
| `create_design_matrix_from_benchmark` | not top-level | Migration surface | S2 | Spec-driven design build from typed benchmark |
| `evaluate_method_performance` | top-level | Migration surface | S2 | Belongs in `fmrimod/benchmarks/` not `fmrimod/dataset/` |
| `design_plot` | top-level | Migration surface | S1 | Move to `fmrimod/visualization/` |

**Cross-module misclassification:** `evaluate_method_performance` and
`design_plot` live in `dataset/compat.py` but their semantic home is
benchmarks and visualization respectively. Retirement is partly a
reshuffle.

## fmrimod/simulate/compat.py — 338 LOC, 3 public exports

**Module docstring:** "Compatibility simulation helpers mirroring
fmrireg names."

| Export | Public-API impact | Class | Tier | Typed replacement target |
| --- | --- | --- | --- | --- |
| `simulate_bold_signal` | top-level | Migration surface | S2 | `fmrimod.simulate.bold(...)` with typed `SimulationSpec` |
| `simulate_noise_vector` | top-level | Migration surface | S1 | `fmrimod.simulate.ar_noise(...)` (already typed in `noise.py`) |
| `simulate_fmri_matrix` | top-level | Migration surface | S2 | Compose the above; one full-pipeline typed builder |

**The most consolidated compat module:** simulate/compat is the
canonical "R names with Python bodies." Retirement here is the
cleanest test of whether top-level R-name promotion can be reversed
in this codebase without breaking real users.

## fmrimod/stats/meta_compat.py — 296 LOC, 5 public exports

**Module docstring:** "Low-level fmrireg-style meta-analysis matrix
helpers."

All five exports are top-level promoted via `fmrimod.__all__` (lines
610-614 in `fmrimod/__init__.py`); the lazy-loader map routes them to
`fmrimod.stats` (lines 321-324, 354). Retirement is therefore S2 by
the public-API auto-escalator, *not* S1 — correcting the initial
reading of this amendment.

| Export | Public-API impact | Class | Tier | Typed replacement target |
| --- | --- | --- | --- | --- |
| `fmri_meta_fit` | top-level | Migration surface | S2 | `fmrimod.group.meta_fe(...)` / `meta_re(...)` already canonical |
| `fmri_meta_fit_cov` | top-level | Migration surface | S2 | Method on the canonical native reducer |
| `fmri_meta_fit_contrasts` | top-level | Migration surface | S2 | Compose canonical reducer + `ContrastResult` |
| `fmri_meta_fit_extended` | top-level | Migration surface | S2 | Same as above with covariate inputs |
| `meta_effective_n` | top-level | Internal helper | S2 to promote | Move to `fmrimod/group/diagnostics.py`; numeric utility, not R-name parity |

**stats/meta_compat.py overlaps the native `fmrimod.group` reducers.**
The native path (`fmrimod.group.meta_fe`, `meta_re`, etc.) is the
canonical group-inference surface promised by MISSION.md § 35-39. The
`meta_compat.py` matrix helpers exist because the early `stats.meta`
slice landed before native group reducers stabilized. Retirement here
is "delete the R-shaped detour, keep `meta_effective_n` as a typed
diagnostic." S2 only fires if `stats.meta` is determined to be spine
(see bullet 7 in `major-issues-lets-talk/post-01KRHVA1NC98BP4KQ1Z29WYXWG`).

## fmrimod/ar/compat.py — 157 LOC, 2 public exports

**Module docstring after promotion:** "Deprecated location for
`plan_from_phi` / `whiten_with_phi`."

| Export | Public-API impact | Class | Tier | Typed replacement target |
| --- | --- | --- | --- | --- |
| `plan_from_phi` | not top-level (ar-only) | Deprecated compat shim | S1 shipped | Canonical home is `fmrimod/ar/plan.py` next to `WhiteningPlan` |
| `whiten_with_phi` | not top-level (ar-only) | Deprecated compat shim | S1 shipped | Canonical home is `fmrimod/ar/plan.py` next to `WhitenResult` |

**ar/compat.py is now accurately named.** The typed constructors live
beside the dataclasses in `fmrimod/ar/plan.py`; `ar/compat.py` only
preserves old import paths and emits `DeprecationWarning`.

## Aggregate findings

1. **Of 43 public compat exports, ~5 are pure R-name shims with no
   unique typed value:** `fmri_mem_dataset`, `fmri_ols_fit`, `fmri_rlm`,
   `simulate_bold_signal`, `simulate_fmri_matrix`. Retiring these
   genuinely shrinks the public-API footprint.
2. **The originally mis-housed typed constructors have been promoted:**
   the matrix-fit helpers in `536677e`, `LatentDataset` /
   `latent_dataset` into `fmrimod.dataset.latent`, and
   `plan_from_phi` / `whiten_with_phi` into `fmrimod.ar.plan`. The
   remaining typed-utility promotion candidate is `meta_effective_n`.
3. **36 of 43 exports are top-level promoted into `fmrimod.__all__`.**
   That means most retirement work is **S2** (auto-escalator fires on
   public-API change). Steward approval per AGENTS.md § Work requests.
4. **The compat modules are not internal bridges.** Module docstrings
   are explicit ("intentionally thin," "migration code," "mirroring
   fmrireg names," "low-level fmrireg-style"). Direct callers from
   inside fmrimod (excluding subpackage re-exports) total only **5
   files**. The bullet-2 critique of "internal-bridge debt" is
   structurally wrong; the right critique is "uncurated migration
   surface."
5. **`fmrimod/stats/meta_compat.py` overlaps the native
   `fmrimod.group` reducers.** Retirement here is partly a routing
   decision (bullet 7 of the bad-cop post): if `fmrimod.stats` is
   spine, `meta_compat.py` is debt to retire; if it is compat, the
   whole subpackage should carry the compat label, not just this file.

## Recommended follow-up beads

1. ~~**Promote `fit_glm_from_matrix` / `fit_glm_from_suffstats` out of
   `glm/compat.py`**~~ — **shipped in `536677e`** (typed `fmrimod/glm/matrix.py`
   module; top-level names preserved via re-export).
2. ~~**Promote `LatentDataset` out of `dataset/compat.py`**~~ —
   canonical home is now `fmrimod/dataset/latent.py`; old imports
   re-export the same objects.
3. ~~**Promote `plan_from_phi` / `whiten_with_phi` out of
   `ar/compat.py`**~~ — canonical home is now `fmrimod/ar/plan.py`;
   old imports are deprecated shims.
4. **Promote `meta_effective_n` out of `stats/meta_compat.py`** to
   `fmrimod/group/diagnostics.py`. **S0** (not top-level promoted; pure
   numeric utility).
5. **Delete `glm.compat.estimate`** — it is a no-op stub that raises.
   **S0** (no real callers verified).
6. **One epic per remaining R-name-shim cluster** (`fmri_ols_fit` /
   `fmri_rlm`; `fmri_mem_dataset`; `simulate_*`; the `*_lm_contrasts`
   family; the four `fmri_meta_fit*` helpers). Each epic carries a
   deprecation cycle and a typed replacement constructor before
   removal. **S2 per epic.**

The split makes the ~1897 LOC pile **tractable**: the typed-constructor
mis-housing is now mostly closed; the rest are deliberate migration
surfaces under steward gate.

## What this inventory does NOT do

- Does not delete any export.
- Does not break any public API; promoted constructors keep import shims.
- Does not enumerate every internal usage; representative callers per
  module are sampled to verify the classification.
- Does not propose timing; tier estimates are the only commitment.

The follow-up beads above are the actionable next moves.
