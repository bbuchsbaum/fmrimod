# Trio Pending API Workplan v1

> **ARCHIVED — pre-MISSION-rewrite snapshot (2026-05-11).**
> This workplan drove inventory `pending` rows to zero. The current
> [`MISSION.md`](../../../MISSION.md) explicitly *deprioritizes*
> function-by-function R coverage in favor of flagship workflows, so
> "drive pending to zero" is no longer the gating contract.
> Cross-module work is now coordinated through the governance loop in
> [`GOVERNANCE.md`](../../../GOVERNANCE.md) and mote beads.

Status: Archived 2026-05-13 — superseded by MISSION / GOVERNANCE.
Original status when written: Active

## Purpose

This workplan turns the pending rows from
`docs/contracts/trio_api_inventory_v1.md` into tracked mote work. It avoids
leaving the inventory as an unowned checklist while still keeping the work
grouped into reviewable chunks.

## Tracking Issues

| Mote ID | Area | Pending rows covered |
| --- | --- | --- |
| `bd-01KRCSGJEY1SASN3Y02CX802TQ` | verification transcript | Unit tests, golden tests, R parity tests, core parity matrix, benchmark schema validation, and docs build transcript |

## Resolved Issues

| Mote ID | Resolution |
| --- | --- |
| `bd-01KRCSG4YXF1HGP1CZRWC39BS6` | Resolved `Fcontrasts`, `condition_map`, `evaluate`, `global_onsets`, and `samples` for `fmridesign`; the corresponding inventory rows now have stable Python surfaces and focused test evidence. |
| `bd-01KRCSG4YBRA595YP0WEE98ZQX` | Resolved `acquisition_onsets`, `amplitudes`, `evaluate`, `gen_empirical_hrf`, `gen_hrf_library`, `global_onsets`, `hrf_blocked`, `hrf_bspline_generator`, `hrf_daguerre_generator`, `hrf_fir_generator`, `hrf_fourier_generator`, `hrf_lagged`, `hrf_set`, `hrf_tent_generator`, `samples`, and `shift` for `fmrihrf`; the corresponding inventory rows now have stable Python surfaces and focused HRF test evidence. |
| `bd-01KRCSGJF4SE4NYVP3R7AGY279` | Added `docs/source/design/fmrireg_migration.rst` and linked it from the design docs index and README. The guide covers model construction, config/OLS/AR/robust options, contrasts, result writing, LSA/LSS, simulation, group/meta workflows, and intentionally different or unsupported behavior. |
| `bd-01KRCSG4ZKV7FHX5TV11RTYYY4` | Resolved fmrireg result, dataset, group-data, and fitted-HRF accessor rows: `ar_parameters`, `coef_image`, `coef_names`, `fitted_hrf`, `get_contrasts`, `get_covariates`, `get_data`, `get_data_matrix`, `get_mask`, `get_rois`, `get_subjects`, `n_subjects`, `p_values`, `pvalues`, `se`, `standard_error`, `stats`, `tidy`, `tidy_fitted_hrf`, and `zscores` are ported; `get_formula` is intentionally pythonized as a best-effort formula/term summary. The fmrireg `global_onsets` row was also reconciled to the existing shared top-level helper. |
| `bd-01KRCSG4Z360YJRCSZWBXC65BV` | Resolved fmrireg fitting and contrast helper rows: `apply_soft_projection`, `compute_lm_contrasts`, `compute_lm_contrasts_from_suffstats`, `estimate`, `fit_contrasts`, `fit_glm_from_suffstats`, `fit_glm_on_transformed_series`, `fit_glm_with_config`, `fmri_ols_fit`, `lowrank_control`, `paired_diff_block`, `soft_projection`, `t_to_beta_se`, `flip_sign`, and `hrf_smoothing_kernel` are ported; `fmri_rlm` is pythonized around the explicit Python model/config contract. |
| `bd-01KRCSGJEF8WYGSZSF8XC6T2JT` | Resolved fmrireg dataset, IO, and benchmark helper rows: `create_design_matrix_from_benchmark`, `data_chunks`, `evaluate`, `evaluate_method_performance`, `extract_csv_data`, `fmri_mem_dataset`, `get_benchmark_summary`, `list_benchmark_datasets`, `load_benchmark_dataset`, `read_fmri_config`, `read_h5_full`, `read_nifti_full`, `register_basis`, `resolve_basis`, and `samples` are ported; `design_plot`, `fmri_latent_lm`, and `latent_dataset` are pythonized around table/object-based Python contracts. |
| `bd-01KRCSGJFMDS68ZPTZG1N89T62` | Resolved fmrireg meta and second-level variant rows: `fmri_meta_fit`, `fmri_meta_fit_contrasts`, `fmri_meta_fit_cov`, `fmri_meta_fit_extended`, and `meta_effective_n` are ported as low-level matrix helpers backed by the scoped Python second-level implementation. |

## Resolution Rule

Each pending API row must end in one of these states:

- `ported`: stable Python API plus exact test and docs evidence.
- `pythonized`: explicit Python API decision plus exact test and docs evidence.
- `scoped_out`: rationale documented in the inventory and, when relevant, the
  migration guide.

Rows should not remain `pending` after the corresponding mote issue is closed.
