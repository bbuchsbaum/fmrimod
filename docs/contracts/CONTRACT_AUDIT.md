# Contract Document Audit

Date: 2026-05-13

Owner: `bd-01KRHTJ3EDNSAYTSFHVZPDZ28A`

This file is the checked-in routing table for top-level
`docs/contracts/*_v1.md` documents. It answers a narrow governance question:
which contract docs are live rules, which are owned follow-up work, and which
should eventually be archived or superseded?

The table is intentionally about current routing, not historical credit. A
contract can have strong evidence and still be routed to a bead if its public
semantics are changing.

Allowed dispositions:

- `keep`: live contract with an existing red-check anchor or consumer.
- `bead`: live enough to keep, but current work is explicitly owned by the
  listed bead.
- `archive`: no longer a live rule after the named successor is in place.
- `supersede`: still useful, but should be replaced by a named broader
  contract.

| Contract | Current owner | Red check or anchor | Consumer or workflow | Disposition |
| --- | --- | --- | --- | --- |
| `docs/contracts/adversarial_benchmark_report_schema_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `uv run pytest tests/test_benchmarks/test_adversarial_report_schema.py tests/test_benchmarks/test_tier_e_adversarial_gauntlet.py -q` | `benchmarks.parity.adversarial_schema` and `benchmarks/parity/tier_e_adversarial_gauntlet/workflow.py` | `keep` |
| `docs/contracts/compat_retirement_inventory_v1.md` | `bd-01KRHTHY5309X3VZT515S9E3H3` | `rg -n "Typed replacement target" docs/contracts/compat_retirement_inventory_v1.md` | namespace-promotion and compat-retirement planning for `fmrimod/*/compat.py` | `bead` |
| `docs/contracts/fitlins_nilearn_overlap_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `uv run pytest tests/test_benchmarks/test_parity_proof_artifacts.py cross_testing/test_caveats_index.py -q` | `benchmarks/parity/proof_artifacts.json`, parity workflows, and caveat governance | `keep` |
| `docs/contracts/fmriar_parity_audit_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `uv run pytest tests/test_ar/test_fmriar_export_compat.py tests/test_ar/test_rpy2_parity.py -q` | `fmrimod.ar` public export and R-parity surface | `keep` |
| `docs/contracts/fmridataset_consolidation_plan_v1.md` | `bd-01KRFD264CZ250Q5388MJ6FWTQ` | `uv run pytest tests/test_dataset/test_canonical_contracts.py tests/test_dataset/test_fmridataset_facade.py -q` | dataset consolidation work and `fmridataset` facade routing | `keep` |
| `docs/contracts/fmridesign_design_formula_typed_seam_v1.md` | `bd-01KRGFGW2JA7RQNV0RR950Z6VQ` | `uv run pytest tests/design/test_fmridesign_typed_seam.py tests/test_glm/test_fmri_lm_spec_dataset.py -q` | typed design/formula seam and `fmri_lm(spec, dataset)` path | `keep` |
| `docs/contracts/fmridesign_test_surface_v1.md` | `bd-01KRGG0B84BPR1EJ2V74S2NST7` | `uv run pytest tests/design/test_rpy2_parity.py tests/design/test_event_model.py tests/design/test_convolve_formula_hrf.py -q` | R fmridesign test-surface map for design, event, HRF, and contrast APIs | `keep` |
| `docs/contracts/fmrihrf_test_port_audit_v1.md` | `bd-01KRGCY5YHSV59MR1JMYYE12N4` | `uv run pytest tests/hrf/test_fmrihrf_export_compat.py tests/hrf/test_fmrihrf_ported_edge_cases.py tests/hrf/test_r_equivalence.py -q` | HRF port evidence and typed-HRF hardening work | `keep` |
| `docs/contracts/fmrilss_test_surface_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `uv run pytest tests/test_single/test_lss.py tests/test_single/test_lsa.py tests/test_single/test_sbhm.py -q` | single-trial LSS/LSA/SBHM parity and migration evidence | `keep` |
| `docs/contracts/fmrireg_test_surface_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `uv run pytest tests/test_glm/test_fmrireg_compat.py tests/test_stats/test_meta_compat.py tests/test_dataset/test_fmrireg_compat.py -q` | fmrireg compatibility map across GLM, dataset, and group statistics | `keep` |
| `docs/contracts/group_hdf5_alignment_manifest_v1.md` | `bd-01KRFMW3DVT9HHY5NGCG0PEWZV` | `uv run pytest tests/test_group/test_io.py::test_hdf5_roundtrips_language_neutral_alignment_manifest -q` | group HDF5 read/write schema in `fmrimod.group.io` | `keep` |
| `docs/contracts/group_lmm_native_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `uv run pytest tests/test_group/test_reducers.py::test_lmm_ri_voxelwise_theta_runs_natively tests/test_group/test_reducers.py::test_lmm_ri_slope1_voxelwise_full_covariance_runs_natively -q` | native repeated-measures LMM reducers in `fmrimod.group.reducers` | `keep` |
| `docs/contracts/group_reducer_concurrency_v1.md` | `bd-01KRHTJ9WFSSBZSDGAN4V7PHGS` | `uv run pytest tests/test_group/test_reducers.py::test_perm_onesample_parallel_matches_serial tests/test_group/test_reducers.py::test_ols_voxelwise_parallel_matches_serial -q` | reducer threading policy and future policy/kernel/registry split | `bead` |
| `docs/contracts/not_implemented_audit_v1.md` | `bd-01KRHW5K5FQPC1NW09JQP6XH5N` | `rg -nE "raise NotImplementedError" fmrimod/` | `NotImplementedError` tier ledger across `fmrimod/` | `keep` |
| `docs/contracts/parametric_contrast_sugar_v1.md` | `bd-01KRM5FRQP16S05T3DA57DQY31` | `uv run pytest tests/test_contrast/test_parametric_sugar_v1_contract.py tests/test_benchmarks/test_tier_e_bids_parametric_group_followthrough.py -q` | parametric semantic-contrast sugar (`modulator(...).within(...).slope(...)`) and `group_dataset_from_contrasts` collector in `fmrimod.contrast` and `fmrimod.dataset` | `bead` |
| `docs/contracts/public_api_policy_v1.md` | `bd-01KRHTHY5309X3VZT515S9E3H3` | `uv run pytest tests/test_public_api/test_api_inventory.py -q` | public API inventory and top-level namespace governance | `keep` |
| `docs/contracts/scoped_vs_strict_gap_v1.md` | `bd-01KRNN0H73CCYGFJSJ30JPVFTW` | `uv run pytest tests/test_glm/test_fmrireg_compat.py -q` | scoped-mypy epic gate vs full-strict `[tool.mypy]` reconciliation; verified findings (incl. fixed `fmri_rlm` bug) | `keep` |
| `docs/contracts/second_level_parity_v1.md` | `bd-01KRFMD3JFKMH2ETGPRRAWAFME` | `uv run pytest tests/test_stats/test_group_fit_interface.py tests/test_benchmarks/test_tier_c_second_level_workflow.py -q` | group-fit parity and pending DataFrame-first second-level API | `bead` |
| `docs/contracts/trio_api_inventory_v1.md` | `bd-01KRHTHY5309X3VZT515S9E3H3` | `uv run pytest tests/test_public_api/test_api_inventory.py -q` | current R trio surface map; successor should be a unified seven-package inventory | `supersede` |
| `docs/contracts/type_gate_caveats_v1.md` | `bd-01KRNN0H73CCYGFJSJ30JPVFTW` | `uv run pytest tests/test_public_api/test_contract_doc_audit.py -q` | full-strict type-gate residue index; exits via `bd-01KRRM1DB1CP8E5CEA2Q1Z49Z5` and `bd-01KRRM1DZH80R9SPHDH6QG6ZGR` | `bead` |

## Follow-up Rules

- New top-level `docs/contracts/*_v1.md` files must add one row here in the
  same change.
- A row with `Disposition` = `bead` must name the owning bead that will either
  make the contract live or move it to a successor/archive location.
- A row with `Disposition` = `supersede` or `archive` must name the successor
  or archive route in the consumer cell.
- This audit is about top-level contract documents only. Files under
  `docs/contracts/archive/` remain historical evidence and are not audited here.
