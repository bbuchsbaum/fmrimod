# Contract Document Audit

This audit keeps `docs/contracts/*_v1.md` from becoming unactionable planning
prose. Every top-level v1 contract must have an owner, an executable or grep
anchor, a consumer boundary, and a disposition.

Dispositions:

- `keep`: live contract with an active consumer and red check.
- `bead`: live concern that needs follow-up work before the contract can be
  treated as stable.
- `archive`: historical contract that should move under `docs/contracts/archive/`.
- `supersede`: contract whose content should be folded into a newer source of
  truth.

| Document | Owner | Red check / anchor | Consumer | Disposition |
| --- | --- | --- | --- | --- |
| `adversarial_benchmark_report_schema_v1.md` | `bd-01KRHRVCFJA19RZPPA6K2D46X9` | `tests/test_benchmarks/test_adversarial_report_schema.py` | `benchmarks/parity/tier_e_adversarial_gauntlet/` report schema | keep |
| `compat_retirement_inventory_v1.md` | `bd-01KRHTCQSMV6GKVGV9EXFAHJ9B` | `docs/contracts/compat_retirement_inventory_v1.md`; `tests/test_glm/test_fmrireg_compat.py` | `fmrimod/*/compat.py` migration surfaces | keep |
| `fitlins_nilearn_overlap_v1.md` | `bd-01KRFKYRP9ERR436SA4MYM4HAH` | `cross_testing/test_spm_auditory_parity.py`; `cross_testing/test_fitlins_cli_derivative_parity.py`; `tests/test_benchmarks/test_tier_c_second_level_workflow.py` | Nilearn/FitLins parity workflows and proof artifacts | keep |
| `fmriar_parity_audit_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `tests/test_ar/test_fmriar_export_compat.py`; `tests/test_ar/test_rpy2_parity.py` | `fmrimod.ar` parity and compatibility surface | keep |
| `fmridataset_consolidation_plan_v1.md` | `bd-01KRFD264CZ250Q5388MJ6FWTQ` | `tests/test_dataset/test_canonical_contracts.py`; `tests/test_dataset/test_fmridataset_facade.py` | `fmrimod.dataset` source-of-truth consolidation | keep |
| `fmridesign_design_formula_typed_seam_v1.md` | `bd-01KRGFGW2JA7RQNV0RR950Z6VQ` | `tests/design/test_fmridesign_typed_seam.py`; `docs/contracts/trio_api_inventory_v1.md` | typed formula/design lowering seam | keep |
| `fmridesign_test_surface_v1.md` | `bd-01KRGG0B84BPR1EJ2V74S2NST7` | `tests/design/`; `tests/hrf/` | fmridesign test-port coverage ledger | keep |
| `fmrihrf_test_port_audit_v1.md` | `bd-01KRGCY5YHSV59MR1JMYYE12N4` | `tests/hrf/`; `tests/test_hrf/test_hrf_norm.py` | `fmrimod.hrf` typed HRF surface | keep |
| `fmrilss_test_surface_v1.md` | `bd-01KRGF28DC3K5PNG9XCGZS697D` | `tests/test_single/`; `tests/test_model/test_fmri_model.py` | single-trial/LSS beta-estimation surface | keep |
| `fmrireg_test_surface_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `tests/test_glm/`; `tests/test_ar/`; `tests/test_stats/` | fmrireg-lineage GLM, AR, stats, and model coverage ledger | keep |
| `group_hdf5_alignment_manifest_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `tests/test_group/test_io.py`; `tests/test_dataset/test_group_data.py` | group dataset HDF5/alignment schema boundary | bead |
| `group_lmm_native_v1.md` | `bd-01KRFMD36DJRTDHG1BW0YERBF4` | `tests/test_group/test_reducers.py`; `tests/test_group/test_fmrigds_reducer_parity.py` | native group LMM reducer boundary | bead |
| `group_reducer_concurrency_v1.md` | `bd-01KRHTJ9WFSSBZSDGAN4V7PHGS` | `tests/test_group/test_reducers.py`; `fmrimod/group/reducers.py` | reducer policy/kernel/registry split | bead |
| `not_implemented_audit_v1.md` | `bd-01KRHW5K5FQPC1NW09JQP6XH5N` | `docs/contracts/not_implemented_audit_v1.md`; `fmrimod/utils/generics.py` | `NotImplementedError` tier ledger across `fmrimod/` | keep |
| `parametric_contrast_sugar_v1.md` | `bd-01KRM5FRQP16S05T3DA57DQY31` | `tests/test_contrast/test_parametric_sugar_v1_contract.py`; `tests/test_benchmarks/test_tier_e_bids_parametric_group_followthrough.py` | `fmrimod.contrast.modulator` and `fmrimod.contrast.group_dataset_from_contrasts` v1 surface | keep |
| `public_api_policy_v1.md` | `bd-01KRHTHY5309X3VZT515S9E3H3` | `tests/test_public_api/test_api_inventory.py`; `docs/contracts/api_inventory_v1.json` | top-level API promotion and inventory policy | keep |
| `second_level_parity_v1.md` | `bd-01KRGWWFNX93Q2RX8PZH6RNBP4` | `tests/test_benchmarks/test_tier_c_second_level_workflow.py`; `benchmarks/parity/tier_c_second_level/workflow.py` | second-level Nilearn parity semantics | keep |
| `scoped_vs_strict_gap_v1.md` | `bd-01KRNN0H73CCYGFJSJ30JPVFTW` | `docs/contracts/scoped_vs_strict_gap_v1.md`; `tests/test_glm/test_fmrireg_compat.py` | scoped-mypy epic gate vs full-strict `[tool.mypy]` reconciliation | keep |
| `trio_api_inventory_v1.md` | `bd-01KRGFGW2JA7RQNV0RR950Z6VQ` | `tests/test_public_api/test_api_inventory.py`; `docs/contracts/api_inventory_v1.json` | fmrireg/fmridesign/fmrihrf top-level inventory | keep |
| `type_gate_caveats_v1.md` | `bd-01KRNN0H73CCYGFJSJ30JPVFTW` | `docs/contracts/decisions/0002-full-strict-is-the-authoritative-type-gate.md`; `docs/contracts/type_gate_caveats_v1.md` | full-strict type-gate residue and its exit beads (`bd-01KRRM1DB1CP8E5CEA2Q1Z49Z5`, `bd-01KRRM1DZH80R9SPHDH6QG6ZGR`) | keep |

## Maintenance Rule

When a new top-level `docs/contracts/*_v1.md` file is added, update this audit
in the same change. When a contract has no live consumer or executable anchor,
mark it `archive`, `supersede`, or `bead` rather than leaving it as inert
planning text.
