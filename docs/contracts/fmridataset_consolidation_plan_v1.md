# fmridataset Consolidation Plan v1

Status: Active migration plan

Date: 2026-05-12

## Purpose

This plan defines how `../fmridataset-py` should be folded into `fmrimod`
without leaving two competing dataset implementations. The target is one
Python distribution with modular internals:

- `fmrimod.dataset` is the canonical home for dataset abstractions, storage
  backends, BIDS-HDF5 readers, selectors, series objects, study containers,
  and conversion helpers.
- `fmrimod` remains the installable toolkit that joins dataset IO, design,
  HRFs, GLM fitting, single-trial estimation, group statistics, simulation,
  and result IO.
- A future `fmridataset` package, if kept, is a compatibility facade only. It
  must re-export from `fmrimod.dataset` and must not contain independent
  implementation logic.

This plan is subordinate to the repo mission and vision: `fmridataset` is not
being folded in to achieve mechanical export parity. It is being folded in to
make the load-bearing `fmri_dataset -> fmri_lm -> contrast -> group_fit` seam
typed, inspectable, parity-validated, and strong enough for the flagship
workflows in `./benchmarks/` and `cross_testing/`.

## Source Of Truth Rule

There are two source-of-truth levels, and both must stay singular:

1. Implementation source of truth: `fmrimod.dataset`.
2. Migration planning source of truth: this file.

Beads, issue descriptions, release notes, and other docs may point to this
file, but they should not restate a parallel task list or API matrix. If the
plan changes, update this file first.

The compatibility package rule is strict: `fmridataset.FmriDataset is
fmrimod.dataset.FmriDataset` should be true for public class re-exports after
the facade exists. The compatibility package may own import/deprecation tests,
but not a second backend registry, `SamplingFrame`, `FmriDataset`, or
`data_chunks` implementation.

## Formal PRD

### Problem

The existing plan establishes the right architectural direction, but broad code
motion is still unsafe until two things are made unambiguous:

1. every public `fmridataset.__all__` name has an explicit disposition; and
2. the semantic conflicts between the current repos are converted into
   executable contracts.

The current donor facade exports 146 public names. The current
`fmrimod.dataset` facade exports 30 public names. The migration therefore
needs a real inventory and contract-test gate before implementation starts.

### Product Goal

Make `fmrimod.dataset` the single implementation source of truth for dataset
IO and containers, while preserving a deliberate compatibility path for users
of `fmridataset`. The winning Python shape is allowed to differ from the R
surface when that improves typing, composition, neuroim integration, or the
end-to-end modeling workflow.

### Users

- `fmrimod` users who need dataset IO to feed design, GLM, single-trial, group,
  and simulation workflows.
- Existing `fmridataset` users who need a predictable migration path.
- Maintainers and agents porting code from `../fmridataset-py` into `fmrimod`.

### Success Criteria

- A public-surface inventory in this file classifies all 146 donor
  `fmridataset.__all__` names.
- No donor name is moved or aliased without one of these dispositions:
  `adopt`, `merge`, `rename`, `compat_alias`, `internal_only`, or
  `scoped_out`.
- A canonical definitions table in this file states exactly one owner and one
  meaning for each disputed concept.
- A contract test file exists before broad implementation work starts:
  `tests/test_dataset/test_canonical_contracts.py`.
- The contract tests fail against duplicate or ambiguous definitions and pass
  only when the canonical behavior is implemented.
- The `fmridataset` compatibility package, if retained, contains no independent
  implementation logic.
- Every ported dataset object supports the four-stage seam:
  `fmri_dataset -> fmri_lm -> contrast -> group_fit`.
- Workflow-bearing functionality is prioritized over donor export completeness.
  Donor names that do not support a flagship workflow may remain unported when
  they are classified as `internal_only` or `scoped_out` with a reason.
- Real image, mask, and contrast pathways should route through
  `neuroim-python` typed containers. Direct `nibabel` use belongs at IO
  boundaries only.
- Any semantic divergence from R `fmridataset`, Nilearn, FitLins, or SPM that
  affects a flagship workflow must be recorded in `docs/contracts/CAVEATS.md`
  with an owner and exit criterion.

### Non-Goals

- This PRD does not require porting every backend in the first implementation
  slice.
- This PRD does not require porting every donor export. The inventory prevents
  ambiguity; it does not force low-leverage API accretion.
- This PRD does not require keeping R-compatible semantics when they conflict
  with established `fmrimod` timing semantics.
- This PRD does not permit a second source of truth in beads, release notes, or
  a parallel API spreadsheet.

### Mission-Aligned Prioritization

When implementation order is ambiguous, choose the path that improves flagship
workflow quality first:

1. Does it strengthen `dataset -> lm -> contrast -> group_fit`?
2. Does it replace ad hoc NumPy/nibabel handling with typed neuroim-compatible
   containers at the right boundary?
3. Does it make an analysis more inspectable, serializable, or reproducible?
4. Does it retire a caveat or enable a benchmark/parity workflow?
5. Does it preserve a thin `fmridataset` facade without adding a second
   implementation?

If the answer is no to all five, the work is backlog even if the donor package
exports a function for it.

### Canonical Definitions

| Concept | Canonical owner | Canonical meaning | Compatibility rule |
| --- | --- | --- | --- |
| Timing class | `fmrimod.sampling.SamplingFrame` | one timing object for design and dataset workflows | `fmridataset.SamplingFrame` may re-export this class only |
| `SamplingFrame.samples` | `fmrimod.sampling.SamplingFrame.samples` | acquisition/sample times in seconds | integer sample indices need an explicit helper, not a second meaning |
| Run/block IDs | `fmrimod.sampling.SamplingFrame` plus explicit helper methods | internal representation is 0-based unless a helper requests 1-based ids | R-style 1-based ids must be opt-in and named |
| Dataset class | `fmrimod.dataset.FmriDataset` | the only public dataset container | `fmridataset.FmriDataset is fmrimod.dataset.FmriDataset` |
| Timepoint count | `fmrimod.dataset.FmriDataset.n_timepoints` | total number of timepoints | per-run lengths are `blocklens` or `run_lengths` |
| Matrix data access | `fmrimod.dataset.FmriDataset.get_data` | explicit matrix slicing by `rows` and `cols` | run access uses an explicit name such as `get_run_data` |
| Storage protocol | `fmrimod.dataset.StorageBackend` and `BackendDims` | the only backend contract and dimensions type | old imports re-export these names |
| Backend registry | `fmrimod.dataset.BackendRegistry` | the only backend registry | no facade-owned registry state |
| Latent dataset | `fmrimod.dataset.LatentDataset` | storage-backed latent dataset class | temporary score/loadings helper is merged or renamed |
| Chunks | `fmrimod.dataset.data_chunks` and related chunk types | typed chunks carrying data and indices | index-only chunk helpers need distinct names |
| Compatibility package | `fmridataset` | import facade | no independent backend, dataset, timing, or chunk code |

### Public Surface Inventory Requirement

The inventory must live in this file under `Public Surface Inventory`. Each row
must use this schema:

| Donor name | Current donor role | Existing `fmrimod` surface | Disposition | Canonical owner | Required tests | Notes |
| --- | --- | --- | --- | --- | --- | --- |

Disposition meanings:

- `adopt`: move donor implementation into `fmrimod.dataset` as the canonical
  implementation.
- `merge`: combine donor behavior with an existing `fmrimod` implementation
  and retire the duplicate.
- `rename`: expose behavior under a different canonical Python name.
- `compat_alias`: keep the donor spelling only as a facade alias to a canonical
  `fmrimod.dataset` name.
- `internal_only`: keep behavior available only behind an internal module or
  helper.
- `scoped_out`: intentionally do not port; document why.

Inventory acceptance rules:

- The inventory row count must equal the current donor `__all__` count, 146,
  unless this file documents why the donor count changed.
- `SamplingFrame`, `FmriDataset`, `LatentDataset`, `StorageBackend`,
  `BackendRegistry`, and `data_chunks` must be reviewed manually and cannot be
  bulk-classified.
- Every `adopt`, `merge`, `rename`, or `compat_alias` row must name at least
  one required test file.
- Every `scoped_out` row must include a rationale.
- The inventory is incomplete if any row has `pending`, `unknown`, or an empty
  disposition.

### Contract Test Requirement

Before broad implementation work starts, add
`tests/test_dataset/test_canonical_contracts.py` with tests for these
contracts:

- `SamplingFrame` imported through `fmrimod` and `fmrimod.dataset` is the same
  class, or `fmrimod.dataset` does not expose a timing class at all.
- `SamplingFrame.samples` returns acquisition times in seconds.
- integer sample indices are available only through an explicitly named helper.
- 1-based run IDs are available only through an explicit opt-in helper.
- `matrix_dataset(...)` returns the canonical `fmrimod.dataset.FmriDataset`.
- `n_timepoints` is total timepoints.
- per-run lengths are exposed as `blocklens` or `run_lengths`.
- `get_data(rows=..., cols=...)` is matrix slicing.
- run access uses an explicit run method, not positional `get_data(run)`.
- `fmridataset` facade identity tests pass once the facade exists.

These tests are the gate for code motion. Until they exist, implementation work
should be limited to inventory, PRD updates, and small preparatory refactors
that do not move public dataset classes.

### Acceptance Gate

We are fully satisfied with the plan only when this file contains:

- the canonical definitions table above;
- a complete public-surface inventory for all donor `__all__` names;
- the contract test requirement above;
- an implementation sequence that starts with those contract tests; and
- a completion definition that forbids duplicate dataset/timing/backend/chunk
  sources of truth.

## Public Surface Inventory

Status: Phase 1 inventory populated.

This section is the only accepted home for the donor public-surface inventory.
It must be completed before broad implementation work begins. The table must
contain one row for each donor `fmridataset.__all__` name. The donor count at
the time this PRD was written is 146 names.

Required schema:

| Donor name | Current donor role | Existing `fmrimod` surface | Disposition | Canonical owner | Required tests | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `__version__` | package version | fmrimod.__version__ | `compat_alias` | `fmrimod.__version__` | `tests/test_dataset/test_canonical_contracts.py` | Facade should report the fmrimod distribution version or a documented compatibility version. |
| `FmriDatasetError` | dataset error classes | none | `adopt` | `fmrimod.dataset.errors` | `tests/test_dataset/test_errors.py` | Canonical dataset errors should live with dataset IO. |
| `BackendIOError` | dataset error classes | none | `adopt` | `fmrimod.dataset.errors` | `tests/test_dataset/test_errors.py` | Canonical dataset errors should live with dataset IO. |
| `ConfigError` | dataset error classes | none | `adopt` | `fmrimod.dataset.errors` | `tests/test_dataset/test_errors.py` | Canonical dataset errors should live with dataset IO. |
| `SamplingFrame` | timing class | fmrimod.SamplingFrame / fmrimod.sampling.SamplingFrame | `merge` | `fmrimod.sampling.SamplingFrame` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: one timing class only; donor class is retired. |
| `StorageBackend` | storage backend protocol | none | `adopt` | `fmrimod.dataset.backend_protocol` | `tests/test_dataset/test_backend_protocol.py` | Manual review required: one backend protocol only. |
| `BackendDims` | storage backend protocol | none | `adopt` | `fmrimod.dataset.backend_protocol` | `tests/test_dataset/test_backend_protocol.py` | Manual review required: one backend protocol only. |
| `BackendRegistry` | backend registry | none | `adopt` | `fmrimod.dataset.backend_registry` | `tests/test_dataset/test_backend_registry.py` | Manual review required: no facade-owned registry state. |
| `get_backend_registry` | backend registry | none | `adopt` | `fmrimod.dataset.backend_registry` | `tests/test_dataset/test_backend_registry.py` | Manual review required: no facade-owned registry state. |
| `register_backend` | backend registry | none | `adopt` | `fmrimod.dataset.backend_registry` | `tests/test_dataset/test_backend_registry.py` | Manual review required: no facade-owned registry state. |
| `create_backend` | backend registry | none | `adopt` | `fmrimod.dataset.backend_registry` | `tests/test_dataset/test_backend_registry.py` | Manual review required: no facade-owned registry state. |
| `is_backend_registered` | backend registry | none | `adopt` | `fmrimod.dataset.backend_registry` | `tests/test_dataset/test_backend_registry.py` | Manual review required: no facade-owned registry state. |
| `list_backend_names` | backend registry | none | `adopt` | `fmrimod.dataset.backend_registry` | `tests/test_dataset/test_backend_registry.py` | Manual review required: no facade-owned registry state. |
| `unregister_backend` | backend registry | none | `adopt` | `fmrimod.dataset.backend_registry` | `tests/test_dataset/test_backend_registry.py` | Manual review required: no facade-owned registry state. |
| `MatrixBackend` | matrix backend | none | `adopt` | `fmrimod.dataset.backends.matrix` | `tests/test_dataset/test_matrix_backend.py` | Base-install backend; first backend port target. |
| `BidsH5ScanBackend` | BIDS-HDF5 scan backend | none | `adopt` | `fmrimod.dataset.backends.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; port after core registry. |
| `SharedH5Connection` | BIDS-HDF5 scan backend | none | `adopt` | `fmrimod.dataset.backends.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; port after core registry. |
| `matrix_backend` | matrix backend constructor | none | `adopt` | `fmrimod.dataset.backend_constructors` | `tests/test_dataset/test_matrix_backend.py` | Constructor for canonical MatrixBackend. |
| `nifti_backend` | NIfTI backend constructor | fmrimod.dataset.adapters.NibabelAdapter (different) | `merge` | `fmrimod.dataset.backend_constructors` | `tests/test_dataset/test_nifti_backend.py` | Requires nifti extra; reconcile with existing NibabelAdapter. |
| `h5_backend` | HDF5 backend constructor | none | `adopt` | `fmrimod.dataset.backend_constructors` | `tests/test_dataset/test_h5_backend.py` | Requires hdf5 extra. |
| `zarr_backend` | Zarr backend constructor | none | `adopt` | `fmrimod.dataset.backend_constructors` | `tests/test_dataset/test_zarr_backend.py` | Requires zarr extra. |
| `latent_backend` | latent backend constructor | none | `adopt` | `fmrimod.dataset.backend_constructors` | `tests/test_dataset/test_latent_backend.py` | Requires hdf5 extra for file-backed latent data. |
| `study_backend` | study backend constructor | none | `adopt` | `fmrimod.dataset.backend_constructors` | `tests/test_dataset/test_study_backend.py` | Composite backend over canonical datasets. |
| `backend_open` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `backend_close` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `backend_get_dims` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `backend_get_mask` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `backend_get_data` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `backend_get_metadata` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `backend_get_loadings` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `backend_reconstruct_voxels` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `validate_backend` | backend functional methods | none | `compat_alias` | `fmrimod.dataset.backend_methods` | `tests/test_dataset/test_backend_methods.py` | Function facade over backend methods; no separate behavior. |
| `FmriDataset` | dataset container | fmrimod.dataset.FmriDataset (different contract) | `merge` | `fmrimod.dataset.FmriDataset` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: one public dataset class only. |
| `MatrixDataset` | in-memory dataset subclass | none | `adopt` | `fmrimod.dataset.MatrixDataset` | `tests/test_dataset/test_dataset_constructors.py` | Should subclass or specialize canonical FmriDataset. |
| `LatentDataset` | latent dataset class | fmrimod.dataset.LatentDataset (temporary helper) | `merge` | `fmrimod.dataset.LatentDataset` | `tests/test_dataset/test_latent_dataset.py` | Manual review required: keep storage-backed class and retire temporary helper. |
| `StudyDataset` | multi-subject dataset | none | `adopt` | `fmrimod.dataset.StudyDataset` | `tests/test_dataset/test_study_dataset.py` | Canonical multi-subject container. |
| `BidsH5StudyDataset` | BIDS-HDF5 study dataset | none | `adopt` | `fmrimod.dataset.BidsH5StudyDataset` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra. |
| `matrix_dataset` | matrix dataset constructor | fmrimod.matrix_dataset / fmrimod.dataset.fmri_mem_dataset | `merge` | `fmrimod.dataset.matrix_dataset` | `tests/test_dataset/test_canonical_contracts.py` | Top-level fmrimod may expose convenience wrapper only. |
| `fmri_dataset` | generic dataset constructor | fmrimod.fmri_dataset / fmrimod.dataset.FmriDataset | `merge` | `fmrimod.dataset.fmri_dataset` | `tests/test_dataset/test_dataset_constructors.py` | Reconcile adapter-wrapper contract with backend-backed contract. |
| `fmri_mem_dataset` | legacy matrix constructor | fmrimod.dataset.fmri_mem_dataset | `compat_alias` | `fmrimod.dataset.matrix_dataset` | `tests/test_dataset/test_dataset_constructors.py` | Compatibility alias only. |
| `fmri_h5_dataset` | legacy HDF5 constructor | none | `compat_alias` | `fmrimod.dataset.h5_dataset helper` | `tests/test_dataset/test_h5_backend.py` | Compatibility wrapper around HDF5 backend plus fmri_dataset. |
| `fmri_latent_dataset` | legacy latent constructor | none | `compat_alias` | `fmrimod.dataset.latent_dataset` | `tests/test_dataset/test_latent_dataset.py` | Compatibility alias only. |
| `fmri_dataset_legacy` | path-oriented legacy constructor | none | `compat_alias` | `fmrimod.dataset.fmri_dataset` | `tests/test_dataset/test_legacy_constructors.py` | Compatibility wrapper; not a second constructor contract. |
| `fmri_study_dataset` | legacy study constructor | none | `compat_alias` | `fmrimod.dataset.study_dataset` | `tests/test_dataset/test_study_dataset.py` | Compatibility alias only. |
| `fmri_zarr_dataset` | legacy zarr constructor | none | `compat_alias` | `fmrimod.dataset.zarr_dataset` | `tests/test_dataset/test_zarr_backend.py` | Compatibility alias only. |
| `latent_dataset` | latent dataset constructor | fmrimod.dataset.latent_dataset (temporary helper) | `merge` | `fmrimod.dataset.latent_dataset` | `tests/test_dataset/test_latent_dataset.py` | Must construct canonical storage-backed LatentDataset. |
| `study_dataset` | study dataset constructor | none | `adopt` | `fmrimod.dataset.study_dataset` | `tests/test_dataset/test_study_dataset.py` | Constructor for canonical StudyDataset. |
| `zarr_dataset` | Zarr dataset constructor | none | `adopt` | `fmrimod.dataset.zarr_dataset` | `tests/test_dataset/test_zarr_backend.py` | Requires zarr extra. |
| `bids_h5_dataset` | BIDS-HDF5 study helpers | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; BIDS directory ingestion may require bids extra. |
| `compress_bids_study` | BIDS-HDF5 study helpers | none | `adopt` | `fmrimod.dataset.bids_h5_write` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; BIDS directory ingestion may require bids extra. |
| `bids_h5_scan_backend` | BIDS-HDF5 scan backend | none | `adopt` | `fmrimod.dataset.backends.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; port after core registry. |
| `h5_shared_connection` | BIDS-HDF5 scan backend | none | `adopt` | `fmrimod.dataset.backends.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; port after core registry. |
| `subset_bids_h5` | BIDS-HDF5 study helpers | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; BIDS directory ingestion may require bids extra. |
| `study_to_group` | BIDS-HDF5 study helpers | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; BIDS directory ingestion may require bids extra. |
| `get_data` | dataset accessors | dataset methods exist with different signatures | `merge` | `fmrimod.dataset.data_access` | `tests/test_dataset/test_canonical_contracts.py` | Matrix access must use rows/cols; run access gets explicit method. |
| `get_data_matrix` | dataset accessors | dataset methods exist with different signatures | `merge` | `fmrimod.dataset.data_access` | `tests/test_dataset/test_canonical_contracts.py` | Matrix access must use rows/cols; run access gets explicit method. |
| `get_mask` | dataset accessors | dataset methods exist with different signatures | `merge` | `fmrimod.dataset.data_access` | `tests/test_dataset/test_canonical_contracts.py` | Matrix access must use rows/cols; run access gets explicit method. |
| `get_latent_scores` | latent accessors | fmrimod.dataset.LatentDataset helper methods (different) | `merge` | `fmrimod.dataset.latent` | `tests/test_dataset/test_latent_dataset.py` | Accessor facade over canonical LatentDataset. |
| `get_spatial_loadings` | latent accessors | fmrimod.dataset.LatentDataset helper methods (different) | `merge` | `fmrimod.dataset.latent` | `tests/test_dataset/test_latent_dataset.py` | Accessor facade over canonical LatentDataset. |
| `get_component_info` | latent accessors | fmrimod.dataset.LatentDataset helper methods (different) | `merge` | `fmrimod.dataset.latent` | `tests/test_dataset/test_latent_dataset.py` | Accessor facade over canonical LatentDataset. |
| `get_TR` | TR accessor | fmrimod.SamplingFrame.TR | `compat_alias` | `fmrimod.sampling.SamplingFrame` | `tests/test_dataset/test_canonical_contracts.py` | Function facade over canonical timing class. |
| `get_run_lengths` | run length accessors | fmrimod.SamplingFrame.blocklens | `compat_alias` | `fmrimod.sampling.SamplingFrame` | `tests/test_dataset/test_canonical_contracts.py` | Per-run lengths use blocklens/run_lengths, not n_timepoints. |
| `get_total_duration` | duration accessors | fmrimod.SamplingFrame sample timing helpers | `merge` | `fmrimod.sampling.SamplingFrame` | `tests/test_dataset/test_canonical_contracts.py` | Derive from canonical timing class. |
| `get_run_duration` | duration accessors | fmrimod.SamplingFrame sample timing helpers | `merge` | `fmrimod.sampling.SamplingFrame` | `tests/test_dataset/test_canonical_contracts.py` | Derive from canonical timing class. |
| `n_runs` | run count accessor | fmrimod.SamplingFrame.n_blocks / dataset.n_runs | `merge` | `fmrimod.sampling.SamplingFrame` | `tests/test_dataset/test_canonical_contracts.py` | Use one canonical run/block count vocabulary. |
| `n_timepoints` | timepoint count accessor | fmrimod.dataset currently per-run list | `merge` | `fmrimod.dataset.FmriDataset` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: canonical meaning is total count. |
| `blocklens` | run length accessors | fmrimod.SamplingFrame.blocklens | `compat_alias` | `fmrimod.sampling.SamplingFrame` | `tests/test_dataset/test_canonical_contracts.py` | Per-run lengths use blocklens/run_lengths, not n_timepoints. |
| `blockids` | block id accessor | fmrimod.SamplingFrame.blockids (0-based) | `rename` | `fmrimod.sampling.SamplingFrame.run_ids` | `tests/test_dataset/test_canonical_contracts.py` | Donor 1-based ids must become explicit opt-in helper/facade alias. |
| `samples` | sample index accessor | fmrimod.SamplingFrame.samples (time grid) | `rename` | `fmrimod.dataset.sample_indices` | `tests/test_dataset/test_canonical_contracts.py` | Donor integer sample ids cannot share the SamplingFrame.samples meaning. |
| `all_timepoints` | all-row index helper | none | `adopt` | `fmrimod.dataset.data_access` | `tests/test_dataset/test_data_access.py` | Returns 0-based row indices. |
| `subject_ids` | subject id accessor | group_data helpers only | `merge` | `fmrimod.dataset.study` | `tests/test_dataset/test_study_dataset.py` | Accessor over canonical StudyDataset/FmriGroup. |
| `is_sampling_frame` | sampling frame predicate | none | `compat_alias` | `fmrimod.sampling.SamplingFrame` | `tests/test_dataset/test_canonical_contracts.py` | Facade predicate over single timing class. |
| `participants` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `tasks` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `sessions` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `scan_manifest` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `parcellation_info` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `get_confounds` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `get_loadings` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `reconstruct_voxels` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `encoding_info` | BIDS-HDF5 accessors | none | `adopt` | `fmrimod.dataset.bids_h5` | `tests/test_dataset/test_bids_h5_dataset.py` | Requires hdf5 extra; preserve tested BIDS-HDF5 contracts. |
| `with_rowData` | R-style row metadata helper | none | `internal_only` | `fmrimod.dataset.compat` | `tests/test_dataset/test_legacy_constructors.py` | Keep only if needed for compatibility; not canonical Python API. |
| `mask_to_logical` | mask conversion utilities | none | `adopt` | `fmrimod.dataset.mask_utils` | `tests/test_dataset/test_mask_utils.py` | Canonical mask utility module. |
| `mask_to_volume` | mask conversion utilities | none | `adopt` | `fmrimod.dataset.mask_utils` | `tests/test_dataset/test_mask_utils.py` | Canonical mask utility module. |
| `FmriSeries` | series objects and constructors | none | `adopt` | `fmrimod.dataset.series` | `tests/test_dataset/test_fmri_series.py` | Use canonical selectors and dataset access. |
| `fmri_series` | series objects and constructors | none | `adopt` | `fmrimod.dataset.series` | `tests/test_dataset/test_fmri_series.py` | Use canonical selectors and dataset access. |
| `new_fmri_series` | series objects and constructors | none | `adopt` | `fmrimod.dataset.series` | `tests/test_dataset/test_fmri_series.py` | Use canonical selectors and dataset access. |
| `is_fmri_series` | series objects and constructors | none | `adopt` | `fmrimod.dataset.series` | `tests/test_dataset/test_fmri_series.py` | Use canonical selectors and dataset access. |
| `as_matrix` | series conversion | none | `compat_alias` | `fmrimod.dataset.to_matrix_dataset` | `tests/test_dataset/test_fmri_series.py` | Facade over canonical matrix conversion where applicable. |
| `as_tibble` | R-style table conversion | none | `rename` | `fmrimod.dataset.to_dataframe` | `tests/test_dataset/test_fmri_series.py` | Python canonical name should be dataframe-oriented; facade may keep donor spelling. |
| `resolve_selector` | series objects and constructors | none | `adopt` | `fmrimod.dataset.series` | `tests/test_dataset/test_fmri_series.py` | Use canonical selectors and dataset access. |
| `resolve_timepoints` | series objects and constructors | none | `adopt` | `fmrimod.dataset.series` | `tests/test_dataset/test_fmri_series.py` | Use canonical selectors and dataset access. |
| `series` | series objects and constructors | none | `adopt` | `fmrimod.dataset.series` | `tests/test_dataset/test_fmri_series.py` | Use canonical selectors and dataset access. |
| `as_dask_array` | lazy/dask conversion helpers | none | `adopt` | `fmrimod.dataset.lazy_array` | `tests/test_dataset/test_lazy_array.py` | Requires dask extra where applicable. |
| `as_dask_array_dataset` | lazy/dask conversion helpers | none | `adopt` | `fmrimod.dataset.lazy_array` | `tests/test_dataset/test_lazy_array.py` | Requires dask extra where applicable. |
| `as_delayed_array` | lazy/dask conversion helpers | none | `adopt` | `fmrimod.dataset.lazy_array` | `tests/test_dataset/test_lazy_array.py` | Requires dask extra where applicable. |
| `as_delarr` | lazy/dask conversion helpers | none | `adopt` | `fmrimod.dataset.lazy_array` | `tests/test_dataset/test_lazy_array.py` | Requires dask extra where applicable. |
| `FmriGroup` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `fmri_group` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `as_fmri_group` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `validate_fmri_group` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `subjects` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `n_subjects` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `iter_subjects` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `stream_subjects` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `group_map` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `group_reduce` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `filter_subjects` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `mutate_subjects` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `left_join_subjects` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `sample_subjects` | group dataset surface | fmrimod.dataset.GroupData family (different) | `merge` | `fmrimod.dataset.group` | `tests/test_dataset/test_fmri_group.py` | Reconcile with existing GroupData ingestion helpers; no parallel group abstractions. |
| `SeriesSelector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `IndexSelector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `AllSelector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `ROISelector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `VoxelSelector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `SphereSelector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `MaskSelector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `index_selector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `all_selector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `roi_selector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `voxel_selector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `sphere_selector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `mask_selector` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `resolve_indices` | selector classes and constructors | none | `adopt` | `fmrimod.dataset.selectors` | `tests/test_dataset/test_selectors.py` | Canonical selector machinery for FmriSeries and chunks. |
| `DataChunk` | chunking types and helpers | fmrimod.dataset.data_chunks (index-only helper, different) | `merge` | `fmrimod.dataset.data_chunks` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: one typed chunk contract; index-only helper must be distinctly named. |
| `ChunkIterator` | chunking types and helpers | fmrimod.dataset.data_chunks (index-only helper, different) | `merge` | `fmrimod.dataset.data_chunks` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: one typed chunk contract; index-only helper must be distinctly named. |
| `data_chunk` | chunking types and helpers | fmrimod.dataset.data_chunks (index-only helper, different) | `merge` | `fmrimod.dataset.data_chunks` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: one typed chunk contract; index-only helper must be distinctly named. |
| `data_chunks` | chunking types and helpers | fmrimod.dataset.data_chunks (index-only helper, different) | `merge` | `fmrimod.dataset.data_chunks` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: one typed chunk contract; index-only helper must be distinctly named. |
| `collect_chunks` | chunking types and helpers | fmrimod.dataset.data_chunks (index-only helper, different) | `merge` | `fmrimod.dataset.data_chunks` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: one typed chunk contract; index-only helper must be distinctly named. |
| `exec_strategy` | chunking types and helpers | fmrimod.dataset.data_chunks (index-only helper, different) | `merge` | `fmrimod.dataset.data_chunks` | `tests/test_dataset/test_canonical_contracts.py` | Manual review required: one typed chunk contract; index-only helper must be distinctly named. |
| `to_matrix_dataset` | dataset conversion helpers | none | `adopt` | `fmrimod.dataset.conversions` | `tests/test_dataset/test_conversions.py` | Materialize canonical datasets as matrix datasets. |
| `as_matrix_dataset` | dataset conversion helpers | none | `adopt` | `fmrimod.dataset.conversions` | `tests/test_dataset/test_conversions.py` | Materialize canonical datasets as matrix datasets. |
| `read_fmri_config` | dataset config reader | fmrimod.dataset.read_fmri_config | `merge` | `fmrimod.dataset.config` | `tests/test_dataset/test_config.py` | Merge with current fmrimod config helper. |
| `write_fmri_config` | dataset config writer | none | `adopt` | `fmrimod.dataset.config` | `tests/test_dataset/test_config.py` | YAML/JSON config writer. |
| `lru_cache` | dataset cache helpers | none | `adopt` | `fmrimod.dataset.cache` | `tests/test_dataset/test_cache.py` | Requires cache extra if cachetools is used. |
| `fmri_clear_cache` | dataset cache helpers | none | `adopt` | `fmrimod.dataset.cache` | `tests/test_dataset/test_cache.py` | Requires cache extra if cachetools is used. |
| `fmri_cache_info` | dataset cache helpers | none | `adopt` | `fmrimod.dataset.cache` | `tests/test_dataset/test_cache.py` | Requires cache extra if cachetools is used. |
| `fmri_cache_resize` | dataset cache helpers | none | `adopt` | `fmrimod.dataset.cache` | `tests/test_dataset/test_cache.py` | Requires cache extra if cachetools is used. |
| `generate_example_fmri_data` | example data helpers | fmrimod.simulate helpers (partial) | `merge` | `fmrimod.simulate / fmrimod.dataset.example_helpers` | `tests/test_dataset/test_example_helpers.py` | Keep only if they support docs/tests; avoid duplicating simulation API. |
| `generate_example_events` | example data helpers | fmrimod.simulate helpers (partial) | `merge` | `fmrimod.simulate / fmrimod.dataset.example_helpers` | `tests/test_dataset/test_example_helpers.py` | Keep only if they support docs/tests; avoid duplicating simulation API. |
| `generate_example_paths` | example data helpers | fmrimod.simulate helpers (partial) | `merge` | `fmrimod.simulate / fmrimod.dataset.example_helpers` | `tests/test_dataset/test_example_helpers.py` | Keep only if they support docs/tests; avoid duplicating simulation API. |
| `generate_example_mask` | example data helpers | fmrimod.simulate helpers (partial) | `merge` | `fmrimod.simulate / fmrimod.dataset.example_helpers` | `tests/test_dataset/test_example_helpers.py` | Keep only if they support docs/tests; avoid duplicating simulation API. |
| `generate_benchmark_data` | benchmark toy data helper | fmrimod.dataset benchmark compatibility helpers (partial) | `merge` | `fmrimod.dataset.example_helpers` | `tests/test_dataset/test_example_helpers.py` | Doc/test helper, not core modeling API. |
| `print_dataset_info` | example utility helpers | none | `internal_only` | `fmrimod.dataset.example_helpers` | `tests/test_dataset/test_example_helpers.py` | Docs helper only; do not promote as canonical public API unless needed. |
| `analyze_run` | example utility helpers | none | `internal_only` | `fmrimod.dataset.example_helpers` | `tests/test_dataset/test_example_helpers.py` | Docs helper only; do not promote as canonical public API unless needed. |

## Phase 1 Gate Status

Status: Satisfied for inventory and contract-test creation.

- The donor `fmridataset.__all__` count is 146.
- The inventory above contains 146 classified rows.
- No inventory row is `pending` or `unknown`.
- `tests/test_dataset/test_canonical_contracts.py` exists and names the
  canonical timing, dataset, backend, latent, chunk, and facade contracts.
- Focused verification currently reports `57 passed, 1 xfailed` for
  `tests/test_dataset`.

Remaining strict-xfail gates:

- The installed/importable `fmridataset` package still resolves to the donor
  implementation rather than a re-export facade.

## Non-Goals

- Do not use a namespace package split for this consolidation.
- Do not preserve two stateful dataset classes with the same name.
- Do not keep `fmridataset-py` as a feature-bearing runtime dependency of
  `fmrimod`.
- Do not promote every dataset helper to top-level `fmrimod`; most names should
  live under `fmrimod.dataset`.
- Do not duplicate detailed dataset tutorials in both projects. The canonical
  user docs should live in `fmrimod`.

## Current State

`fmrimod` already has a `fmrimod.dataset` package, but its current dataset layer
is mostly a GLM-facing adapter wrapper plus group-data compatibility helpers.
It includes:

- `fmrimod.dataset.FmriDataset`, a wrapper around a run-wise data source with
  `get_data(run)`;
- `DatasetProtocol`, `MaskProtocol`, and chunking protocols;
- `NumpyAdapter` and `NibabelAdapter`;
- group-data ingestion helpers;
- compatibility helpers such as `fmri_mem_dataset`, `latent_dataset`, and
  `data_chunks`.

`fmridataset-py` owns the richer dataset substrate:

- `StorageBackend`, `BackendDims`, and a backend registry;
- matrix, NIfTI, HDF5, Zarr, latent, study, and BIDS-HDF5 backends;
- `FmriDataset`, `MatrixDataset`, `LatentDataset`, `StudyDataset`, and
  `BidsH5StudyDataset`;
- `FmriSeries`, selectors, mask utilities, lazy arrays, caching, config IO,
  and conversions;
- BIDS-HDF5 read/write helpers including `bids_h5_dataset`,
  `bids_h5_scan_backend`, `subset_bids_h5`, `scan_manifest`,
  `reconstruct_voxels`, `parcellation_info`, and `encoding_info`;
- parity and unit tests for the dataset surface.

## Contract Conflicts To Resolve First

The consolidation should not start with a blind file move. These conflicts must
be resolved before broad imports are rewired.

| Area | Current `fmrimod` behavior | Current `fmridataset-py` behavior | Canonical decision |
| --- | --- | --- | --- |
| `SamplingFrame.samples` | time grid in seconds | 1-based sample indices | Keep one `fmrimod.sampling.SamplingFrame`; `samples` remains the time grid. Add explicit R-compatible helpers for integer sample indices when needed. |
| `SamplingFrame.blockids` | 0-based block ids | 1-based run ids | Keep one canonical internal 0-based representation; expose explicit `run_ids(one_based=True)` or generic helper behavior where R parity requires 1-based ids. |
| `SamplingFrame` class location | `fmrimod.sampling.SamplingFrame` | `fmridataset.sampling_frame.SamplingFrame` | `fmrimod.sampling.SamplingFrame` is the only timing class. Dataset code imports it. |
| `FmriDataset.get_data` | positional `run` returns one run | `rows`, `cols` return matrix slice | Canonical dataset API uses explicit matrix slicing and explicit run access. Avoid overloading positional `get_data(run)`. |
| `n_timepoints` | list of per-run lengths on GLM-facing datasets | total timepoint count | Canonical `n_timepoints` is total count; per-run lengths are `blocklens` or `run_lengths`. |
| `event_table` | optional `None` | non-null empty `DataFrame` by default | Canonical object stores a `DataFrame`; compatibility helpers can accept `None`. |
| `LatentDataset` | small score/loadings helper in `compat.py` | storage-backed latent dataset and backend | Keep the storage-backed implementation. Retire the temporary helper or turn it into a constructor path. |
| `data_chunks` | returns voxel index chunks in `compat.py` | yields typed data chunks with data and indices | Keep the richer typed chunk contract; expose index-only helper with a distinct name if still needed. |

## Target Public Shape

The accepted public dataset surface is:

```text
fmrimod.dataset
  BackendDims
  StorageBackend
  BackendRegistry
  FmriDataset
  MatrixDataset
  LatentDataset
  StudyDataset
  BidsH5StudyDataset
  matrix_dataset
  fmri_dataset
  fmri_mem_dataset
  nifti_backend
  h5_backend
  zarr_backend
  latent_backend
  study_backend
  bids_h5_dataset
  compress_bids_study
  data_chunks
  fmri_series
  selectors
  conversions
  mask utilities
```

Top-level `fmrimod` should expose only common workflow constructors and
analysis entry points. Backend registry details, selectors, BIDS-HDF5 helpers,
and storage-specific APIs should stay under `fmrimod.dataset`.

## Target Internal Shape

The internal package should remain modular:

```text
fmrimod/dataset/
  __init__.py
  backend_protocol.py
  backend_registry.py
  backend_constructors.py
  backend_methods.py
  dataset.py
  constructors.py
  data_access.py
  data_chunks.py
  series.py
  selectors.py
  mask_utils.py
  lazy_array.py
  conversions.py
  config.py
  errors.py
  study.py
  latent.py
  bids_h5.py
  bids_h5_write.py
  group_data.py
  adapters/
  backends/
```

Existing names may remain where compatibility cost is low, but the important
rule is that each concept has one implementation module and all public imports
point there.

## Migration Phases

### Phase 0: Freeze The Authority Boundary

1. Treat `../fmridataset-py` as the donor implementation, not a peer runtime
   package.
2. Make this plan the only detailed migration checklist.
3. Do not add feature work to `../fmridataset-py` unless the same change is
   planned for `fmrimod.dataset`.
4. If beads are created, each bead should point to a section in this file and
   contain only tracking metadata.

Exit criteria:

- This plan exists in `docs/contracts/`.
- Future tracking artifacts reference this file instead of duplicating it.

### Phase 1: Build The Public Surface Inventory

1. Inventory `fmridataset.__all__` and all public test-referenced names.
2. Inventory current `fmrimod.dataset` public names.
3. Classify each name as `adopt`, `merge`, `rename`, `compat_alias`,
   `internal_only`, or `scoped_out`.
4. Record conflict decisions in this file.
5. Add `tests/test_dataset/test_canonical_contracts.py` before broad code
   motion.

Exit criteria:

- No public name is moved without a classification.
- Every conflict in the table above has an executable test target assigned.
- The inventory covers all 146 donor `__all__` names or documents why the count
  changed.
- No inventory row remains `pending` or `unknown`.
- The canonical contract test file exists and names the expected single source
  of truth for timing, dataset, backend, latent, and chunk behavior.

### Phase 2: Establish The Canonical Core Contracts

1. Make `fmrimod.sampling.SamplingFrame` the only timing class used by dataset
   code.
2. Add explicit helpers for R-compatible integer samples and one-based run ids
   instead of changing `SamplingFrame.samples`.
3. Define one canonical `StorageBackend` protocol and `BackendDims`.
4. Define one canonical `FmriDataset` with:
   - matrix access by rows/columns;
   - explicit run access, for example `get_run_data(run)`;
   - total `n_timepoints`;
   - per-run lengths via `blocklens` or `run_lengths`;
   - one censor representation plus explicit run-censor access.
5. Add a local GLM adapter helper so model code consumes the canonical dataset
   without creating another dataset class.

Exit criteria:

- Current GLM/model tests pass against the canonical dataset contract.
- Dataset parity tests for timing, block lengths, data shape, and masks pass.
- There is no separate dataset-owned `SamplingFrame` implementation.

### Phase 3: Port Backend Infrastructure

1. Move the backend protocol, registry, backend constructors, backend methods,
   errors, and config IO into `fmrimod.dataset`.
2. Move matrix backend first because it is the smallest end-to-end fixture.
3. Move optional backends behind import guards:
   - HDF5 requires the `hdf5` extra;
   - NIfTI requires the `nifti` extra;
   - Zarr requires the `zarr` extra;
   - BIDS directory ingestion requires the `bids` extra.
4. Keep backend registration idempotent and deterministic.

Exit criteria:

- Matrix backend tests pass in the base environment.
- Optional backend tests skip with clear messages when dependencies are absent.
- `create_backend()` can construct the built-in backends that have installed
  dependencies.

Current status:

- Base error classes, `BackendDims`, `StorageBackend`, `BackendRegistry`,
  backend method facades, and `MatrixBackend` are implemented under
  `fmrimod.dataset`.
- The built-in registry currently registers only `matrix`. Optional backends
  remain unregistered until their implementations land.
- Focused verification currently reports `108 passed, 1 xfailed` for
  `tests/test_dataset`, `tests/test_model/test_fmri_model.py`, and
  `tests/test_glm/test_strategies.py`.

### Phase 4: Port Dataset Constructors And Accessors

1. Port `matrix_dataset`, `fmri_dataset`, and legacy constructor aliases.
2. Port data access helpers: `get_data`, `get_data_matrix`, `get_mask`.
3. Port dataset methods: `n_runs`, `n_timepoints`, `blocklens`,
   `get_total_duration`, `get_run_duration`, `get_TR`, `subject_ids`, and
   related accessors.
4. Resolve `TR` and `tr` spelling by accepting both at construction boundaries
   while storing one canonical representation.

Exit criteria:

- Constructor tests from `fmridataset-py` pass after import remapping.
- Existing `fmrimod` top-level constructor tests still pass.
- The public behavior for `n_timepoints`, `blocklens`, `TR`, and run access is
  documented in this file and enforced by tests.

Current status:

- `fmrimod.dataset.matrix_dataset` is now the canonical in-memory dataset
  constructor. Top-level `fmrimod.matrix_dataset` is a convenience wrapper
  that delegates into `fmrimod.dataset`.
- `fmrimod.dataset.fmri_dataset` is now the canonical constructor wrapper for
  existing adapter-like data sources.
- `fmrimod.dataset.matrix_backend` is now the canonical in-memory backend
  constructor.
- `fmrimod.dataset.matrix_dataset` constructs a canonical `MatrixBackend` and
  adapts it into the current `FmriDataset` wrapper.
- `FmriDataset.storage_backend` exposes the underlying backend when the data
  source is backend-backed.
- Matrix construction accepts both `tr` and the compatibility spelling `TR`,
  and requires them to agree when both are supplied.
- Matrix slicing through `get_data(rows=..., cols=...)`, explicit
  `get_run_data(run)`, total `n_timepoints`, and `run_lengths` / `blocklens`
  are covered by `tests/test_dataset/test_canonical_contracts.py`.
- Canonical free-function accessors now live under `fmrimod.dataset`:
  `get_data`, `get_data_matrix`, `get_mask`, `get_TR`, `get_run_lengths`,
  `get_total_duration`, `get_run_duration`, `n_runs`, `n_timepoints`,
  `blocklens`, `blockids`, `samples`, `all_timepoints`, `subject_ids`, and
  `is_sampling_frame`.
- `blockids(...)` defaults to canonical 0-based ids; R-style 1-based ids are
  only returned with `one_based=True`. `samples(...)` returns acquisition times
  in seconds.
- Canonical mask helpers `mask_to_logical` and `mask_to_volume` now live under
  `fmrimod.dataset`.
- Canonical typed chunk helpers now live under `fmrimod.dataset`: `DataChunk`,
  `ChunkIterator`, `data_chunk`, `data_chunks`, `collect_chunks`, and
  `exec_strategy`.
- The old bare voxel-index chunk behavior is kept only under the distinct name
  `voxel_index_chunks`. `data_chunks(...)` now returns typed `DataChunk`
  objects with data, voxel indices, row indices, and chunk number.
- Canonical spatial selectors now live under `fmrimod.dataset.selectors`:
  `SeriesSelector`, `IndexSelector`, `AllSelector`, `ROISelector`,
  `VoxelSelector`, `SphereSelector`, `MaskSelector`, and their constructor
  helpers.
- Canonical series helpers now live under `fmrimod.dataset.series`:
  `FmriSeries`, `fmri_series`, `new_fmri_series`, `is_fmri_series`,
  `as_matrix`, `to_dataframe`, `as_tibble`, `resolve_selector`,
  `resolve_timepoints`, and `series`.
- Selector/series tests prove that selectors resolve against canonical dataset
  masks and dimensions, `fmri_series(...)` uses canonical `rows`/`cols` data
  access, temporal metadata comes from `SamplingFrame.samples` and 0-based run
  ids, and series selection composes with typed `DataChunk` output.
- Focused verification currently reports `128 passed` for
  `tests/test_dataset`, `tests/test_model/test_fmri_model.py`, and
  `tests/test_glm/test_strategies.py`.

### Phase 5: Port Higher-Level Dataset Objects

1. Port `MatrixDataset`, `LatentDataset`, `StudyDataset`, and `FmriGroup`.
2. Port `FmriSeries` and selector machinery.
3. Merge or retire temporary `fmrimod.dataset.compat.LatentDataset`.
4. Ensure study and group containers use the same mask and timing validation
   rules as single-subject datasets.
5. Shape study/group/latent objects around the production seam:
   dataset construction, first-level fitting, contrast output, and group
   reduction.

Exit criteria:

- Study, latent, group, selector, and series tests pass after import remapping.
- There is one `LatentDataset` class.
- There is one chunk/data-series representation.
- At least one study/group object path feeds a modeling or group-reduction
  workflow without dropping into ad hoc NumPy reshaping.

### Phase 6: Port BIDS-HDF5 Last

1. Port `BidsH5ScanBackend`, `SharedH5Connection`, and the BIDS-HDF5 study
   reader after the core backend registry is stable.
2. Port the writer only after read-side fixtures are green.
3. Preserve the tested BIDS-HDF5 contracts:
   - archive root advertises `format=bids_h5_study`;
   - compression mode is `parcellated` or `latent`;
   - feature-space scan backends use `spatial=(K, 1, 1)` and an all-true mask;
   - parcellated `n_features` comes from `/parcellation/cluster_ids`;
   - latent `n_features` comes from `/latent_meta/n_components`;
   - `reconstruct_voxels(scan_name)` validates `basis @ loadings.T + offset`.
4. Keep direct `nibabel` and PyBIDS use at ingestion/IO boundaries. Canonical
   in-memory image/mask/contrast objects should be neuroim-compatible.

Exit criteria:

- `tests/test_bids_h5_dataset.py` equivalent passes under `fmrimod.dataset`.
- BIDS-HDF5 tests do not require full BIDS directory ingestion unless the
  `bids` extra is installed.
- A BIDS-HDF5-backed dataset can enter the same modeling path as a matrix
  dataset where the required data are present.

### Phase 7: Integrate With Modeling Workflows

1. Update GLM, single-trial, simulation, and docs examples to use the canonical
   dataset access helpers.
2. Replace assumptions that `n_timepoints` is a list with `blocklens` or
   `run_lengths`.
3. Replace positional `get_data(run)` usage with explicit run access.
4. Ensure `matrix_dataset(...)` returns an object that can flow directly into
   `fmri_lm`, LSS/LSA/OASIS helpers, and simulation examples.
5. Add or update at least one flagship-style benchmark/parity workflow when a
   dataset feature becomes workflow-bearing.

Exit criteria:

- Existing model, GLM, single-trial, and simulation tests pass.
- At least one end-to-end test fits a GLM from a canonical `matrix_dataset`.
- At least one chunked fit path uses the canonical chunk contract.
- Any unavoidable divergence from R/Nilearn/FitLins behavior is documented in
  `docs/contracts/CAVEATS.md` with an owning mote and retirement criterion.

### Phase 8: Add The Compatibility Package

1. Convert the old `fmridataset` distribution into a facade that depends on
   `fmrimod`.
2. Re-export public names from `fmrimod.dataset`.
3. Keep compatibility tests focused on identity and import behavior, for
   example:

   ```python
   import fmridataset
   import fmrimod.dataset as dataset

   assert fmridataset.FmriDataset is dataset.FmriDataset
   assert fmridataset.matrix_dataset is dataset.matrix_dataset
   ```

4. Add deprecation messaging only at package/documentation boundaries, not on
   every function call.

Exit criteria:

- The compatibility package has no backend implementation files.
- Public class and constructor re-exports are identity-equivalent.
- Users can migrate imports mechanically from `fmridataset` to
  `fmrimod.dataset`.

Current status:

- A thin `fmridataset` package now ships from this distribution and is included
  in the wheel/sdist package list. Its package-local README is a migration
  pointer to `fmrimod.dataset`, not a parallel documentation set.
- The facade currently re-exports implemented canonical objects only:
  `FmriDataset`, `MatrixBackend`, `SamplingFrame`, backend protocol/registry
  objects, `matrix_backend`, `matrix_dataset`, `fmri_dataset`, backend method
  facades, data accessors, temporal metadata accessors, typed chunk helpers,
  selector helpers, series helpers, mask helpers, and dataset error classes.
- The facade has no independent dataset, backend, timing, or registry
  implementation. Submodules such as `fmridataset.backend_protocol`,
  `fmridataset.backend_registry`, `fmridataset.data_access`,
  `fmridataset.data_chunks`, `fmridataset.dataset_constructors`,
  `fmridataset.dataset_methods`, `fmridataset.fmri_series`,
  `fmridataset.mask_utils`, `fmridataset.selectors`, and
  `fmridataset.backends.matrix_backend` are re-export modules only.
- `tests/test_dataset/test_fmridataset_facade.py` enforces identity-equivalent
  root and submodule imports. The former strict xfail in
  `tests/test_dataset/test_canonical_contracts.py` now passes.
- Donor names whose canonical implementations are not yet ported remain absent
  from the facade until their `fmrimod.dataset` owners exist. The facade must
  not add placeholder implementations.

### Phase 9: Documentation Consolidation

1. Move canonical dataset tutorials into `fmrimod` docs.
2. Replace `fmridataset` docs with a short migration pointer.
3. Add one `fmrimod.dataset` API overview.
4. Update examples so dataset IO appears as the substrate for modeling, not as
   a separate project.
5. Align examples with the mission: typed composition first, convenience
   wrappers only as thin facades.

Exit criteria:

- Only `fmrimod` contains detailed dataset API docs.
- Compatibility-package docs point to `fmrimod.dataset`.
- No duplicate BIDS-HDF5 or backend registry tutorials exist.

Current status:

- `docs/tutorials/datasets.qmd` is the canonical dataset API overview. It
  covers timing, matrix/image/archive IO, latent data, chunking, and migration
  from `fmridataset` as one modeling substrate.
- `fmridataset/README.md` is a short migration pointer and contains no backend
  or BIDS-HDF5 tutorial content.

### Phase 10: Verification And Release Gate

Run the verification sequence before considering the consolidation complete:

```bash
python -m pytest tests/test_dataset
python -m pytest tests/test_glm tests/test_model tests/test_single tests/test_simulate
python -m pytest tests/test_import_cycles.py tests/test_type_hints_resolution.py
python -m mypy fmrimod
git diff --check
```

Optional dependency gates should run in environments that install the relevant
extras:

```bash
python -m pytest tests/test_dataset -m "not parity"
python -m pytest tests/test_dataset/test_bids_h5_dataset.py
```

Exit criteria:

- Base tests pass without HDF5, NIfTI, Zarr, PyBIDS, or Dask installed.
- Optional tests pass when their extras are installed.
- Strict typing covers the migrated dataset package.
- Import-cycle checks pass.
- The old `fmridataset` facade tests pass.

## Optional Dependency Plan

Use these extras in `pyproject.toml`:

| Extra | Dependencies | Owns |
| --- | --- | --- |
| `hdf5` | `h5py` | HDF5 backend, latent HDF5, BIDS-HDF5 read/write |
| `nifti` | `nibabel` | NIfTI backend and image metadata |
| `zarr` | `zarr` | Zarr backend |
| `dask` | `dask[array]` | delayed/lazy array conversion |
| `cache` | `cachetools` | optional LRU cache helpers |
| `bids` | `pybids`, `nibabel` | raw BIDS directory ingestion |
| `full` | all of the above | complete dataset IO stack |

The base install should remain NumPy/Pandas/SciPy-oriented and must support
matrix datasets and modeling workflows without heavy IO dependencies.

Optional IO extras are implementation details of dataset ingestion and storage,
not alternate modeling pathways. Once data are loaded, downstream modeling
should see the same canonical dataset/series/chunk contracts regardless of
whether the source was matrix, HDF5, NIfTI, Zarr, BIDS-HDF5, or neuroim.

## Reviewable Work Packages

These are the natural patch boundaries. If beads are created, use these titles
and point each bead to this section instead of copying the details.

1. Dataset consolidation inventory and contract tests.
2. Canonical `SamplingFrame` and dataset protocol reconciliation.
3. Backend protocol, registry, and matrix backend port.
4. Latent dataset/backend port, retiring the temporary helper.
5. Study/group dataset port aligned with group inference.
6. Optional backend extras and constructors.
7. HDF5, NIfTI, Zarr backend port using canonical backend contracts.
8. BIDS-HDF5 reader and writer port.
9. Modeling integration: GLM, single-trial, simulation, examples, and
   benchmark/parity workflows.
10. Compatibility `fmridataset` facade.
11. Documentation consolidation and release gate.

## Completion Definition

The consolidation is complete only when all of these are true:

- `fmrimod.dataset` owns the implementation.
- `fmridataset`, if present, is a re-export facade.
- There is one timing class.
- There is one storage backend protocol and one backend registry.
- There is one public `FmriDataset` class.
- There is one `LatentDataset` class.
- There is one chunk contract.
- All current `fmridataset-py` tests that are still in scope have either been
  ported or explicitly classified as `scoped_out` in this file.
- Existing `fmrimod` modeling tests pass against the canonical dataset objects.
- Optional backends are extras, not hard base dependencies.
- Workflow-bearing dataset paths are validated through the same modeling seam,
  not just through import or constructor smoke tests.
- Any active dataset-related parity caveat is either closed or has a mote owner
  and concrete retirement criterion in `docs/contracts/CAVEATS.md`.

## Maintenance Rule

When a consolidation patch changes public dataset behavior, update this file in
the same patch. Do not create a new migration checklist elsewhere.
