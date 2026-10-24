# 0004 — One storage contract and a typed space for datasets

Status: Accepted (maintainer, 2026-10-24)
Date: 2026-10-24
Decides: R-source drift v2 steward decision **D1** (fmridataset direction);
the S2 part of row DS-02.

## Context

R fmridataset deleted its pre-frame surface in `f305b44` and replaced it
with the `fmri_frame` architecture (`inst/architecture/ADR-001` to
`ADR-010`). No R code corresponds to `fmrimod.dataset` any more
([`r_source_drift_v2/fmridataset.md`](../r_source_drift_v2/fmridataset.md),
DS-25), so drift is judged against the semantic rules the redesign wrote
down rather than against function names.

fmrimod's dataset layer predated those rules and had grown two
abstractions for one job:

- a storage contract (`StorageBackend`: `get_data(rows, cols)`, flat mask),
  which the consolidation plan names "the only backend contract";
- six run-wise adapters in front of it (`NumpyAdapter`, `NeuroVecAdapter`,
  `NibabelAdapter`, `BackendAdapter`, and private latent and BIDS-HDF5
  subject adapters).

Each adapter re-implemented runs, masks and affines its own way:

- masks came back flat, `(n, 1, 1)` or 3-D depending on the path;
- `get_data(rows=, cols=)` materialised everything and then sliced;
- the GLM, results writer and study lowering probed `dataset._source` for
  an affine.

Spatial identity was a mask plus an optional affine. Shape was the only
check, which is how DS-02, DS-07 and DS-11 arose.

## Options (from the v2 inventory)

- **(a)** Freeze R `3ae565e` as the permanent spec; port correctness only.
- **(b)** Adopt R's frames, array sources, feature spaces and entities.
- **(c)** Keep the `fmrimod.dataset` API; adopt individual frame rules
  where they fix real defects.

## Decision

**(c).** We don't build the frame architecture: frames, collections,
studies and entity registries, lazy array sources, FDS manifests. It would
be a second first-level container, against the plan's single-`FmriDataset`
rule. A pandas subject table is the Python idiom, and `GroupDataset`
already plays the frame role at the group stage.

Adopted from the redesign:

1. **One storage contract.** `FmriDataset` is one `StorageBackend` plus
   one `SamplingFrame`. A run is a row range, and `rows`/`cols` reads go to
   the backend, which reads only what is asked. Run-wise access is
   derived, not stored.
2. **Spatial identity is the space, never the shape** (ADR-001,
   `7cb121e`). `VolumeSpace` (`fmrimod/dataset/space.py`) types a
   dataset's columns:
   - the grid;
   - the in-mask voxels, in C order;
   - the 4x4 affine, or `None` when unknown;
   - the shape the mask is reported in.

   Geometry is never assumed: an unknown affine only matches another
   unknown affine. Backends report a `space`, multi-run sources stack
   through `ConcatBackend`, and that refuses any part whose grid, affine
   or mask differs from part 0's.
3. **Per-source run lengths** (`631cf0d`). Backends report
   `source_run_lengths`, and a sampling frame with one run per source must
   match them run by run.
4. **Typed spatial hand-off to the group stage.** Voxel-routed study
   estimates of datasets with a known affine lower to the subjects' common
   packed `VoxelSpace`. Matrix data and explicit `sample_labels` keep
   `SampleLabelSpace`.

Why a native type rather than `neuroim.NeuroSpace`: neuroim-python is an
optional extra (a git dependency, not on PyPI), and matrix datasets have
no geometry. `VolumeSpace` converts from `NeuroSpace` at the IO boundary
and to `fmrimod.group.VoxelSpace` at the seam. Spatial comparison of
neuroim objects uses `NeuroSpace.compatible_with`, so neuroim stays the
substrate wherever it is present. Why not `VoxelSpace` itself:
- it turns an unknown affine into the identity;
- it cannot represent an empty mask;
- it lives in the downstream `group` stage.

## Compatibility

- `FmriDataset(data_source, event_table, censor)` is unchanged.
  `sampling_frame=` is new, keyword-only, and required with a bare backend.
- The public adapters remain, as thin facades that expose `.backend`.
  Any other `DatasetProtocol` object (a user-written adapter) is read
  through `RunwiseSourceBackend`, the one remaining positional
  `get_data(run)` call.
- New public surface: `FmriDataset.space`, `get_affine()` (`None` when
  unknown), `run_rows(run)`, and `SpatialContext.from_space`.
- Behaviour changes: `NumpyAdapter` refuses a mask that selects a
  different number of voxels than there are columns. `NibabelAdapter`
  image masks keep voxels above zero, as `NiftiBackend` always did.

## Not adopted, and why

| R feature | Disposition |
| --- | --- |
| `fmri_frame` / `fmri_collection` / `fmri_study`, entity registries (DS-15, DS-18) | Skip: a second container; pandas tables are the idiom. |
| Array sources, block plans, memory budgets (DS-16) | Skip until a MISSION-justified lazy-array request exists. The backends' sliced reads are the pushdown that matters today. |
| Surface, parcel and composite first-level feature spaces (DS-17) | Defer. Parcels and bases exist at the group stage. A first-level surface dataset needs a `VolumeSpace` sibling, not a change to it. |
| FDS manifests, digests, provenance graph (DS-19) | Defer. Upstream declares its canonicalisation R-only. |
| R's negative-index exclusion and duplicate refusal (ADR-010) | Not ported. Python keeps numpy's semantics. |

## Consequences

- DS-07, DS-10 and DS-11 are fixed structurally rather than patched per
  adapter. DS-02 is closed, including its S2 part.
- New backends implement one ABC and report a space. Composite sources
  compose by `ConcatBackend`.
- Still open:
  - mask binarisation differs between array masks (`astype(bool)`) and
    image masks (`> 0`). R uses `as.logical`. A follow-up should pick one
    rule.
  - `LatentDataset`'s voxel view has no affine, because latent backends
    carry none.
