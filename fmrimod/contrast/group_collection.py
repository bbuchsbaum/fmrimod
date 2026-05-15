"""Collect first-level ``ContrastResult`` objects into a group dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fmrimod.glm.contrasts import ContrastResult
from fmrimod.glm.spatial import SpatialContext
from fmrimod.group.dataset import GroupDataset, contrast_intent_payload, group_dataset
from fmrimod.group.errors import AdapterContractError
from fmrimod.group.space import SampleLabelSpace, VoxelSpace


def group_dataset_from_contrasts(
    contrasts: Mapping[str, ContrastResult | Sequence[ContrastResult]],
    *,
    covariates: pd.DataFrame | None = None,
) -> GroupDataset:
    """Lower one first-level contrast per subject into a ``GroupDataset``."""
    if not isinstance(contrasts, Mapping) or not contrasts:
        raise AdapterContractError("contrasts must be a non-empty subject mapping")

    subjects = tuple(str(subject) for subject in contrasts)
    results = tuple(_single_result(value) for value in contrasts.values())
    contrast_name = _shared_contrast_name(results)
    estimates = tuple(
        _one_dimensional(result.estimate, field="estimate") for result in results
    )
    ses = tuple(_required_se(result) for result in results)
    stats = tuple(_one_dimensional(result.stat, field="stat") for result in results)
    p_values = tuple(
        _one_dimensional(result.p_value, field="p_value") for result in results
    )
    n_samples = int(estimates[0].size)
    for subject, estimate, se, stat, p_value in zip(
        subjects, estimates, ses, stats, p_values
    ):
        if estimate.size != n_samples or se.size != n_samples:
            raise AdapterContractError(
                f"ContrastResult for subject {subject!r} has incompatible "
                "estimate/se length"
            )
        if stat.size != n_samples or p_value.size != n_samples:
            raise AdapterContractError(
                f"ContrastResult for subject {subject!r} has incompatible "
                "stat/p_value length"
            )

    spatial = _shared_spatial(subjects, results)
    if spatial is None:
        labels = tuple(f"feature{i:03d}" for i in range(n_samples))
        space = SampleLabelSpace(labels)
        row_data = pd.DataFrame(index=pd.Index(labels, name="sample"))
    else:
        mask_idx = np.flatnonzero(np.asarray(spatial.mask, dtype=bool).reshape(-1))
        space = VoxelSpace(
            shape=spatial.spatial_shape,
            affine=spatial.affine,
            mask_idx=mask_idx,
            storage="packed",
        )
        row_data = pd.DataFrame(
            {"voxel_index": mask_idx.astype(int)},
            index=pd.Index([f"voxel{int(i)}" for i in mask_idx], name="sample"),
        )

    contrast_data = pd.DataFrame(
        [_contrast_metadata(results[0])],
        index=pd.Index([contrast_name], name="contrast"),
    )
    return group_dataset(
        {
            "beta": _stack_assay(estimates),
            "se": _stack_assay(ses),
            "t": _stack_assay(stats),
            "p": _stack_assay(p_values),
        },
        space=space,
        subjects=subjects,
        contrasts=(contrast_name,),
        col_data=_align_covariates(covariates, subjects),
        row_data=row_data,
        contrast_data=contrast_data,
        metadata={
            "source_format": "contrast_results",
            "subject_contrast_metadata": {
                subject: _contrast_metadata(result)
                for subject, result in zip(subjects, results)
            },
        },
    )


def _single_result(value: ContrastResult | Sequence[ContrastResult]) -> ContrastResult:
    if isinstance(value, ContrastResult):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raise NotImplementedError(
            "Multi-contrast group collection is deferred to v2; see "
            "docs/contracts/parametric_contrast_sugar_v1.md"
        )
    raise AdapterContractError(
        "group_dataset_from_contrasts values must be ContrastResult objects"
    )


def _shared_contrast_name(results: Sequence[ContrastResult]) -> str:
    names = tuple(str(result.name) for result in results)
    if any(not name for name in names):
        raise AdapterContractError("ContrastResult.name is required")
    if len(set(names)) != 1:
        raise AdapterContractError(
            "v1 group_dataset_from_contrasts requires one shared contrast name"
        )
    return names[0]


def _one_dimensional(value: object, *, field: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise AdapterContractError(f"ContrastResult.{field} must be one-dimensional")
    return arr


def _required_se(result: ContrastResult) -> NDArray[np.float64]:
    if result.se is None:
        raise AdapterContractError(
            "ContrastResult.se is required for v1 group_dataset_from_contrasts"
        )
    return _one_dimensional(result.se, field="se")


def _stack_assay(values: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    return np.column_stack(values).astype(np.float64)[:, :, np.newaxis]


def _shared_spatial(
    subjects: Sequence[str],
    results: Sequence[ContrastResult],
) -> SpatialContext | None:
    spatial = tuple(result.spatial for result in results)
    present = tuple(item for item in spatial if item is not None)
    if not present:
        return None
    if len(present) != len(spatial):
        raise AdapterContractError(
            "SpatialContext is present for only some subjects; refusing to "
            "silently drop spatial metadata"
        )
    first = present[0]
    for subject, item in zip(subjects[1:], present[1:]):
        if not _same_spatial(first, item):
            raise AdapterContractError(
                "Incompatible SpatialContext across ContrastResult objects; "
                f"first subject {subjects[0]!r} differs from {subject!r}"
            )
    return first


def _same_spatial(left: SpatialContext, right: SpatialContext) -> bool:
    left_affine = (
        np.eye(4, dtype=np.float64)
        if left.affine is None
        else np.asarray(left.affine, dtype=np.float64)
    )
    right_affine = (
        np.eye(4, dtype=np.float64)
        if right.affine is None
        else np.asarray(right.affine, dtype=np.float64)
    )
    return (
        left.spatial_shape == right.spatial_shape
        and np.array_equal(
            np.asarray(left.mask, dtype=bool),
            np.asarray(right.mask, dtype=bool),
        )
        and np.allclose(left_affine, right_affine)
    )


def _contrast_metadata(result: ContrastResult) -> dict[str, object]:
    return {
        "name": str(result.name),
        "contrast_intent": _json_string(contrast_intent_payload(result)),
        "touched_columns": _json_string(tuple(result.touched_columns)),
        "touched_column_details": _json_string(tuple(result.touched_column_details)),
        "caveats": _json_string(tuple(result.caveats)),
        "stat_type": str(result.stat_type),
        "spatial": result.spatial is not None,
    }


def _json_string(value: object) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _align_covariates(
    covariates: pd.DataFrame | None,
    subjects: Sequence[str],
) -> pd.DataFrame | None:
    if covariates is None:
        return None
    if not isinstance(covariates, pd.DataFrame):
        raise AdapterContractError("covariates must be a pandas DataFrame")
    frame = covariates.copy()
    if "subject" in frame.columns:
        indexed = frame.set_index("subject", drop=False)
        missing = [subject for subject in subjects if subject not in indexed.index]
        if missing:
            raise AdapterContractError(
                "Covariates are missing for subjects: " + ", ".join(missing)
            )
        frame = indexed.loc[list(subjects)].reset_index(drop=True)
    if len(frame) != len(subjects):
        raise AdapterContractError(
            f"covariates rows ({len(frame)}) must match subjects ({len(subjects)})"
        )
    return frame


__all__ = ["group_dataset_from_contrasts"]
