"""Study-level dataset containers that feed native group analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .errors import FmriDatasetError
from .fmri_dataset import FmriDataset

if TYPE_CHECKING:
    from fmrimod.group import GroupDataset, VoxelSpace

MaskStrategy = Literal["subject_specific", "intersect", "union"]
# Group rows that receive a subject's estimate entries, and those entries.
_Route = tuple[NDArray[np.intp], NDArray[np.intp]]


def _unwrap_dataset(value: object, *, row: int, dataset_col: str) -> FmriDataset:
    """Accept R-style length-one list cells while storing canonical datasets."""
    if isinstance(value, FmriDataset):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        only = value[0]
        if isinstance(only, FmriDataset):
            return only
    raise FmriDatasetError(
        f"{dataset_col} row {row} must contain an FmriDataset or a length-one "
        "sequence containing an FmriDataset"
    )


def _as_1d_float_array(
    value: object,
    *,
    assay: str,
    subject: str,
    contrast: str,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise FmriDatasetError(
            f"{assay} for subject '{subject}', contrast '{contrast}' "
            "must be one-dimensional"
        )
    return arr


def _voxel_space(dataset: FmriDataset) -> VoxelSpace:
    """Return a dataset's mask as a packed voxel space (C-order voxel ids)."""
    from fmrimod.group import VoxelSpace

    mask = np.asarray(dataset.get_mask(), dtype=bool)
    shape = mask.shape if mask.ndim == 3 else (int(mask.size), 1, 1)
    get_affine = getattr(getattr(dataset, "_source", None), "get_affine", None)
    affine = None
    if callable(get_affine):
        affine = np.asarray(get_affine(), dtype=np.float64)
    return VoxelSpace(
        shape=shape,
        affine=affine if affine is not None and affine.shape == (4, 4) else None,
        mask_idx=np.flatnonzero(mask),
        storage="packed",
    )


def _route_samples(
    datasets: Sequence[FmriDataset],
    subjects: Sequence[str],
    lengths: Sequence[int],
    mask_strategy: MaskStrategy,
) -> tuple[int, list[_Route], NDArray[np.intp] | None]:
    """Map each subject's estimate entries onto the group sample axis.

    Returns the number of group samples, one ``(rows, entries)`` pair per
    subject (the group rows that receive that subject's estimate entries),
    and the voxel id of each group row, or ``None`` when the estimates are
    not voxel maps of the subjects' masks (e.g. ROI summaries).

    Voxel maps are matched by voxel id through each subject's own mask, never
    by position (R ``e998bd4``): ``subject_specific`` refuses non-identical
    masks, ``intersect``/``union`` align them with
    :func:`fmrimod.group.common_mask`, and masks on different voxel grids are
    refused.
    """
    from fmrimod.group import GroupSpaceError, assert_compatible_spaces, common_mask

    if any(n != dataset.n_voxels for n, dataset in zip(lengths, datasets, strict=True)):
        if len(set(lengths)) != 1:
            raise FmriDatasetError(
                "all contrast estimates must have the same sample length"
            )
        positions = np.arange(lengths[0], dtype=np.intp)
        return lengths[0], [(positions, positions)] * len(datasets), None

    spaces = [_voxel_space(dataset) for dataset in datasets]
    for subject, space in zip(subjects, spaces, strict=True):
        try:
            assert_compatible_spaces(spaces[0], space)
        except GroupSpaceError as exc:
            raise FmriDatasetError(
                f"subject '{subject}' mask is on a different voxel grid than "
                f"subject '{subjects[0]}': {exc}"
            ) from exc

    common = spaces[0]
    if mask_strategy == "subject_specific":
        differing = [
            subject
            for subject, space in zip(subjects, spaces, strict=True)
            if not np.array_equal(
                np.asarray(space.mask_idx), np.asarray(common.mask_idx)
            )
        ]
        if differing:
            raise FmriDatasetError(
                "subject masks differ under mask_strategy='subject_specific' "
                f"(subjects: {', '.join(differing)}); use 'intersect' or "
                "'union' to align voxel estimates by voxel id"
            )
    else:
        rule: Literal["intersection", "union"] = (
            "intersection" if mask_strategy == "intersect" else "union"
        )
        try:
            for space in spaces[1:]:
                _, _, common = common_mask(common, space, rule=rule)
        except GroupSpaceError as exc:
            raise FmriDatasetError(
                f"subject masks have no voxels in common: {exc}"
            ) from exc

    routes = []
    for space in spaces:
        rows, entries, _ = common_mask(common, space, rule="intersection")
        routes.append((rows, entries))
    return common.n_samples, routes, np.asarray(common.mask_idx, dtype=np.intp)


class _ContrastResultLike(Protocol):
    """Duck-typed view of fmrimod.glm.ContrastResult for study lowering.

    Only the attributes the study-to-group lowering reads are declared; this
    keeps the typed seam decoupled from the concrete result class while still
    rejecting object-typed lookups in mypy.
    """

    @property
    def estimate(self) -> object: ...

    @property
    def stat_type(self) -> str: ...


def _normalise_contrast_results(
    contrast_results: Mapping[str, object],
    *,
    subjects: Sequence[str],
) -> tuple[tuple[str, ...], dict[tuple[str, str], object]]:
    """Return contrast names plus a (subject, contrast) result lookup."""
    missing = [subject for subject in subjects if subject not in contrast_results]
    if missing:
        raise FmriDatasetError(
            "contrast_results is missing subjects: " + ", ".join(missing)
        )

    contrast_order: list[str] = []
    lookup: dict[tuple[str, str], object] = {}
    for subject in subjects:
        value = contrast_results[subject]
        nested: Mapping[object, object]
        if hasattr(value, "estimate"):
            contrast_name = str(getattr(value, "name", "c1") or "c1")
            nested = {contrast_name: value}
        elif isinstance(value, Mapping):
            nested = value
        else:
            raise FmriDatasetError(
                f"contrast_results for subject '{subject}' must be a contrast result "
                "or a mapping of contrast name to result"
            )

        for contrast, result in nested.items():
            contrast_name = str(contrast)
            if contrast_name not in contrast_order:
                contrast_order.append(contrast_name)
            lookup[(subject, contrast_name)] = result

    if not contrast_order:
        raise FmriDatasetError("contrast_results must contain at least one contrast")

    for subject in subjects:
        missing_contrasts = [
            contrast for contrast in contrast_order if (subject, contrast) not in lookup
        ]
        if missing_contrasts:
            raise FmriDatasetError(
                f"subject '{subject}' is missing contrasts: "
                + ", ".join(missing_contrasts)
            )

    return tuple(contrast_order), lookup


@dataclass(frozen=True)
class StudyDataset:
    """Subject table plus per-subject :class:`FmriDataset` objects.

    The container owns study-level subject metadata and provides the canonical
    path from first-level contrast results into ``fmrimod.group.GroupDataset``.
    It deliberately does not implement group reducers itself.
    """

    subjects: pd.DataFrame
    id: str  # noqa: A003 - mirrors the donor fmridataset/fmri_group contract.
    dataset_col: str = "dataset"
    space: str | None = None
    mask_strategy: MaskStrategy = "subject_specific"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subjects, pd.DataFrame):
            raise FmriDatasetError("subjects must be a pandas DataFrame")
        if self.id not in self.subjects.columns:
            raise FmriDatasetError(f"subject id column '{self.id}' is missing")
        if self.dataset_col not in self.subjects.columns:
            raise FmriDatasetError(f"dataset column '{self.dataset_col}' is missing")
        if self.mask_strategy not in {"subject_specific", "intersect", "union"}:
            raise FmriDatasetError(
                "mask_strategy must be one of: subject_specific, intersect, union"
            )

        frame = self.subjects.copy()
        ids = frame[self.id].map(str)
        if ids.isna().any() or (ids == "").any():
            raise FmriDatasetError(f"subject id column '{self.id}' must be non-empty")
        if ids.duplicated().any():
            dupes = ", ".join(ids[ids.duplicated()].unique())
            raise FmriDatasetError(f"subject ids must be unique; duplicates: {dupes}")
        frame[self.id] = ids
        frame[self.dataset_col] = [
            _unwrap_dataset(value, row=i, dataset_col=self.dataset_col)
            for i, value in enumerate(frame[self.dataset_col])
        ]

        object.__setattr__(self, "subjects", frame)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def subject_ids(self) -> tuple[str, ...]:
        """Return subject identifiers in study order."""
        return tuple(str(value) for value in self.subjects[self.id])

    @property
    def n_subjects(self) -> int:
        """Return the number of subjects in the study."""
        return len(self.subject_ids)

    @property
    def datasets(self) -> tuple[FmriDataset, ...]:
        """Return per-subject datasets in study order."""
        return tuple(self.subjects[self.dataset_col])

    @property
    def col_data(self) -> pd.DataFrame:
        """Return subject metadata indexed by subject id."""
        frame = self.subjects.drop(columns=[self.dataset_col]).copy()
        return frame.set_index(self.id, drop=False)

    def get_subject_dataset(self, subject: str) -> FmriDataset:
        """Return one subject's dataset by id."""
        subject = str(subject)
        matches = self.subjects[self.subjects[self.id] == subject]
        if matches.empty:
            raise FmriDatasetError(f"unknown subject id: {subject}")
        return cast(FmriDataset, matches.iloc[0][self.dataset_col])

    def to_group_dataset(
        self,
        contrast_results: Mapping[str, object],
        *,
        sample_labels: Sequence[object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> GroupDataset:
        """Materialize first-level contrast results for native group reducers.

        ``contrast_results`` is keyed by subject id. Each value may be a single
        ``ContrastResult``-like object with ``estimate`` and optional ``se`` /
        ``stat`` / ``p_value`` attributes, or a mapping of contrast name to such
        objects.
        """
        from fmrimod.group import GroupDataset, SampleLabelSpace
        from fmrimod.group.dataset import (
            CONTRAST_INTENT_COLUMN,
            contrast_intent_payload_json,
        )

        subjects = self.subject_ids
        contrasts, lookup = _normalise_contrast_results(
            contrast_results, subjects=subjects
        )

        estimates = {
            (subject, contrast): _as_1d_float_array(
                cast(_ContrastResultLike, result).estimate,
                assay="estimate",
                subject=subject,
                contrast=contrast,
            )
            for (subject, contrast), result in lookup.items()
        }
        lengths: list[int] = []
        for subject in subjects:
            sizes = {int(estimates[(subject, c)].size) for c in contrasts}
            if len(sizes) != 1:
                raise FmriDatasetError(
                    "all contrast estimates must have the same sample length"
                )
            lengths.append(sizes.pop())
        n_samples, routes, voxel_index = _route_samples(
            self.datasets, subjects, lengths, self.mask_strategy
        )

        assays: dict[str, NDArray[np.float64]] = {}

        def put(
            assay_name: str,
            subject_index: int,
            contrast_index: int,
            values: NDArray[np.float64],
        ) -> None:
            if assay_name not in assays:
                assays[assay_name] = np.full(
                    (n_samples, len(subjects), len(contrasts)),
                    np.nan,
                    dtype=np.float64,
                )
            rows, entries = routes[subject_index]
            assays[assay_name][rows, subject_index, contrast_index] = values[entries]

        for subject_index, subject in enumerate(subjects):
            for contrast_index, contrast in enumerate(contrasts):
                result = lookup[(subject, contrast)]
                estimate = estimates[(subject, contrast)]
                put("beta", subject_index, contrast_index, estimate)

                for attr, assay_name in (
                    ("se", "se"),
                    ("stat", "stat"),
                    ("p_value", "p"),
                ):
                    value = getattr(result, attr, None)
                    if value is None:
                        continue
                    arr = _as_1d_float_array(
                        value, assay=assay_name, subject=subject, contrast=contrast
                    )
                    if arr.size != estimate.size:
                        raise FmriDatasetError(
                            f"{assay_name} arrays must match estimate sample length"
                        )
                    put(assay_name, subject_index, contrast_index, arr)

        labels = (
            tuple(str(value) for value in sample_labels)
            if sample_labels is not None
            else tuple(f"sample{i + 1}" for i in range(n_samples))
        )
        if len(labels) != n_samples:
            raise FmriDatasetError(
                "sample_labels length must match contrast estimate sample length"
            )

        contrast_data = pd.DataFrame(index=pd.Index(contrasts, name="contrast"))
        intent_payloads: list[str | None] = []
        for contrast in contrasts:
            payloads = [
                contrast_intent_payload_json(lookup[(subject, contrast)])
                for subject in subjects
            ]
            present = [payload for payload in payloads if payload is not None]
            if not present:
                intent_payloads.append(None)
                continue
            if len(present) != len(payloads):
                raise FmriDatasetError(
                    "contrast_intent payload is missing for some subjects in "
                    f"contrast '{contrast}'"
                )
            first = present[0]
            if any(payload != first for payload in present[1:]):
                raise FmriDatasetError(
                    f"contrast_intent payloads differ for contrast '{contrast}'"
                )
            intent_payloads.append(first)
        if any(payload is not None for payload in intent_payloads):
            contrast_data[CONTRAST_INTENT_COLUMN] = intent_payloads
        if all(
            hasattr(lookup[(subjects[0], contrast)], "stat_type")
            for contrast in contrasts
        ):
            contrast_data["stat_type"] = [
                str(cast(_ContrastResultLike, lookup[(subjects[0], contrast)]).stat_type)
                for contrast in contrasts
            ]

        row_data = pd.DataFrame(index=pd.Index(labels, name="sample"))
        if voxel_index is not None:
            row_data["voxel_index"] = voxel_index.astype(int)

        group_metadata = {
            "source_format": "study_dataset",
            "source": "fmrimod.dataset.StudyDataset",
            "space": self.space,
            "mask_strategy": self.mask_strategy,
            **dict(self.metadata),
        }
        if metadata:
            group_metadata.update(metadata)

        return GroupDataset(
            assays=assays,
            space=SampleLabelSpace(labels),
            subjects=subjects,
            contrasts=contrasts,
            col_data=self.col_data,
            row_data=row_data,
            contrast_data=contrast_data,
            metadata=group_metadata,
        )


def study_dataset(
    subjects: pd.DataFrame,
    *,
    id: str = "subject_id",  # noqa: A002 - public API mirrors donor spelling.
    dataset_col: str = "dataset",
    space: str | None = None,
    mask_strategy: MaskStrategy = "subject_specific",
    metadata: Mapping[str, object] | None = None,
) -> StudyDataset:
    """Construct the canonical study-level dataset container."""
    return StudyDataset(
        subjects=subjects,
        id=id,
        dataset_col=dataset_col,
        space=space,
        mask_strategy=mask_strategy,
        metadata={} if metadata is None else metadata,
    )


def fmri_group(
    subjects: pd.DataFrame,
    *,
    id: str,  # noqa: A002 - public API mirrors donor spelling.
    dataset_col: str = "dataset",
    space: str | None = None,
    mask_strategy: MaskStrategy = "subject_specific",
    metadata: Mapping[str, object] | None = None,
) -> StudyDataset:
    """R-compatible spelling for constructing a study/group subject table."""
    return study_dataset(
        subjects,
        id=id,
        dataset_col=dataset_col,
        space=space,
        mask_strategy=mask_strategy,
        metadata=metadata,
    )


def study_to_group(
    study: StudyDataset,
    contrast_results: Mapping[str, object] | None = None,
    *,
    sample_labels: Sequence[object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> GroupDataset:
    """Convert a :class:`StudyDataset` plus contrast results to ``GroupDataset``."""
    if contrast_results is None:
        method = getattr(study, "to_group", None)
        if callable(method):
            return cast("GroupDataset", method())
        raise TypeError("contrast_results is required for StudyDataset conversion")
    return study.to_group_dataset(
        contrast_results, sample_labels=sample_labels, metadata=metadata
    )


__all__ = ["StudyDataset", "fmri_group", "study_dataset", "study_to_group"]
