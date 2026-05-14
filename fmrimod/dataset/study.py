"""Study-level dataset containers that feed native group analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .errors import FmriDatasetError
from .fmri_dataset import FmriDataset

if TYPE_CHECKING:
    from fmrimod.group import GroupDataset

MaskStrategy = Literal["subject_specific", "intersect", "union"]


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
        return matches.iloc[0][self.dataset_col]

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

        assays: dict[str, NDArray[np.float64]] = {}
        n_samples: int | None = None
        for subject_index, subject in enumerate(subjects):
            for contrast_index, contrast in enumerate(contrasts):
                result = lookup[(subject, contrast)]
                estimate = _as_1d_float_array(
                    result.estimate,
                    assay="estimate",
                    subject=subject,
                    contrast=contrast,
                )
                if n_samples is None:
                    n_samples = int(estimate.size)
                    shape = (n_samples, len(subjects), len(contrasts))
                    assays["beta"] = np.full(shape, np.nan, dtype=np.float64)
                elif estimate.size != n_samples:
                    raise FmriDatasetError(
                        "all contrast estimates must have the same sample length"
                    )
                assays["beta"][:, subject_index, contrast_index] = estimate

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
                    if arr.size != n_samples:
                        raise FmriDatasetError(
                            f"{assay_name} arrays must match estimate sample length"
                        )
                    if assay_name not in assays:
                        assays[assay_name] = np.full(
                            (n_samples, len(subjects), len(contrasts)),
                            np.nan,
                            dtype=np.float64,
                        )
                    assays[assay_name][:, subject_index, contrast_index] = arr

        if n_samples is None:
            raise FmriDatasetError(
                "contrast_results must contain at least one estimate"
            )

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
                str(lookup[(subjects[0], contrast)].stat_type) for contrast in contrasts
            ]

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
            row_data=pd.DataFrame(index=pd.Index(labels, name="sample")),
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
    contrast_results: Mapping[str, object],
    *,
    sample_labels: Sequence[object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> GroupDataset:
    """Convert a :class:`StudyDataset` plus contrast results to ``GroupDataset``."""
    return study.to_group_dataset(
        contrast_results, sample_labels=sample_labels, metadata=metadata
    )


__all__ = ["StudyDataset", "fmri_group", "study_dataset", "study_to_group"]
