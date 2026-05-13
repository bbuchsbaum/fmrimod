"""Core group-analysis dataset contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dtypes import as_group_float_array
from .errors import AdapterContractError, UnsupportedGroupFeatureError
from .space import GroupSpace, SampleLabelSpace


def _coerce_axis(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    out = tuple(str(value) for value in values)
    if not out:
        raise AdapterContractError(f"{name} must be non-empty")
    if any(value == "" for value in out):
        raise AdapterContractError(f"{name} must not contain empty labels")
    if len(set(out)) != len(out):
        raise AdapterContractError(f"{name} must be unique")
    return out


def _validate_axis_frame(
    frame: pd.DataFrame | None,
    *,
    expected_rows: int,
    name: str,
) -> pd.DataFrame | None:
    if frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise AdapterContractError(f"{name} must be a pandas DataFrame")
    if len(frame) != expected_rows:
        raise AdapterContractError(
            f"{name} rows ({len(frame)}) must match expected axis length ({expected_rows})"
        )
    return frame.copy()


@dataclass(frozen=True)
class GroupDataset:
    """Materialized group-analysis data.

    Assays are stored as ``sample x subject x contrast`` float64 arrays.
    """

    assays: Mapping[str, Any]
    space: GroupSpace
    subjects: Sequence[Any]
    contrasts: Sequence[Any]
    col_data: pd.DataFrame | None = None
    row_data: pd.DataFrame | None = None
    contrast_data: pd.DataFrame | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.assays, Mapping) or not self.assays:
            raise AdapterContractError("assays must be a non-empty mapping")

        coerced: dict[str, NDArray[np.float64]] = {}
        shape: tuple[int, int, int] | None = None
        for name, value in self.assays.items():
            if not isinstance(name, str) or not name:
                raise AdapterContractError("assay names must be non-empty strings")
            arr = as_group_float_array(value, copy=True)
            if arr.ndim != 3:
                raise AdapterContractError(
                    f"assay '{name}' must be a 3-D sample x subject x contrast array"
                )
            if shape is None:
                shape = tuple(int(x) for x in arr.shape)  # type: ignore[assignment]
            elif tuple(arr.shape) != shape:
                raise AdapterContractError("all assays must have identical shapes")
            arr.setflags(write=False)
            coerced[name] = arr

        assert shape is not None
        subjects = _coerce_axis(self.subjects, name="subjects")
        contrasts = _coerce_axis(self.contrasts, name="contrasts")
        self.space.validate()

        if self.space.n_samples != shape[0]:
            raise AdapterContractError(
                f"space samples ({self.space.n_samples}) must match assay samples ({shape[0]})"
            )
        if len(subjects) != shape[1]:
            raise AdapterContractError(
                f"subjects length ({len(subjects)}) must match assay subject axis ({shape[1]})"
            )
        if len(contrasts) != shape[2]:
            raise AdapterContractError(
                f"contrasts length ({len(contrasts)}) must match assay contrast axis ({shape[2]})"
            )

        object.__setattr__(self, "assays", MappingProxyType(coerced))
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "contrasts", contrasts)
        object.__setattr__(
            self,
            "col_data",
            _validate_axis_frame(self.col_data, expected_rows=shape[1], name="col_data"),
        )
        object.__setattr__(
            self,
            "row_data",
            _validate_axis_frame(self.row_data, expected_rows=shape[0], name="row_data"),
        )
        object.__setattr__(
            self,
            "contrast_data",
            _validate_axis_frame(
                self.contrast_data,
                expected_rows=shape[2],
                name="contrast_data",
            ),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``sample x subject x contrast`` shape."""
        first = next(iter(self.assays.values()))
        return tuple(int(x) for x in first.shape)  # type: ignore[return-value]

    @property
    def n_samples(self) -> int:
        return self.shape[0]

    @property
    def n_subjects(self) -> int:
        return self.shape[1]

    @property
    def n_contrasts(self) -> int:
        return self.shape[2]

    def assay(self, name: str = "beta") -> NDArray[np.float64]:
        """Return one assay by name."""
        try:
            return cast(NDArray[np.float64], self.assays[name])
        except KeyError as exc:
            available = ", ".join(sorted(self.assays))
            raise AdapterContractError(
                f"assay '{name}' is not present. Available: {available}"
            ) from exc

    def assay_names(self) -> list[str]:
        """Return assay names in deterministic order."""
        return sorted(self.assays)

    def with_assays(
        self,
        assays: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> GroupDataset:
        """Return a dataset with replaced assays and preserved axes."""
        md = dict(self.metadata)
        if metadata:
            md.update(metadata)
        return GroupDataset(
            assays=assays,
            space=self.space,
            subjects=self.subjects,
            contrasts=self.contrasts,
            col_data=self.col_data,
            row_data=self.row_data,
            contrast_data=self.contrast_data,
            metadata=md,
        )


def group_dataset(
    assays: Mapping[str, Any],
    *,
    space: GroupSpace,
    subjects: Sequence[Any],
    contrasts: Sequence[Any],
    col_data: pd.DataFrame | None = None,
    row_data: pd.DataFrame | None = None,
    contrast_data: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GroupDataset:
    """Construct a :class:`GroupDataset` from materialized assay arrays."""
    return GroupDataset(
        assays=assays,
        space=space,
        subjects=subjects,
        contrasts=contrasts,
        col_data=col_data,
        row_data=row_data,
        contrast_data=contrast_data,
        metadata={} if metadata is None else metadata,
    )


def group_dataset_from_group_data(data: Any) -> GroupDataset:
    """Convert existing ``fmrimod.dataset.GroupData`` into ``GroupDataset``.

    The first implementation supports the existing CSV/long-table loader and
    preserves its axis conventions. Other source formats stay behind their
    existing loaders until native HDF5/NIfTI adapters are added.
    """
    if getattr(data, "format", None) != "csv":
        raise UnsupportedGroupFeatureError(
            "group_dataset_from_group_data currently supports only csv GroupData"
        )

    payload = data.data
    frame = payload["data"].copy()
    effect_cols = dict(payload["effect_cols"])
    subject_col = payload["subject_col"]
    sample_col = payload.get("roi_col") or "sample"
    contrast_col = payload.get("contrast_col") or "contrast"

    if sample_col not in frame.columns:
        frame[sample_col] = "sample1"
    if contrast_col not in frame.columns:
        frame[contrast_col] = "c1"

    subjects = tuple(str(x) for x in data.subjects)
    samples = tuple(str(x) for x in dict.fromkeys(frame[sample_col]))
    contrasts = tuple(str(x) for x in dict.fromkeys(frame[contrast_col]))
    subject_lookup = {label: i for i, label in enumerate(subjects)}
    sample_lookup = {label: i for i, label in enumerate(samples)}
    contrast_lookup = {label: i for i, label in enumerate(contrasts)}
    shape = (len(samples), len(subjects), len(contrasts))

    assays: dict[str, NDArray[np.float64]] = {
        name: np.full(shape, np.nan, dtype=np.float64) for name in effect_cols
    }
    seen: set[tuple[str, str, str, str]] = set()
    for _, row in frame.iterrows():
        subject = str(row[subject_col])
        sample = str(row[sample_col])
        contrast = str(row[contrast_col])
        if subject not in subject_lookup:
            raise AdapterContractError(f"CSV row contains unknown subject: {subject}")
        key_base = (sample, subject, contrast)
        idx = (
            sample_lookup[sample],
            subject_lookup[subject],
            contrast_lookup[contrast],
        )
        for assay_name, column in effect_cols.items():
            key = (*key_base, assay_name)
            if key in seen:
                raise AdapterContractError(
                    "duplicate CSV rows for sample/subject/contrast/assay"
                )
            seen.add(key)
            assays[assay_name][idx] = float(row[column])

    return GroupDataset(
        assays=assays,
        space=SampleLabelSpace(samples),
        subjects=subjects,
        contrasts=contrasts,
        col_data=data.covariates,
        row_data=pd.DataFrame(index=pd.Index(samples, name="sample")),
        contrast_data=pd.DataFrame(index=pd.Index(contrasts, name="contrast")),
        metadata={"source_format": "csv"},
    )
