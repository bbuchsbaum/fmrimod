"""Core group-analysis dataset contract."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dtypes import as_group_float_array
from .errors import (
    AdapterContractError,
    GroupRegistryError,
    UnsupportedGroupFeatureError,
)
from .registry import adapter_registry
from .space import GroupSpace, SampleLabelSpace, VoxelSpace


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

    This adapter reuses the existing ``fmrimod.dataset.group_data`` loaders and
    materializes their supported group-level formats into the native
    ``sample x subject x contrast`` contract.
    """
    data_format = getattr(data, "format", None)
    try:
        adapter = adapter_registry.get(str(data_format))
    except GroupRegistryError as exc:
        raise UnsupportedGroupFeatureError(
            "group_dataset_from_group_data supports only registered GroupData "
            f"formats: {', '.join(adapter_registry.list_names())}"
        ) from exc
    return cast(GroupDataset, adapter(data))


def _group_dataset_from_csv_group_data(data: Any) -> GroupDataset:
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


def _single_contrast(data: Any) -> tuple[str, ...]:
    contrast = data.data.get("contrast")
    return ("c1",) if contrast is None else (str(contrast),)


def _sample_labels(n_samples: int) -> tuple[str, ...]:
    return tuple(f"sample{i + 1}" for i in range(n_samples))


def _group_dataset_from_h5_group_data(data: Any) -> GroupDataset:
    from fmrimod.dataset.compat import read_h5_full

    stats = tuple(str(stat) for stat in data.data.get("stat", ("beta", "se")))
    raw = read_h5_full(data, stat=stats)
    if raw.ndim != 3:
        raise AdapterContractError("H5 GroupData reader must return voxels x subjects x stats")
    samples = _sample_labels(int(raw.shape[0]))
    assays = {
        stat: np.asarray(raw[:, :, stat_idx], dtype=np.float64)[:, :, np.newaxis]
        for stat_idx, stat in enumerate(stats)
    }

    return GroupDataset(
        assays=assays,
        space=SampleLabelSpace(samples),
        subjects=tuple(str(x) for x in data.subjects),
        contrasts=_single_contrast(data),
        col_data=data.covariates,
        row_data=pd.DataFrame(index=pd.Index(samples, name="sample")),
        contrast_data=pd.DataFrame(index=pd.Index(_single_contrast(data), name="contrast")),
        metadata={
            "source_format": "h5",
            "paths": tuple(str(path) for path in data.data.get("paths", ())),
        },
    )


def _first_nifti_path(data: Any) -> str | None:
    for key in ("beta_paths", "se_paths", "var_paths", "t_paths"):
        paths = data.data.get(key)
        if paths:
            return str(paths[0])
    return None


def _nifti_space(data: Any, *, n_samples: int) -> GroupSpace:
    try:
        nib = cast(Any, importlib.import_module("nibabel"))
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        raise UnsupportedGroupFeatureError(
            "NIfTI GroupData materialization requires optional dependency 'nibabel'"
        ) from exc

    path = str(data.data["mask"]) if data.data.get("mask") is not None else _first_nifti_path(data)
    if path is None:
        return SampleLabelSpace(_sample_labels(n_samples))

    img = nib.load(path)
    shape = tuple(int(x) for x in img.shape[:3])
    affine = np.asarray(img.affine, dtype=np.float64)
    template_id = data.data.get("target_space")
    if data.data.get("mask") is not None:
        mask = np.asarray(img.get_fdata()) > 0
        mask_idx = np.flatnonzero(mask.ravel()).astype(np.intp)
        if mask_idx.size != n_samples:
            raise AdapterContractError(
                "NIfTI mask sample count must match materialized assay samples"
            )
        return VoxelSpace(
            shape=shape,
            affine=affine,
            mask_idx=mask_idx,
            storage="packed",
            template_id=template_id,
        )

    if int(np.prod(shape)) != n_samples:
        return SampleLabelSpace(_sample_labels(n_samples))
    return VoxelSpace(shape=shape, affine=affine, template_id=template_id)


def _group_dataset_from_nifti_group_data(data: Any) -> GroupDataset:
    from fmrimod.dataset.compat import read_nifti_full

    try:
        payload = read_nifti_full(data)
    except ImportError as exc:  # pragma: no cover - optional dependency behavior
        raise UnsupportedGroupFeatureError(
            "NIfTI GroupData materialization requires the existing NIfTI reader dependencies"
        ) from exc
    subjects = tuple(str(x) for x in data.subjects)
    n_subjects = len(subjects)
    assays: dict[str, NDArray[np.float64]] = {}
    n_samples: int | None = None
    for name, values in payload.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[0] != n_subjects:
                raise AdapterContractError(
                    f"NIfTI assay '{name}' rows must match number of subjects"
                )
            assays[name] = arr.T[:, :, np.newaxis]
            n_samples = int(arr.shape[1])
        elif arr.ndim == 1 and arr.shape[0] == n_subjects and n_samples is not None:
            assays[name] = np.broadcast_to(
                arr.reshape(1, n_subjects, 1),
                (n_samples, n_subjects, 1),
            ).copy()
    if not assays or n_samples is None:
        raise AdapterContractError("NIfTI GroupData did not materialize any assays")

    contrasts = _single_contrast(data)
    return GroupDataset(
        assays=assays,
        space=_nifti_space(data, n_samples=n_samples),
        subjects=subjects,
        contrasts=contrasts,
        col_data=data.covariates,
        row_data=None,
        contrast_data=pd.DataFrame(index=pd.Index(contrasts, name="contrast")),
        metadata={"source_format": "nifti"},
    )


def _maybe_call(value: Any) -> Any:
    return value() if callable(value) else value


def _as_lm_stat_matrix(value: Any, *, stat: str) -> NDArray[np.float64]:
    arr = np.asarray(_maybe_call(value), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise AdapterContractError(
            f"fmrilm statistic '{stat}' must be a 1-D or 2-D array"
        )
    return arr


def _canonical_lm_stat(stat: str) -> str:
    aliases = {
        "betas": "beta",
        "estimate": "beta",
        "effect": "beta",
        "stderr": "se",
        "std_error": "se",
        "tstat": "t",
        "stat": "t",
        "pvalue": "p",
        "p_value": "p",
    }
    return aliases.get(stat, stat)


def _lm_contrast_stat_matrix(
    lm: Any,
    stat: str,
    contrast_name: str,
) -> NDArray[np.float64]:
    contrasts = getattr(lm, "contrasts", None)
    if not isinstance(contrasts, Mapping) or contrast_name not in contrasts:
        raise AdapterContractError(
            f"fmrilm object does not contain stored contrast '{contrast_name}'"
        )
    contrast = contrasts[contrast_name]
    if stat == "beta":
        return _as_lm_stat_matrix(contrast.estimate, stat=stat)
    if stat == "se":
        se = getattr(contrast, "se", None)
        if se is None:
            raise AdapterContractError("stored fmrilm contrast does not expose se")
        return _as_lm_stat_matrix(se, stat=stat)
    if stat in ("t", "z"):
        return _as_lm_stat_matrix(contrast.stat, stat=stat)
    if stat == "p":
        return _as_lm_stat_matrix(contrast.p_value, stat=stat)
    if stat == "var":
        se = getattr(contrast, "se", None)
        if se is None:
            raise AdapterContractError("stored fmrilm contrast does not expose se")
        return _as_lm_stat_matrix(se, stat=stat) ** 2
    raise AdapterContractError(f"fmrilm contrast statistic '{stat}' is not supported")


def _lm_stat_matrix(
    lm: Any,
    stat: str,
    contrast_name: str | None,
) -> NDArray[np.float64]:
    if contrast_name is not None:
        return _lm_contrast_stat_matrix(lm, stat, contrast_name)
    if stat == "beta":
        return _as_lm_stat_matrix(lm.betas, stat=stat)
    if stat == "se":
        return _as_lm_stat_matrix(lm.se, stat=stat)
    if stat == "var":
        return _as_lm_stat_matrix(lm.se, stat=stat) ** 2
    if stat in ("t", "z"):
        return _as_lm_stat_matrix(lm.tstat, stat=stat)
    if stat == "p" and hasattr(lm, "p"):
        return _as_lm_stat_matrix(lm.p, stat=stat)
    raise AdapterContractError(f"fmrilm statistic '{stat}' is not supported")


def _lm_contrast_labels(lm: Any, *, n_contrasts: int) -> tuple[str, ...]:
    raw = getattr(lm, "coef_names", None)
    if raw is None:
        raw = getattr(lm, "coefficient_names", None)
    if raw is not None:
        labels = tuple(str(x) for x in _maybe_call(raw))
        if len(labels) == n_contrasts:
            return labels
    return tuple(f"c{i + 1}" for i in range(n_contrasts))


def _group_dataset_from_fmrilm_group_data(data: Any) -> GroupDataset:
    lm_list = list(data.data["lm_list"])
    if not lm_list:
        raise AdapterContractError("fmrilm GroupData requires at least one model")
    requested_stats = tuple(
        _canonical_lm_stat(str(stat))
        for stat in data.data.get("stat", ("beta", "se"))
    )
    contrast_name = data.data.get("contrast")
    if contrast_name is not None:
        contrast_name = str(contrast_name)

    first_matrix = _lm_stat_matrix(lm_list[0], requested_stats[0], contrast_name)
    n_contrasts, n_samples = (
        (1, first_matrix.shape[1]) if contrast_name else first_matrix.shape
    )
    contrasts = (
        (contrast_name,)
        if contrast_name is not None
        else _lm_contrast_labels(lm_list[0], n_contrasts=n_contrasts)
    )
    shape = (n_samples, len(lm_list), n_contrasts)
    assays: dict[str, NDArray[np.float64]] = {
        stat: np.empty(shape, dtype=np.float64) for stat in requested_stats
    }
    for subject_idx, lm in enumerate(lm_list):
        for stat in requested_stats:
            matrix = _lm_stat_matrix(lm, stat, contrast_name)
            if contrast_name is not None and matrix.shape[0] != 1:
                raise AdapterContractError("stored fmrilm contrast statistic must be 1-D")
            if matrix.shape != (n_contrasts, n_samples):
                raise AdapterContractError(
                    "all fmrilm statistics must have matching coefficient x sample shape"
                )
            assays[stat][:, subject_idx, :] = matrix.T

    samples = _sample_labels(n_samples)
    return GroupDataset(
        assays=assays,
        space=SampleLabelSpace(samples),
        subjects=tuple(str(x) for x in data.subjects),
        contrasts=contrasts,
        col_data=data.covariates,
        row_data=pd.DataFrame(index=pd.Index(samples, name="sample")),
        contrast_data=pd.DataFrame(index=pd.Index(contrasts, name="contrast")),
        metadata={"source_format": "fmrilm"},
    )


def register_core_adapters(*, overwrite: bool = True) -> None:
    """Register built-in GroupData materializers."""
    for name, function, description in (
        ("csv", _group_dataset_from_csv_group_data, "CSV GroupData materializer"),
        ("h5", _group_dataset_from_h5_group_data, "HDF5 GroupData materializer"),
        ("nifti", _group_dataset_from_nifti_group_data, "NIfTI GroupData materializer"),
        (
            "fmrilm",
            _group_dataset_from_fmrilm_group_data,
            "fmri_lm GroupData materializer",
        ),
    ):
        adapter_registry.register(
            name,
            function,
            description=description,
            overwrite=overwrite,
        )


register_core_adapters()
