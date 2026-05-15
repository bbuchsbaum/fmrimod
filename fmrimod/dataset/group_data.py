"""Group-level dataset construction helpers.

Provides a minimal parity-oriented API for composing group-level analysis inputs.

This mirrors the R `fmrireg` workflow:

- `group_data(..., format = "auto")` dispatches by input type and extensions.
- `format` may be `"h5"`, `"nifti"`, `"csv"`, or `"fmrilm"`.
- Constructors validate shape/length consistency and basic file/covariate inputs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ALLOWED_FORMATS = ("h5", "nifti", "csv", "fmrilm")


@dataclass(frozen=True)
class GroupData:
    """Container for group-level analysis inputs."""

    format: str
    subjects: List[str]
    data: Dict[str, object] = field(default_factory=dict)
    covariates: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if self.format not in ALLOWED_FORMATS:
            raise ValueError(
                "format must be one of: " + ", ".join(ALLOWED_FORMATS)
            )
        if len(self.subjects) == 0:
            raise ValueError("'subjects' must be non-empty")

    @property
    def n_subjects(self) -> int:
        """Number of subjects represented."""
        return len(self.subjects)


def _coerce_path_list(values: object, *, name: str) -> List[str]:
    if isinstance(values, (str, os.PathLike)):
        paths = [Path(values)]
    elif isinstance(values, Sequence):
        paths = [Path(v) for v in values]
    else:
        raise TypeError(f"'{name}' must be a string path or a sequence of paths")

    if len(paths) == 0:
        raise ValueError(f"'{name}' must contain at least one path")

    return [str(p) for p in paths]


def _coerce_subjects(subjects: Optional[Sequence[str]], n_subjects: int) -> List[str]:
    if subjects is None:
        return [f"subject_{idx + 1}" for idx in range(n_subjects)]

    if len(subjects) != n_subjects:
        raise ValueError(
            f"'subjects' length ({len(subjects)}) does not match number of subjects ({n_subjects})"
        )
    return list(subjects)


def _validate_files(paths: Sequence[str], *, name: str, validate: bool) -> None:
    if not validate:
        return
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {name}: " + ", ".join(missing))


def _nifti_shape(path: str) -> tuple[int, int, int]:
    """Read the first three spatial dimensions from a NIfTI file."""
    try:
        from neuroim import read_image
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        raise ImportError(
            "nifti validation requires optional dependency 'neuroim'. "
            "Install neuroim-python to validate NIfTI group data."
        ) from exc

    data = np.asarray(read_image(path, type="vol").as_array())
    if data.ndim < 3:
        raise ValueError(f"Expected at least 3D NIfTI file, got shape {data.shape}")

    return tuple(int(v) for v in data.shape[:3])  # type: ignore[return-value]


def _infer_subject_ids(paths: Sequence[str]) -> List[str]:
    inferred: List[str] = []
    for path in paths:
        base = Path(path).name
        # NIfTI naming convention: extract BIDS subject if available.
        stem = base[:-7] if base.lower().endswith(".nii.gz") else Path(base).stem
        match = re.search(r"sub-([^/_]+)", stem)
        if match:
            inferred.append(f"sub-{match.group(1)}")
        else:
            inferred.append(stem)
    return inferred


def _validate_covariates(
    covariates: Optional[pd.DataFrame],
    n_subjects: int,
) -> Optional[pd.DataFrame]:
    if covariates is None:
        return None
    if not isinstance(covariates, pd.DataFrame):
        raise TypeError("'covariates' must be a pandas DataFrame")
    if len(covariates) != n_subjects:
        raise ValueError(
            f"'covariates' rows ({len(covariates)}) must match subjects ({n_subjects})"
        )
    return covariates


def _looks_like_fmri_lm(obj: object) -> bool:
    required_attrs = ("betas", "tstat", "se", "n_voxels")
    return all(hasattr(obj, attr) for attr in required_attrs)


def detect_group_data_format(data: object) -> str:
    """Best-effort format detection for group_data inputs."""
    def _as_path_string(value: object) -> str:
        return str(Path(value))  # type: ignore[arg-type]

    if isinstance(data, (str, os.PathLike)):
        data = [data]

    if isinstance(data, Sequence) and not isinstance(data, (bytes, bytearray)):
        first = data[0] if len(data) else None
        if all(isinstance(item, (str, os.PathLike)) for item in data):
            first_path = _as_path_string(first)
            lower = first_path.lower()
            if lower.endswith((".h5", ".hdf5")):
                return "h5"
            if lower.endswith(".csv") and len(data) == 1:
                return "csv"
            if lower.endswith((".nii", ".nii.gz")):
                return "nifti"
        if _looks_like_fmri_lm(first):
            if not all(_looks_like_fmri_lm(item) for item in data):
                raise ValueError(
                    "Could not auto-detect format for mixed object list"
                )
            return "fmrilm"

    if isinstance(data, pd.DataFrame):
        return "csv"
    if isinstance(data, Mapping):
        names = set(data.keys())
        if ("beta" in names and ("se" in names or "var" in names)) or ("t" in names):
            return "nifti"
        if {"subjects", "lm_list"}.issubset(names):
            return "fmrilm"
    raise ValueError("Could not auto-detect group_data format. Please set format explicitly.")


def group_data_from_h5(
    paths: Sequence[str] | os.PathLike | str,
    subjects: Optional[Sequence[str]] = None,
    covariates: Optional[pd.DataFrame] = None,
    mask: Optional[str] = None,
    contrast: Optional[str] = None,
    stat: Sequence[str] = ("beta", "se"),
    validate: bool = True,
) -> GroupData:
    paths_list = _coerce_path_list(paths, name="paths")
    _validate_files(paths_list, name="h5 paths", validate=validate)

    if subjects is None:
        subjects = _infer_subject_ids(paths_list)
    else:
        subjects = _coerce_subjects(subjects, len(paths_list))
    covariates = _validate_covariates(covariates, len(paths_list))
    if not list(stat):
        raise ValueError("'stat' must be a non-empty list-like")

    return GroupData(
        format="h5",
        subjects=subjects,
        data={
            "paths": paths_list,
            "mask": str(mask) if mask is not None else None,
            "contrast": contrast,
            "stat": list(stat),
        },
        covariates=covariates,
    )


def group_data_from_nifti(
    beta_paths: Optional[Sequence[str] | str] = None,
    se_paths: Optional[Sequence[str] | str] = None,
    var_paths: Optional[Sequence[str] | str] = None,
    t_paths: Optional[Sequence[str] | str] = None,
    df: Optional[Sequence[int] | int] = None,
    subjects: Optional[Sequence[str]] = None,
    covariates: Optional[pd.DataFrame] = None,
    mask: Optional[str] = None,
    target_space: Optional[str] = None,
    validate: bool = True,
) -> GroupData:
    has_beta = beta_paths is not None
    has_t = t_paths is not None
    if not has_beta and not has_t:
        raise ValueError("Must provide either beta_paths or t_paths")
    if has_beta and has_t:
        raise ValueError("Provide either beta_paths or t_paths, not both")

    has_se = se_paths is not None
    has_var = var_paths is not None
    if has_beta and not (has_se or has_var):
        raise ValueError("When using beta_paths, provide se_paths or var_paths")
    if has_se and has_var:
        raise ValueError("Provide either se_paths or var_paths, not both")
    if has_t and df is None:
        raise ValueError("When using t_paths, df is required")

    if has_beta:
        beta_list = _coerce_path_list(beta_paths, name="beta_paths")
        se_list = _coerce_path_list(se_paths, name="se_paths") if has_se else None
        var_list = _coerce_path_list(var_paths, name="var_paths") if has_var else None
        t_list = None
    else:
        beta_list = None
        se_list = None
        var_list = None
        t_list = _coerce_path_list(t_paths, name="t_paths")

    primary = beta_list if beta_list is not None else t_list
    n_subjects = len(primary)

    if se_list is not None and len(se_list) != n_subjects:
        raise ValueError("Length of se_paths must match beta_paths")
    if var_list is not None and len(var_list) != n_subjects:
        raise ValueError("Length of var_paths must match beta_paths")
    if t_list is not None and len(t_list) != n_subjects:
        raise ValueError("Length of t_paths must match beta_paths/t_paths")

    validate_paths = list(primary)
    if se_list is not None:
        validate_paths.extend(se_list)
    if var_list is not None:
        validate_paths.extend(var_list)
    if t_list is not None:
        validate_paths.extend(t_list)
    _validate_files(validate_paths, name="nifti paths", validate=validate)

    if subjects is None:
        subjects = _infer_subject_ids(primary)
    else:
        subjects = _coerce_subjects(subjects, n_subjects)

    if validate:
        reference_shape = _nifti_shape(validate_paths[0])
        for path in validate_paths[1:]:
            shape = _nifti_shape(path)
            if shape != reference_shape:
                raise ValueError(
                    f"NIfTI dimensions mismatch for {path}: expected {reference_shape}, "
                    f"got {shape}"
                )

        if mask is not None:
            mask_shape = _nifti_shape(str(mask))
            if mask_shape != reference_shape:
                raise ValueError(
                    f"Mask dimensions {mask_shape} do not match NIfTI dimensions {reference_shape}"
                )

    covariates = _validate_covariates(covariates, n_subjects)

    if df is None:
        df_values = None
    elif isinstance(df, Integral):
        df_values = [int(df)] * n_subjects
    else:
        df_values = list(df)
        if len(df_values) != n_subjects:
            raise ValueError("Length of df must be 1 or equal to number of subjects")

    return GroupData(
        format="nifti",
        subjects=subjects,
        data={
            "beta_paths": beta_list,
            "se_paths": se_list,
            "var_paths": var_list,
            "t_paths": t_list,
            "df": df_values,
            "mask": str(mask) if mask is not None else None,
            "target_space": str(target_space) if target_space is not None else None,
        },
        covariates=covariates,
    )


def group_data_from_csv(
    data: str | pd.DataFrame | os.PathLike,
    effect_cols: Mapping[str, str] | Sequence[str],
    subject_col: str = "subject",
    roi_col: Optional[str] = None,
    contrast_col: Optional[str] = None,
    covariate_cols: Optional[Sequence[str]] = None,
    wide_format: bool = False,
    subjects: Optional[Sequence[str]] = None,
    covariates: Optional[pd.DataFrame] = None,
) -> GroupData:
    if isinstance(data, (str, os.PathLike)):
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {data_path}")
        df = pd.read_csv(data_path)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("'data' must be a path or pandas DataFrame")

    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found")
    if wide_format:
        raise NotImplementedError(
            "wide_format=True is not yet supported; provide long-format CSV/data"
        )

    if not isinstance(effect_cols, Mapping):
        raise TypeError("'effect_cols' must be a mapping of effect names to column names")
    validated_effect_cols = dict(effect_cols)
    for name, column in validated_effect_cols.items():
        if column not in df.columns:
            prefix_matches = [c for c in df.columns if c.startswith(str(column))]
            if not prefix_matches:
                raise ValueError(f"Effect column for '{name}' not found: {column}")

    if "beta" in validated_effect_cols and not (
        "se" in validated_effect_cols or "var" in validated_effect_cols
    ):
        raise ValueError("Effect columns must include se or var when beta is provided")

    if roi_col is not None and roi_col not in df.columns:
        raise ValueError(f"ROI column '{roi_col}' not found")
    if contrast_col is not None and contrast_col not in df.columns:
        raise ValueError(f"Contrast column '{contrast_col}' not found")

    if covariate_cols is not None:
        missing = [col for col in covariate_cols if col not in df.columns]
        if missing:
            raise ValueError(
                "Missing covariate columns: " + ", ".join(missing)
            )

    subject_values = list(df[subject_col])
    subjects = (
        list(subjects)
        if subjects is not None
        else list(dict.fromkeys(subject_values))
    )
    if subjects and len(subject_values) == 0:
        raise ValueError("No subjects found in CSV data")

    covariate_frame: Optional[pd.DataFrame]
    if covariate_cols is not None:
        subjects_sorted = pd.Index(subjects)
        uniq_rows = df.drop_duplicates(subject_col).set_index(subject_col)
        missing = subjects_sorted.difference(uniq_rows.index)
        if len(missing):
            raise ValueError(
                "Covariates are missing for subjects: " + ", ".join(missing.astype(str))
            )
        covariate_frame = uniq_rows.loc[subjects_sorted, covariate_cols]
    else:
        covariate_frame = None
    if covariates is not None:
        if covariate_frame is not None:
            raise ValueError("Cannot pass both covariate_cols and covariates")
        covariate_frame = _validate_covariates(covariates, len(subjects))

    return GroupData(
        format="csv",
        subjects=subjects,
        data={
            "data": df,
            "effect_cols": validated_effect_cols,
            "subject_col": subject_col,
            "roi_col": roi_col,
            "contrast_col": contrast_col,
            "covariate_cols": list(covariate_cols) if covariate_cols else None,
            "wide_format": wide_format,
        },
        covariates=covariate_frame,
    )


def group_data_from_fmrilm(
    lm_list: Sequence[object],
    contrast: Optional[str] = None,
    stat: Sequence[str] = ("beta", "se"),
    subjects: Optional[Sequence[str]] = None,
    covariates: Optional[pd.DataFrame] = None,
) -> GroupData:
    if not isinstance(lm_list, Sequence):
        raise TypeError("'lm_list' must be a sequence of model fits")
    if len(lm_list) == 0:
        raise ValueError("'lm_list' must not be empty")
    if not all(_looks_like_fmri_lm(obj) for obj in lm_list):
        raise TypeError("All elements of lm_list must behave like fmri_lm objects")

    n_subjects = len(lm_list)
    subjects = _coerce_subjects(subjects, n_subjects)
    covariates = _validate_covariates(covariates, n_subjects)

    if not list(stat):
        raise ValueError("'stat' must be a non-empty list-like")

    return GroupData(
        format="fmrilm",
        subjects=subjects,
        data={
            "lm_list": list(lm_list),
            "contrast": contrast,
            "stat": list(stat),
        },
        covariates=covariates,
    )


def group_data(
    data: object,
    format: str = "auto",
    **kwargs: object,
) -> GroupData:
    """Create a ``GroupData`` object from various supported inputs."""
    if format not in ("auto", *ALLOWED_FORMATS):
        raise ValueError(
            f"Unsupported format '{format}'. Use 'auto', {', '.join(ALLOWED_FORMATS)}"
        )

    if format == "auto":
        format = detect_group_data_format(data)

    if format == "h5":
        return group_data_from_h5(data, **kwargs)
    if format == "nifti":
        if isinstance(data, Mapping):
            args = {
                "beta_paths": data.get("beta"),
                "se_paths": data.get("se"),
                "var_paths": data.get("var"),
                "t_paths": data.get("t"),
                "df": data.get("df"),
            }
            args = {k: v for k, v in args.items() if v is not None}
            args.update(kwargs)
            return group_data_from_nifti(**args)
        return group_data_from_nifti(data, **kwargs)
    if format == "csv":
        return group_data_from_csv(data, **kwargs)
    if format == "fmrilm":
        return group_data_from_fmrilm(data, **kwargs)

    raise ValueError(f"Unsupported format '{format}'")
