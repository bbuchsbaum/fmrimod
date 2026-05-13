"""fmrireg-style dataset, IO, and benchmark compatibility helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ..hrf.core import HRF, HrfParamValue
from ..sampling import SamplingFrame
from .fmri_dataset import FmriDataset
from .group_data import GroupData


# An HRF specifier mirrors what the registry / `gen_hrf` accept.
HrfSpec = Union[HRF, Callable[..., NDArray[np.float64]], str]
"""HRF specifier accepted by the dataset-compat builders: typed HRF
instance, registry key string, or callable returning HRF samples."""

_BASIS_REGISTRY: Dict[str, Callable[..., object]] = {}


def fmri_mem_dataset(
    data: Union[ArrayLike, Sequence[ArrayLike]],
    tr: Union[float, Sequence[float]],
    run_length: Optional[Union[int, Sequence[int]]] = None,
    *,
    event_table: Optional[pd.DataFrame] = None,
    mask: Optional[NDArray[np.bool_]] = None,
) -> FmriDataset:
    """Construct an in-memory fMRI dataset from matrix data."""
    from .constructors import matrix_dataset

    return matrix_dataset(
        data,
        tr,
        run_length=run_length,
        event_table=event_table,
        mask=mask,
    )


@dataclass
class LatentDataset:
    """Small Python representation of a latent-component dataset."""

    scores: NDArray[np.float64]
    loadings: Optional[NDArray[np.float64]]
    sampling_frame: SamplingFrame
    event_table: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64)
        if self.scores.ndim != 2:
            raise ValueError("scores must be a 2-D matrix")
        if self.loadings is not None:
            self.loadings = np.asarray(self.loadings, dtype=np.float64)
            if self.loadings.ndim != 2:
                raise ValueError("loadings must be a 2-D matrix")
            if self.loadings.shape[0] != self.scores.shape[1]:
                raise ValueError("loadings rows must match score columns")
        if sum(self.sampling_frame.blocklens) != self.scores.shape[0]:
            raise ValueError("sampling frame length must match scores rows")

    @property
    def n_runs(self) -> int:
        return len(self.sampling_frame.blocklens)

    @property
    def n_timepoints(self) -> list[int]:
        return [int(v) for v in self.sampling_frame.blocklens]

    @property
    def n_voxels(self) -> int:
        if self.loadings is None:
            return int(self.scores.shape[1])
        return int(self.loadings.shape[1])

    def get_scores(self, run: int = 0) -> NDArray[np.float64]:
        start = int(sum(self.sampling_frame.blocklens[:run]))
        end = start + int(self.sampling_frame.blocklens[run])
        return self.scores[start:end]

    def get_data(self, run: int = 0) -> NDArray[np.float64]:
        scores = self.get_scores(run)
        if self.loadings is None:
            return scores
        return scores @ self.loadings

    def get_mask(self) -> NDArray[np.bool_]:
        return np.ones((self.n_voxels, 1, 1), dtype=bool)

    def get_sampling_frame(self) -> SamplingFrame:
        return self.sampling_frame

    def get_censor(self, run: int = 0) -> None:
        return None


def latent_dataset(
    scores: ArrayLike,
    loadings: Optional[ArrayLike] = None,
    tr: float = 1.0,
    run_length: Optional[Union[int, Sequence[int]]] = None,
    *,
    event_table: Optional[pd.DataFrame] = None,
) -> LatentDataset:
    """Construct a latent-component dataset from scores and optional loadings."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    if scores_arr.ndim != 2:
        raise ValueError("scores must be a 2-D matrix")
    if run_length is None:
        blocklens = [scores_arr.shape[0]]
    elif isinstance(run_length, int):
        if scores_arr.shape[0] % run_length:
            raise ValueError("run_length must divide score rows")
        blocklens = [int(run_length)] * (scores_arr.shape[0] // int(run_length))
    else:
        blocklens = [int(v) for v in run_length]
        if sum(blocklens) != scores_arr.shape[0]:
            raise ValueError("sum(run_length) must equal score rows")
    sf = SamplingFrame(blocklens=blocklens, tr=tr)
    return LatentDataset(scores_arr, None if loadings is None else np.asarray(loadings), sf, event_table)


def fmri_latent_lm(
    model: object,
    dataset: LatentDataset,
    config: Optional[object] = None,
    **kwargs: object,
) -> object:
    """Fit a GLM to latent scores using the normal Python model/config contract.

    ``model`` is an ``FmriModel`` and ``config`` an ``FmriLmConfig``;
    both are typed as ``object`` to avoid a heavy import cycle through
    the GLM layer at this compat surface. The return is the concrete
    ``FmriLm`` produced by ``fit_glm_with_config`` (resolved
    structurally by callers)."""
    from ..glm import fit_glm_with_config

    fit = fit_glm_with_config(model, dataset.scores, cfg=config, dataset=dataset, **kwargs)
    fit.dataset = dataset
    return fit


def voxel_index_chunks(
    x: object,
    nchunks: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> list[NDArray[np.intp]]:
    """Return bare voxel-index chunks for legacy callers."""
    from .data_chunks import voxel_index_chunks as _voxel_index_chunks

    return _voxel_index_chunks(x, nchunks=nchunks, chunk_size=chunk_size)


def extract_csv_data(
    gd: GroupData,
    roi: Optional[str] = None,
    contrast: Optional[str] = None,
) -> Dict[str, Union[pd.DataFrame, NDArray[np.float64]]]:
    """Extract effect-size arrays from CSV-backed group data."""
    if not isinstance(gd, GroupData) or gd.format != "csv":
        raise TypeError("Input must be CSV-backed GroupData")
    df = gd.data["data"].copy()
    roi_col = gd.data.get("roi_col")
    contrast_col = gd.data.get("contrast_col")
    if roi is not None:
        if roi_col is None:
            raise ValueError("Cannot filter by ROI: no ROI column specified")
        df = df[df[roi_col] == roi]
        if df.empty:
            raise ValueError(f"No data found for ROI: {roi}")
    if contrast is not None:
        if contrast_col is None:
            raise ValueError("Cannot filter by contrast: no contrast column specified")
        df = df[df[contrast_col] == contrast]
        if df.empty:
            raise ValueError(f"No data found for contrast: {contrast}")

    effect_cols = gd.data["effect_cols"]
    out: Dict[str, Union[pd.DataFrame, NDArray[np.float64]]] = {
        "data": df.reset_index(drop=True),
    }
    if "beta" in effect_cols:
        out["beta"] = df[effect_cols["beta"]].to_numpy(dtype=np.float64)
    if "se" in effect_cols:
        out["se"] = df[effect_cols["se"]].to_numpy(dtype=np.float64)
        out["var"] = out["se"] ** 2
    if "var" in effect_cols:
        out["var"] = df[effect_cols["var"]].to_numpy(dtype=np.float64)
        out["se"] = np.sqrt(np.maximum(out["var"], 0.0))
    if "t" in effect_cols:
        out["t"] = df[effect_cols["t"]].to_numpy(dtype=np.float64)
    return out


def read_h5_full(gd: GroupData, stat: Optional[Sequence[str]] = None) -> NDArray[np.float64]:
    """Read full HDF5-backed group data into ``voxels x subjects x stats``."""
    if not isinstance(gd, GroupData) or gd.format != "h5":
        raise TypeError("Input must be H5-backed GroupData")
    import h5py

    paths = gd.data["paths"]
    stats = list(stat if stat is not None else gd.data.get("stat", ["beta", "se"]))
    arrays: list[list[NDArray[np.float64]]] = []
    n_voxels: Optional[int] = None
    for path in paths:
        subject_arrays = []
        with h5py.File(path, "r") as handle:
            for name in stats:
                key = "se" if name == "var" and "se" in handle else name
                if key not in handle:
                    raise KeyError(f"Dataset {name!r} not found in {path}")
                values = np.asarray(handle[key], dtype=np.float64).ravel()
                if name == "var" and key == "se":
                    values = values ** 2
                if n_voxels is None:
                    n_voxels = int(values.size)
                elif values.size != n_voxels:
                    raise ValueError("HDF5 statistic lengths must match")
                subject_arrays.append(values)
        arrays.append(subject_arrays)
    data = np.empty((int(n_voxels or 0), len(paths), len(stats)), dtype=np.float64)
    for subject, subject_arrays in enumerate(arrays):
        for stat_idx, values in enumerate(subject_arrays):
            data[:, subject, stat_idx] = values
    return data


def read_nifti_full(gd: GroupData, use_mask: Optional[bool] = None) -> Dict[str, NDArray[np.float64]]:
    """Read full NIfTI-backed group data into subject-by-voxel matrices."""
    if not isinstance(gd, GroupData) or gd.format != "nifti":
        raise TypeError("Input must be NIfTI-backed GroupData")
    from neuroim import read_image

    mask = None
    if gd.data.get("mask") is not None:
        mask = np.asarray(read_image(gd.data["mask"], type="vol").as_array()) > 0
    if use_mask is None:
        use_mask = mask is not None

    out: Dict[str, NDArray[np.float64]] = {}
    for stat, key in (("beta", "beta_paths"), ("se", "se_paths"), ("var", "var_paths"), ("t", "t_paths")):
        paths = gd.data.get(key)
        if not paths:
            continue
        rows = []
        for path in paths:
            arr = np.asarray(read_image(path, type="vol").as_array(), dtype=np.float64)
            rows.append(arr[mask].ravel() if use_mask and mask is not None else arr.ravel())
        out[stat] = np.vstack(rows)
    if "var" not in out and "se" in out:
        out["var"] = out["se"] ** 2
    if gd.data.get("df") is not None:
        out["df"] = np.asarray(gd.data["df"], dtype=np.float64)
    return out


def read_fmri_config(path: Union[str, "os.PathLike[str]"]) -> Dict[str, object]:
    """Read a JSON or YAML fMRI config file into a dictionary.

    Values are returned as decoded JSON/YAML, so the dict is honestly
    heterogeneous (``object`` rather than a tighter type).
    """
    resolved = Path(path)
    text = resolved.read_text()
    if resolved.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise ImportError("YAML config reading requires PyYAML") from exc
    data = yaml.safe_load(text)
    return {} if data is None else dict(data)


def register_basis(name: str, constructor: Callable[..., object]) -> bool:
    """Register a custom basis constructor for resolve_basis."""
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not callable(constructor):
        raise TypeError("constructor must be callable")
    _BASIS_REGISTRY[name] = constructor
    return True


def resolve_basis(name: str, **kwargs: HrfParamValue) -> object:
    """Resolve a registered basis or fall back to the HRF registry."""
    if name in _BASIS_REGISTRY:
        return _BASIS_REGISTRY[name](**kwargs)
    import fmrimod

    try:
        return fmrimod.as_hrf(name, **kwargs)
    except Exception as exc:
        raise KeyError(f"Unknown basis {name!r}") from exc


def _benchmark_data() -> Dict[str, Dict[str, object]]:
    rng = np.random.default_rng(42)
    n_time = 48
    tr = 1.5
    onsets = np.array([4, 10, 16, 22, 28, 34], dtype=float)
    labels = np.array(["A", "B", "A", "B", "A", "B"], dtype=object)
    true_betas = np.array([[1.0, 0.7, 0.2, 1.4], [0.3, 1.1, 0.5, -0.2]])
    base = {
        "description": "Deterministic canonical HRF benchmark fixture",
        "event_onsets": onsets,
        "condition_labels": labels,
        "TR": tr,
        "total_time": n_time * tr,
        "target_snr": "high",
        "true_betas_condition": true_betas,
        "true_hrf_parameters": {"basis": "spmg1"},
        "noise_parameters": {"sigma": 0.05, "seed": 42},
        "event_table": pd.DataFrame({"onset": onsets, "condition": labels, "duration": 0.5}),
    }
    X = create_design_matrix_from_benchmark(base, "spmg1", include_intercept=False)
    Y = np.asarray(X) @ true_betas + rng.normal(scale=0.05, size=(n_time, true_betas.shape[1]))
    base["Y_noisy"] = Y
    return {
        "BM_Canonical_HighSNR": base,
        "metadata": {"source": "fmrimod synthetic compatibility fixture", "n_datasets": 1},
    }


def load_benchmark_dataset(dataset_name: str = "BM_Canonical_HighSNR") -> Dict[str, object]:
    """Load a deterministic built-in benchmark fixture."""
    data = _benchmark_data()
    if dataset_name == "all":
        return {k: v for k, v in data.items() if k != "metadata"}
    if dataset_name in data:
        return data[dataset_name]
    available = ", ".join(k for k in data if k != "metadata")
    raise KeyError(f"Dataset {dataset_name!r} not found. Available datasets: {available}")


def list_benchmark_datasets() -> pd.DataFrame:
    """List deterministic benchmark fixtures."""
    data = load_benchmark_dataset("all")
    return pd.DataFrame(
        {
            "Dataset": list(data),
            "Description": [payload.get("description", "") for payload in data.values()],
        }
    )


def get_benchmark_summary(dataset_name: str) -> Dict[str, object]:
    """Return dimensions and design metadata for a benchmark fixture."""
    dataset = load_benchmark_dataset(dataset_name)
    labels = np.asarray(dataset["condition_labels"], dtype=object)
    unique, counts = np.unique(labels, return_counts=True)
    Y = np.asarray(dataset["Y_noisy"])
    return {
        "description": dataset["description"],
        "dimensions": {
            "n_timepoints": int(Y.shape[0]),
            "n_voxels": int(Y.shape[1]),
            "n_events": int(labels.size),
            "n_conditions": int(unique.size),
        },
        "experimental_design": {
            "conditions": list(unique),
            "events_per_condition": dict(zip(unique.tolist(), counts.astype(int).tolist())),
            "TR": dataset["TR"],
            "total_time": dataset["total_time"],
            "target_snr": dataset.get("target_snr"),
        },
        "hrf_information": dataset.get("true_hrf_parameters"),
        "noise_information": dataset.get("noise_parameters"),
    }


def create_design_matrix_from_benchmark(
    dataset_name: Union[str, Mapping[str, object]],
    hrf: HrfSpec = "spmg1",
    include_intercept: bool = True,
) -> pd.DataFrame:
    """Create a condition-wise design matrix for a benchmark fixture."""
    dataset = dataset_name if isinstance(dataset_name, Mapping) else load_benchmark_dataset(dataset_name)
    import fmrimod

    n_time = int(np.asarray(dataset.get("Y_noisy", np.zeros((48, 1)))).shape[0])
    time_grid = np.arange(n_time, dtype=np.float64) * float(dataset["TR"])
    labels = np.asarray(dataset["condition_labels"], dtype=object)
    columns: Dict[str, NDArray[np.float64]] = {}
    for condition in pd.unique(labels):
        onsets = np.asarray(dataset["event_onsets"], dtype=np.float64)[labels == condition]
        reg = fmrimod.regressor(onsets, hrf, duration=0.5, amplitude=1.0)
        values = np.asarray(fmrimod.evaluate(reg, time_grid), dtype=np.float64)
        if values.ndim == 1:
            columns[str(condition)] = values
        else:
            for idx in range(values.shape[1]):
                columns[f"{condition}_{idx + 1}"] = values[:, idx]
    df = pd.DataFrame(columns)
    if include_intercept:
        df.insert(0, "Intercept", 1.0)
    return df


def _safe_cor(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float(1.0 if np.allclose(x, y) else 0.0)
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_method_performance(
    dataset_name: str,
    estimated_betas: ArrayLike,
    method_name: str = "Unknown",
) -> Dict[str, object]:
    """Evaluate estimated condition betas against benchmark ground truth."""
    dataset = load_benchmark_dataset(dataset_name)
    true = np.asarray(dataset["true_betas_condition"], dtype=np.float64)
    est = np.asarray(estimated_betas, dtype=np.float64)
    if est.shape != true.shape:
        raise ValueError(f"estimated_betas shape {est.shape} does not match true betas {true.shape}")
    err = true - est
    condition_metrics = {}
    for idx in range(true.shape[0]):
        mse = float(np.mean(err[idx] ** 2))
        condition_metrics[f"condition_{idx + 1}"] = {
            "correlation": _safe_cor(true[idx], est[idx]),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }
    return {
        "method_name": method_name,
        "dataset_name": dataset_name,
        "overall_metrics": {
            "correlation": _safe_cor(true.ravel(), est.ravel()),
            "mse": float(np.mean(err ** 2)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
        },
        "condition_metrics": condition_metrics,
        "voxel_metrics": {
            "correlations": np.array([_safe_cor(true[:, j], est[:, j]) for j in range(true.shape[1])]),
            "mse_values": np.mean(err ** 2, axis=0),
        },
    }


def design_plot(
    model: object,
    term_name: Optional[str] = None,
    longnames: bool = False,
    **kwargs: object,
) -> pd.DataFrame:
    """Return a long-form design matrix table suitable for plotting."""
    del term_name, longnames, kwargs
    design = model.design_matrix() if hasattr(model, "design_matrix") else model
    df = design.copy() if isinstance(design, pd.DataFrame) else pd.DataFrame(np.asarray(design))
    df = df.reset_index(names="time_index")
    return df.melt(id_vars="time_index", var_name="condition", value_name="value")
