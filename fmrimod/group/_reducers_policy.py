"""Chunking, threading, and dataset-adapter helpers for the group reducers.

Third slice of bd-01KRHTJ9WFSSBZSDGAN4V7PHGS (policy/kernel/registry
split). Sits between the pure-compute kernels in
:mod:`fmrimod.group._reducers_kernels` and the public reducer
implementations in :mod:`fmrimod.group.reducers`: how features are
chunked and dispatched across workers, how BLAS thread limits are
applied, how an adapter pulls ``beta``/``var`` out of a
:class:`GroupDataset`, and how reduce results and design matrices are
packaged.

Nothing here is part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from math import ceil
from typing import TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from patsy import dmatrix

from .dataset import GroupDataset
from .errors import AdapterContractError

T = TypeVar("T")


@contextmanager
def _maybe_limit_blas_threads(blas_threads: int | None) -> Iterator[None]:
    if blas_threads is None:
        yield
        return
    if int(blas_threads) < 1:
        raise AdapterContractError("blas_threads must be >= 1")
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - optional dependency behavior
        yield
        return
    with threadpool_limits(limits=int(blas_threads)):
        yield


def _feature_chunks(
    n_features: int,
    *,
    n_jobs: int = 1,
    chunk_size: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    if int(n_jobs) < 1:
        raise AdapterContractError("n_jobs must be >= 1")
    if chunk_size is not None and int(chunk_size) < 1:
        raise AdapterContractError("chunk_size must be >= 1")
    if n_features < 1:
        return [], 1
    n_workers = min(int(n_jobs), n_features)
    size = int(chunk_size) if chunk_size is not None else max(1, ceil(n_features / n_workers))
    chunks = [
        (start, min(start + size, n_features))
        for start in range(0, n_features, size)
    ]
    return chunks, min(n_workers, len(chunks))


def _run_feature_chunks(
    n_features: int,
    worker: Callable[[int, int], T],
    *,
    n_jobs: int = 1,
    chunk_size: int | None = None,
    blas_threads: int | None = None,
) -> tuple[list[T], int, int | None]:
    chunks, n_workers = _feature_chunks(
        n_features,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
    if not chunks:
        return [], n_workers, chunk_size
    if n_workers == 1:
        with _maybe_limit_blas_threads(blas_threads):
            return [worker(start, end) for start, end in chunks], n_workers, chunk_size
    with _maybe_limit_blas_threads(blas_threads), ThreadPoolExecutor(
        max_workers=n_workers
    ) as pool:
        results = list(pool.map(lambda bounds: worker(*bounds), chunks))
    return results, n_workers, chunk_size


def _beta_and_var(
    dataset: GroupDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    beta = dataset.assay("beta")
    if "var" in dataset.assays:
        var = dataset.assay("var")
    elif "se" in dataset.assays:
        se = dataset.assay("se")
        var = se * se
    else:
        raise AdapterContractError("meta:fe requires beta with var or se")
    return beta, var


def _group_col_data() -> pd.DataFrame:
    return pd.DataFrame(index=pd.Index(["group"], name="subject"))


def _reduced_dataset(
    dataset: GroupDataset,
    assays: dict[str, NDArray[np.float64]],
    *,
    method: str,
    metadata: dict[str, object] | None = None,
) -> GroupDataset:
    return GroupDataset(
        assays=assays,
        space=dataset.space,
        subjects=["group"],
        contrasts=dataset.contrasts,
        col_data=_group_col_data(),
        row_data=dataset.row_data,
        contrast_data=dataset.contrast_data,
        metadata={
            **dict(dataset.metadata),
            "operation": "reduce",
            "reduce_method": method,
            **({} if metadata is None else metadata),
        },
    )


def _design_matrix(
    dataset: GroupDataset,
    *,
    X: NDArray[np.float64] | None = None,
    formula: str = "~ 1",
) -> tuple[NDArray[np.float64], list[str]]:
    if X is not None:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise AdapterContractError("X must be a 2-D subjects x predictors matrix")
        if X_arr.shape[0] != dataset.n_subjects:
            raise AdapterContractError("X rows must match number of subjects")
        if not np.all(np.isfinite(X_arr)):
            raise AdapterContractError("X must contain finite values")
        return X_arr, [f"x{i}" for i in range(X_arr.shape[1])]

    clean = formula.replace(" ", "")
    if clean in ("~1", "1"):
        return np.ones((dataset.n_subjects, 1), dtype=np.float64), ["Intercept"]
    if dataset.col_data is None:
        raise AdapterContractError("formula with covariates requires dataset.col_data")
    design = dmatrix(
        formula, dataset.col_data.reset_index(drop=True), return_type="dataframe"
    )
    return np.asarray(design, dtype=np.float64), list(design.columns)


__all__ = [
    "T",
    "_maybe_limit_blas_threads",
    "_feature_chunks",
    "_run_feature_chunks",
    "_beta_and_var",
    "_group_col_data",
    "_reduced_dataset",
    "_design_matrix",
]
