"""Fitting strategies for fMRI GLMs.

Implements runwise and chunkwise strategies for processing multi-run
fMRI datasets.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from ..dataset.data_access import get_run_data
from ..model.config import FmriLmConfig
from .errors import UnsupportedEngineConfiguration
from .preprocess import (
    _soft_projection_residual_df,
    _soft_subspace_projection,
    apply_censoring,
    apply_volume_weights,
    compute_dvars,
    dvars_weights,
    extract_nuisance_timeseries,
)
from .solver import (
    LmResult,
    Projection,
    _lean_rss,
    fast_lm_matrix,
    fast_preproject,
)


@contextmanager
def _maybe_limit_blas_threads(blas_threads: Optional[int]) -> "Iterator[None]":
    """Limit BLAS threads within a context when threadpoolctl is available."""
    if blas_threads is None:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
    except Exception:
        yield
        return
    with threadpool_limits(limits=int(blas_threads)):
        yield


def _projection_cache_key(
    X: NDArray[np.float64],
    compute_dtype: object,
    method: str,
) -> Tuple[Any, ...]:
    """Build a stable cache key for a design matrix projection."""
    Xc = np.ascontiguousarray(X, dtype=np.dtype(cast(Any, compute_dtype)))
    h = hashlib.blake2b(digest_size=16)
    h.update(memoryview(cast(Any, Xc)).cast("B"))
    return (Xc.shape, Xc.dtype.str, method, h.digest())


def _dataset_run_lengths(dataset: object) -> list[int] | None:
    """Return per-run lengths from dataset-like objects when available."""
    if hasattr(dataset, "run_lengths"):
        lengths = getattr(dataset, "run_lengths")
    elif hasattr(dataset, "n_timepoints"):
        lengths = getattr(dataset, "n_timepoints")
    else:
        return None
    if isinstance(lengths, int):
        return [int(lengths)]
    return [int(v) for v in lengths]


def _fit_one_run(
    model: object,
    config: FmriLmConfig,
    run_idx: int,
    needs_residuals: bool,
    compute_dtype: object,
) -> tuple[int, LmResult, Projection, Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """Fit one run and return all run-local outputs."""
    Y_r = get_run_data(model.dataset, run_idx)  # type: ignore[attr-defined]
    X_r = model.design_matrix_array(run=run_idx)  # type: ignore[attr-defined]

    censor_r = None
    dataset = model.dataset  # type: ignore[attr-defined]
    if hasattr(dataset, "get_censor"):
        censor_r = dataset.get_censor(run_idx)

    result, proj, X_used, Y_used = fit_run_ols(
        X_r,
        Y_r,
        config,
        censor_r,
        dataset=dataset,
        run=run_idx,
        return_fitted=needs_residuals,
        compute_dtype=compute_dtype,
    )

    if not needs_residuals:
        return run_idx, result, proj, None, None

    if result.fitted is not None:
        residual = Y_used - result.fitted
    else:
        residual = Y_used - X_used @ result.betas
    return run_idx, result, proj, residual, X_used


def fit_run_ols(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    config: FmriLmConfig,
    censor: Optional[NDArray[np.bool_]] = None,
    dataset: Optional[object] = None,
    run: Optional[int] = None,
    return_fitted: bool = True,
    compute_dtype: object = np.float64,
    projection_cache: Optional[Dict[Tuple[Any, ...], Projection]] = None,
) -> Tuple[LmResult, Projection, NDArray[np.float64], NDArray[np.float64]]:
    """Fit a single run with OLS (possibly after preprocessing).

    Parameters
    ----------
    X : NDArray
        Design matrix for this run, shape ``(n, p)``.
    Y : NDArray
        Data matrix for this run, shape ``(n, V)``.
    config : FmriLmConfig
        Fitting configuration.
    censor : NDArray[bool], optional
        Censoring vector for this run.
    dataset : object, optional
        Dataset handle used for nuisance-mask extraction/slicing.
    run : int, optional
        Zero-indexed run number for run-aware nuisance slicing.
    return_fitted : bool
        If ``True``, include fitted values in ``LmResult``. Set ``False``
        for faster RSS-only fits when residual time-series are not needed.
    compute_dtype : numpy dtype-like
        Internal solver dtype (``float64`` default, optional ``float32``).
    projection_cache : dict, optional
        Optional cache mapping design fingerprints to precomputed
        :class:`Projection` objects.

    Returns
    -------
    result : LmResult
        Regression result.
    proj : Projection
        Pre-computed projection (for later contrast computation).
    X_used : NDArray
        The (possibly preprocessed) design matrix that was actually fitted.
    Y_used : NDArray
        The (possibly preprocessed) data matrix that was actually fitted.
    """
    X_fit, Y_fit, prep = _prepare_run_matrices(
        X,
        Y,
        config,
        censor=censor,
        dataset=dataset,
        run=run,
    )

    # 4. Fit OLS
    proj = None
    cache_key = None
    if projection_cache is not None:
        cache_key = _projection_cache_key(X_fit, compute_dtype, config.solver)
        proj = projection_cache.get(cache_key)
    if proj is None:
        proj = fast_preproject(
            X_fit,
            compute_dtype=compute_dtype,
            check_finite=False,
            method=config.solver,
        )
        if projection_cache is not None and cache_key is not None:
            projection_cache[cache_key] = proj
    proj = _with_preprocessing_df(proj, X_fit, prep)
    result = fast_lm_matrix(
        X_fit,
        Y_fit,
        proj,
        return_fitted=return_fitted,
        compute_dtype=compute_dtype,
        check_finite=False,
    )
    result.obs_weights = prep.obs_weights

    return result, proj, X_fit, Y_fit


@dataclass(frozen=True)
class _RunPreprocessing:
    """Preprocessing applied to one run that changes its residual df.

    ``obs_weights`` are the volume weights applied to the rows of ``X`` and
    ``Y`` (after censoring); ``soft_basis`` is ``(U, shrink)`` of the soft
    subspace projection ``P = I - U diag(shrink) U'``.
    """

    obs_weights: NDArray[np.float64] | None = None
    soft_basis: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None


def _with_preprocessing_df(
    proj: Projection,
    X_fit: NDArray[np.float64],
    prep: _RunPreprocessing,
) -> Projection:
    """Return ``proj`` with the residual df of the preprocessed run.

    A zero-weight volume is not an observation (``stats::lm.wfit``,
    fmrireg 60f8430), and a soft subspace projection shrinks the residual
    space to ``tr((I - H) P^2)`` (fmrireg 5edfd16); the soft-projection df
    takes precedence, as in fmrireg. A cached projection is never mutated.
    """
    dfres: float | None = None
    if prep.obs_weights is not None:
        dfres = float(np.count_nonzero(prep.obs_weights > 0) - proj.rank)
    if prep.soft_basis is not None:
        U, shrink = prep.soft_basis
        Q = proj.Q
        if Q is None:
            Q = linalg.qr(X_fit, pivoting=True, mode="economic", check_finite=False)[0]
            Q = Q[:, : proj.rank]
        dfres = _soft_projection_residual_df(U, shrink, Q)
    if dfres is None or dfres == proj.dfres:
        return proj
    return replace(proj, dfres=dfres)


def _prepare_run_matrices(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    config: FmriLmConfig,
    *,
    censor: Optional[NDArray[np.bool_]] = None,
    dataset: Optional[object] = None,
    run: Optional[int] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], _RunPreprocessing]:
    """Apply censoring/weighting/subspace preprocessing for one run."""
    X_fit = X
    Y_fit = Y
    obs_weights: NDArray[np.float64] | None = None
    soft_basis: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None

    if config.na_action == "error" and not np.all(np.isfinite(Y_fit)):
        run_label = "?" if run is None else str(run)
        raise ValueError(
            "Non-finite values (NA/NaN/Inf) in the response data for "
            f"run {run_label}. Set na_action='propagate' in FmriLmConfig "
            "to surface affected voxels as NaN, or clean/impute the data."
        )

    # 1. Censoring — coerce integer 0/1 to boolean for fmrireg parity
    if censor is not None:
        censor = np.asarray(censor)
        if censor.dtype.kind in ("i", "u", "f"):
            unique = np.unique(censor)
            if not np.all(np.isin(unique, [0, 1])):
                raise ValueError("Censor vector must be boolean or binary (0/1)")
            censor = censor.astype(bool)
    if censor is not None and np.any(censor):
        X_fit, Y_fit, _ = apply_censoring(X_fit, Y_fit, censor)

    # 2. Volume weighting
    if config.volume_weights.enabled:
        if config.volume_weights.weights is not None:
            weights = np.asarray(config.volume_weights.weights, dtype=np.float64)

            # R parity: allow all-run weight vectors and slice selected run.
            if (
                dataset is not None
                and run is not None
                and _dataset_run_lengths(dataset) is not None
                and weights.shape[0] != X.shape[0]
            ):
                run_lengths = _dataset_run_lengths(dataset)
                assert run_lengths is not None
                total_rows = int(sum(run_lengths))
                if weights.shape[0] == total_rows:
                    start = int(sum(run_lengths[:run]))
                    end = start + run_lengths[run]
                    weights = weights[start:end]

            if censor is not None and np.any(censor):
                n_rows = X.shape[0]
                n_keep = X_fit.shape[0]
                n_weights = weights.shape[0]
                if n_weights == n_rows:
                    weights = weights[~censor]
                elif n_weights != n_keep:
                    raise ValueError(
                        f"weights length {n_weights} does not match number of timepoints "
                        f"{n_rows} or uncensored rows {n_keep}"
                    )
        else:
            dvars = compute_dvars(Y_fit)
            weights = dvars_weights(
                dvars,
                method=config.volume_weights.method,
                threshold=config.volume_weights.threshold,
            )
        X_fit, Y_fit = apply_volume_weights(X_fit, Y_fit, weights)
        obs_weights = np.asarray(weights, dtype=np.float64)

    # 3. Soft subspace projection
    if config.soft_subspace.enabled:
        if config.soft_subspace.nuisance_matrix is not None:
            nuisance = np.asarray(config.soft_subspace.nuisance_matrix, dtype=np.float64)
            if nuisance.ndim == 1:
                nuisance = nuisance[:, np.newaxis]
            if nuisance.ndim != 2:
                raise ValueError("nuisance_matrix must be 1-D or 2-D")

            # R parity: allow nuisance matrix defined over all runs, then
            # slice this run before optional censoring.
            if (
                dataset is not None
                and run is not None
                and _dataset_run_lengths(dataset) is not None
                and nuisance.shape[0] != X.shape[0]
            ):
                run_lengths = _dataset_run_lengths(dataset)
                assert run_lengths is not None
                total_rows = int(sum(run_lengths))
                if nuisance.shape[0] == total_rows:
                    start = int(sum(run_lengths[:run]))
                    end = start + run_lengths[run]
                    nuisance = nuisance[start:end]

            if censor is not None and np.any(censor):
                nuisance = nuisance[~censor]
            if nuisance.shape[0] != X_fit.shape[0]:
                raise ValueError(
                    f"nuisance_matrix rows ({nuisance.shape[0]}) must match data rows ({X_fit.shape[0]})"
                )

            X_fit, Y_fit, soft_basis = _soft_subspace_projection(
                X_fit,
                Y_fit,
                nuisance,
                config.soft_subspace.lam,
            )
        elif config.soft_subspace.nuisance_mask is not None:
            dataset_mask = None
            if dataset is not None and hasattr(dataset, "get_mask"):
                dataset_mask = dataset.get_mask()
            nuisance = extract_nuisance_timeseries(
                Y_fit,
                config.soft_subspace.nuisance_mask,
                dataset_mask=dataset_mask,
            )
            X_fit, Y_fit, soft_basis = _soft_subspace_projection(
                X_fit,
                Y_fit,
                nuisance,
                config.soft_subspace.lam,
            )

    return X_fit, Y_fit, _RunPreprocessing(obs_weights, soft_basis)


def fit_runwise(
    model: object,  # FmriModel
    config: FmriLmConfig,
    n_jobs: int = 1,
    blas_threads: Optional[int] = None,
    compute_dtype: object = np.float64,
    cache_projections: bool = False,
    chunk_size: int = 5000,
) -> Dict[str, Any]:
    """Fit the GLM run by run and pool results.

    This is the default strategy: each run is fitted independently,
    then beta estimates and variances are combined via fixed-effects
    meta-analysis (inverse-variance weighting).

    Parameters
    ----------
    model : FmriModel
        The fMRI model (provides design matrices and data).
    config : FmriLmConfig
        Fitting configuration.

    Returns
    -------
    dict
        Dictionary with keys ``betas``, ``sigma``, ``dfres``, ``XtXinv``,
        ``projections``, ``run_results``, ``residuals``.
    """
    n_runs = model.n_runs  # type: ignore[attr-defined]
    n_workers = max(1, min(int(n_jobs), int(n_runs)))
    needs_residuals = config.ar.enabled or config.robust.enabled

    run_results: List[Optional[LmResult]] = [None] * n_runs
    run_projections: List[Optional[Projection]] = [None] * n_runs
    run_residuals: Optional[List[Optional[NDArray[np.float64]]]] = (
        [None] * n_runs if needs_residuals else None
    )
    run_X: Optional[List[Optional[NDArray[np.float64]]]] = (
        [None] * n_runs if needs_residuals else None
    )

    # Single-run fast path: allow voxel-chunk parallelism when n_jobs > 1.
    if (
        n_runs == 1
        and n_jobs > 1
        and not needs_residuals
        and np.dtype(cast(Any, compute_dtype)) == np.dtype(np.float64)
    ):
        dataset = model.dataset  # type: ignore[attr-defined]
        Y_r = get_run_data(dataset, 0)
        X_r = model.design_matrix_array(run=0)  # type: ignore[attr-defined]
        censor_r = dataset.get_censor(0) if hasattr(dataset, "get_censor") else None

        X_fit, Y_fit, prep = _prepare_run_matrices(
            X_r,
            Y_r,
            config,
            censor=censor_r,
            dataset=dataset,
            run=0,
        )
        proj = fast_preproject(
            X_fit,
            compute_dtype=compute_dtype,
            check_finite=False,
            method=config.solver,
        )
        proj = _with_preprocessing_df(proj, X_fit, prep)
        result = _fit_chunked_lm(
            X_fit,
            Y_fit,
            proj,
            chunk_size=chunk_size,
            n_jobs=max(1, int(n_jobs)),
            blas_threads=blas_threads,
            compute_dtype=compute_dtype,
        )
        pooled = _pool_run_results([result], [proj])
        return {
            "betas": pooled["betas"],
            "sigma": pooled["sigma"],
            "dfres": pooled["dfres"],
            "XtXinv": pooled["XtXinv"],
            "projections": [proj],
            "run_results": [result],
            "residuals": None,
            "run_X": None,
        }

    if n_workers == 1:
        proj_cache: Optional[Dict[Tuple[Any, ...], Projection]] = {} if cache_projections else None
        for r in range(n_runs):
            # Get per-run data and design
            Y_r = get_run_data(model.dataset, r)  # type: ignore[attr-defined]
            X_r = model.design_matrix_array(run=r)  # type: ignore[attr-defined]

            # Get censor for this run
            censor_r = None
            dataset = model.dataset  # type: ignore[attr-defined]
            if hasattr(dataset, "get_censor"):
                censor_r = dataset.get_censor(r)

            result, proj, X_used, Y_used = fit_run_ols(
                X_r,
                Y_r,
                config,
                censor_r,
                dataset=dataset,
                run=r,
                return_fitted=needs_residuals,
                compute_dtype=compute_dtype,
                projection_cache=proj_cache,
            )
            run_results[r] = result
            run_projections[r] = proj
            if needs_residuals:
                assert run_X is not None
                assert run_residuals is not None
                run_X[r] = X_used
                # Compute residuals only when downstream AR/robust needs them.
                if result.fitted is not None:
                    run_residuals[r] = Y_used - result.fitted
                else:
                    run_residuals[r] = Y_used - X_used @ result.betas
    else:
        if cache_projections:
            # Cache coordination across workers would require explicit locking
            # and can negate gains; disable in threaded mode.
            cache_projections = False
        with _maybe_limit_blas_threads(blas_threads):
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [
                    ex.submit(
                        _fit_one_run,
                        model,
                        config,
                        r,
                        needs_residuals,
                        compute_dtype,
                    )
                    for r in range(n_runs)
                ]
                for fut in futures:
                    t_run_idx, t_result, t_proj, t_residual, t_X_used = fut.result()
                    run_results[t_run_idx] = t_result
                    run_projections[t_run_idx] = t_proj
                    if needs_residuals:
                        assert run_X is not None
                        assert run_residuals is not None
                        assert t_residual is not None
                        assert t_X_used is not None
                        run_X[t_run_idx] = t_X_used
                        run_residuals[t_run_idx] = t_residual

    # Narrow optional element types after fill.
    run_results_typed = [r for r in run_results if r is not None]
    run_proj_typed = [p for p in run_projections if p is not None]
    run_residuals_typed: Optional[List[NDArray[np.float64]]] = None
    run_x_typed: Optional[List[NDArray[np.float64]]] = None
    if needs_residuals:
        assert run_residuals is not None
        assert run_X is not None
        run_residuals_typed = [rr for rr in run_residuals if rr is not None]
        run_x_typed = [xx for xx in run_X if xx is not None]

    # Pool across runs via fixed-effects meta-analysis
    pooled = _pool_run_results(run_results_typed, run_proj_typed)

    return {
        "betas": pooled["betas"],
        "sigma": pooled["sigma"],
        "dfres": pooled["dfres"],
        "XtXinv": pooled["XtXinv"],
        "projections": run_proj_typed,
        "run_results": run_results_typed,
        "residuals": run_residuals_typed,
        "run_X": run_x_typed,
    }


def fit_concat(
    model: object,
    config: FmriLmConfig,
    *,
    compute_dtype: object = np.float64,
) -> Dict:
    """Single-design OLS on the run-concatenated ``X`` and ``Y``.

    The runwise strategy fits each run independently and pools per-run
    betas via inverse-variance weighting. That recovers the same betas
    as a single OLS for orthogonal block-diagonal designs, but the
    residual variance is estimated per-run and pooled — which is *not*
    the textbook variance for cross-run contrasts.

    The concat strategy stacks the per-run design and data matrices,
    runs ``fast_lm_matrix`` once on the concatenated system, and
    returns a single :class:`Projection` and a single ``dfres = n -
    rank``. This is what users want when a contrast vector spans
    multiple runs (e.g. "is A's effect different in run 1 vs run 2?")
    — the variance term then reflects the full concatenated residual
    space, matching what Nilearn's ``run_glm`` would produce on the
    same ``X``.

    Returns
    -------
    dict
        ``betas``, ``sigma``, ``dfres``, ``XtXinv``, ``projections``
        (single-element list), and ``run_results`` (single-element
        list with the concatenated :class:`LmResult`).
    """

    X = np.asarray(
        model.design_matrix_array(run=None),  # type: ignore[attr-defined]
        dtype=compute_dtype,
    )
    Y_full = _gather_concatenated_data(model)

    censor_mask = _gather_concatenated_censor(model, expected_rows=X.shape[0])
    # Route the concatenated system through the same preprocessing pipeline
    # the runwise strategy uses, so censoring, volume weights, and
    # soft-subspace projection are all honored instead of silently dropped.
    # ``run=None`` tells ``_prepare_run_matrices`` to treat X/Y as the
    # already-concatenated, full-length matrices (weights / nuisance are
    # defined over all rows here, so no per-run slicing is needed).
    X, Y_full, prep = _prepare_run_matrices(
        X,
        Y_full,
        config,
        censor=censor_mask,
        dataset=getattr(model, "dataset", None),
        run=None,
    )

    proj = _with_preprocessing_df(fast_preproject(X, compute_dtype=compute_dtype), X, prep)
    lm = fast_lm_matrix(
        X,
        Y_full,
        proj,
        compute_dtype=compute_dtype,
        check_finite=config.na_action == "error",
    )

    return {
        "betas": lm.betas,
        "sigma": np.sqrt(np.maximum(lm.sigma2, 0.0)),
        "dfres": float(proj.dfres),
        "XtXinv": proj.XtXinv,
        "projections": [proj],
        "run_results": [lm],
        "residuals": None,
        "run_X": [X],
    }


def _gather_concatenated_data(model: object) -> NDArray[np.float64]:
    """Return the run-concatenated ``timepoints x voxels`` data matrix."""
    dataset = getattr(model, "dataset", None)
    if dataset is None:
        raise ValueError("fit_concat: model has no `.dataset` attribute")
    matrix = getattr(dataset, "get_data_matrix", None)
    if callable(matrix):
        return np.asarray(matrix(), dtype=np.float64)
    # Fall back to per-run gather.
    n_runs = int(getattr(model, "n_runs", 1))
    parts: list[NDArray[np.float64]] = []
    for r in range(n_runs):
        parts.append(np.asarray(get_run_data(dataset, r), dtype=np.float64))
    return np.vstack(parts)


def _gather_concatenated_censor(
    model: object, *, expected_rows: int
) -> Optional[NDArray[np.bool_]]:
    """Concatenate per-run censor vectors, returning ``None`` if none set.

    The runwise strategy consumes ``dataset.get_censor(run_idx)`` one
    run at a time. The concat strategy stacks the per-run designs and
    data into a single system, so the censor masks need to be stacked
    in the same order. Returns ``None`` when the dataset has no
    censoring (so the caller can skip the row-deletion entirely).
    """
    dataset = getattr(model, "dataset", None)
    if dataset is None:
        return None
    if getattr(dataset, "censor", None) is None:
        return None
    n_runs = int(getattr(model, "n_runs", 1))
    parts: list[NDArray[np.bool_]] = []
    for r in range(n_runs):
        mask = dataset.get_censor(r)
        if mask is None:
            # No censoring on this run — keep every row.
            mask = np.zeros(int(get_run_data(dataset, r).shape[0]), dtype=bool)
        parts.append(np.asarray(mask, dtype=bool).ravel())
    concatenated = np.concatenate(parts)
    if concatenated.shape[0] != expected_rows:
        raise ValueError(
            f"fit_concat: concatenated censor length "
            f"{concatenated.shape[0]} does not match design rows "
            f"{expected_rows}"
        )
    if not np.any(concatenated):
        return None
    return concatenated


def _fit_chunked_lm(
    X_fit: NDArray[np.float64],
    Y_fit: NDArray[np.float64],
    proj: Projection,
    *,
    chunk_size: int,
    n_jobs: int,
    blas_threads: Optional[int],
    compute_dtype: object,
) -> LmResult:
    """Fit OLS for a run by processing voxels in contiguous chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    n_voxels = int(Y_fit.shape[1])
    p_dim = int(X_fit.shape[1])
    n_workers = max(1, int(n_jobs))
    out_dtype = np.dtype(cast(Any, compute_dtype))

    betas = np.empty((p_dim, n_voxels), dtype=out_dtype)
    sigma2 = np.empty(n_voxels, dtype=np.float64)
    rss = np.empty(n_voxels, dtype=np.float64)

    def _solve_one(
        start: int,
        end: int,
    ) -> tuple[
        int,
        int,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        y_chunk = Y_fit[:, start:end]
        beta_chunk = proj.Pinv @ y_chunk
        if proj.is_full_rank and not proj.ill_conditioned:
            rss_chunk = _lean_rss(X_fit, y_chunk, beta_chunk, proj)
        else:
            residuals = y_chunk - X_fit @ beta_chunk
            rss_chunk = np.sum(residuals ** 2, axis=0)
        sigma2_chunk = (
            rss_chunk / proj.dfres if proj.dfres > 0 else np.full_like(rss_chunk, np.nan)
        )
        return start, end, beta_chunk, rss_chunk, sigma2_chunk

    ranges = [
        (start, min(start + chunk_size, n_voxels))
        for start in range(0, n_voxels, chunk_size)
    ]

    if n_workers == 1 or len(ranges) == 1:
        for start, end in ranges:
            s, e, b, r, s2 = _solve_one(start, end)
            betas[:, s:e] = b
            rss[s:e] = r
            sigma2[s:e] = s2
    else:
        with _maybe_limit_blas_threads(blas_threads):
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_solve_one, s, e) for s, e in ranges]
                for fut in futures:
                    s, e, b, r, s2 = fut.result()
                    betas[:, s:e] = b
                    rss[s:e] = r
                    sigma2[s:e] = s2

    return LmResult(
        betas=betas,
        rss=rss,
        sigma2=sigma2,
        dfres=proj.dfres,
        rank=proj.rank,
        fitted=None,
    )


def fit_chunkwise(
    model: object,  # FmriModel
    config: FmriLmConfig,
    *,
    chunk_size: int = 5000,
    n_jobs: int = 1,
    blas_threads: Optional[int] = None,
    compute_dtype: object = np.float64,
    cache_projections: bool = False,
) -> Dict[str, Any]:
    """Fit GLM in voxel chunks (fmrireg-style chunkwise strategy).

    Notes
    -----
    This strategy supports the same per-run OLS preprocessing as
    :func:`fit_run_ols` (censoring, volume weights, soft-subspace).
    Chunkwise is intentionally OLS-only for post-fit composition:
    AR/robust post-processing remains runwise-only and raises a typed
    capability error with a repair path when requested here.
    """
    current_capability = ("OLS", "censoring", "volume weights", "soft-subspace")
    if config.ar.enabled:
        raise UnsupportedEngineConfiguration(
            engine="chunkwise",
            feature="AR modeling",
            repair='use engine="runwise" or set ar.struct="iid"',
            current_capability=current_capability,
        )
    if config.robust.enabled:
        raise UnsupportedEngineConfiguration(
            engine="chunkwise",
            feature="robust fitting",
            repair='use engine="runwise" or set robust.type=False',
            current_capability=current_capability,
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    n_runs = model.n_runs  # type: ignore[attr-defined]
    total_workers = max(1, int(n_jobs))
    run_workers = max(1, min(total_workers, int(n_runs)))
    chunk_workers = max(1, total_workers // run_workers)
    run_results: List[Optional[LmResult]] = [None] * n_runs
    run_projections: List[Optional[Projection]] = [None] * n_runs
    proj_cache: Optional[Dict[Tuple[Any, ...], Projection]] = (
        {} if (cache_projections and run_workers == 1) else None
    )

    def _fit_run(r: int) -> tuple[int, LmResult, Projection]:
        Y_r = get_run_data(model.dataset, r)  # type: ignore[attr-defined]
        X_r = model.design_matrix_array(run=r)  # type: ignore[attr-defined]

        dataset = model.dataset  # type: ignore[attr-defined]
        censor_r = None
        if hasattr(dataset, "get_censor"):
            censor_r = dataset.get_censor(r)

        X_fit, Y_fit, prep = _prepare_run_matrices(
            X_r,
            Y_r,
            config,
            censor=censor_r,
            dataset=dataset,
            run=r,
        )

        proj = None
        cache_key = None
        if proj_cache is not None:
            cache_key = _projection_cache_key(X_fit, compute_dtype, config.solver)
            proj = proj_cache.get(cache_key)
        if proj is None:
            proj = fast_preproject(
                X_fit,
                compute_dtype=compute_dtype,
                check_finite=False,
                method=config.solver,
            )
            if proj_cache is not None and cache_key is not None:
                proj_cache[cache_key] = proj
        proj = _with_preprocessing_df(proj, X_fit, prep)

        result = _fit_chunked_lm(
            X_fit,
            Y_fit,
            proj,
            chunk_size=chunk_size,
            n_jobs=chunk_workers,
            blas_threads=blas_threads,
            compute_dtype=compute_dtype,
        )
        return r, result, proj

    if run_workers == 1:
        for r in range(n_runs):
            run_idx, result, proj = _fit_run(r)
            run_results[run_idx] = result
            run_projections[run_idx] = proj
    else:
        with _maybe_limit_blas_threads(blas_threads):
            with ThreadPoolExecutor(max_workers=run_workers) as ex:
                futures = [ex.submit(_fit_run, r) for r in range(n_runs)]
                for fut in futures:
                    run_idx, result, proj = fut.result()
                    run_results[run_idx] = result
                    run_projections[run_idx] = proj

    run_results_typed = [r for r in run_results if r is not None]
    run_proj_typed = [p for p in run_projections if p is not None]

    pooled = _pool_run_results(run_results_typed, run_proj_typed)
    return {
        "betas": pooled["betas"],
        "sigma": pooled["sigma"],
        "dfres": pooled["dfres"],
        "XtXinv": pooled["XtXinv"],
        "projections": run_proj_typed,
        "run_results": run_results_typed,
        "residuals": None,
        "run_X": None,
    }


def _pool_run_results(
    results: List[LmResult],
    projections: List[Projection],
    *,
    covariances: list[NDArray[np.float64]] | None = None,
    grams: list[NDArray[np.float64]] | None = None,
) -> Dict[str, Any]:
    """Pool per-run OLS results via fixed-effects meta-analysis.

    Uses inverse-variance weighting to combine betas across runs.

    Parameters
    ----------
    results : list of LmResult
        Per-run regression results.
    projections : list of Projection
        Per-run projections.
    covariances, grams : list of NDArray, optional
        Per-voxel ``(X'X)^-1`` and ``X'X`` of each run, shape ``(V, p, p)``,
        when every voxel was solved with its own design (voxelwise AR);
        they replace the projections' matrices.

    Returns
    -------
    dict
        Pooled ``betas``, ``sigma``, ``dfres``, ``XtXinv`` (``(V, p, p)``
        when ``covariances`` are given).
    """
    if len(results) == 1:
        r = results[0]
        p = projections[0]
        return {
            "betas": r.betas,
            "sigma": np.sqrt(r.sigma2),
            "dfres": r.dfres,
            "XtXinv": p.XtXinv if covariances is None else covariances[0],
        }

    p_dim = results[0].betas.shape[0]
    V = results[0].betas.shape[1]

    # Inverse-variance beta pooling (streaming; avoids large temporary stacks).
    # se_{r,j,v}^2 = sigma2_{r,v} * XtXinv_{r,j,j}
    wsum = np.zeros((p_dim, V), dtype=np.float64)
    wbeta = np.zeros((p_dim, V), dtype=np.float64)
    n_exact = np.zeros((p_dim, V), dtype=np.float64)
    exact_beta = np.zeros((p_dim, V), dtype=np.float64)
    beta_mean = np.zeros((p_dim, V), dtype=np.float64)

    for i, (r, proj) in enumerate(zip(results, projections, strict=True)):
        beta_mean += r.betas
        if covariances is None:
            diag_xtxinv = np.maximum(np.diag(proj.XtXinv), 0.0)[:, np.newaxis]
        else:
            diag_xtxinv = np.maximum(np.einsum("vjj->jv", covariances[i]), 0.0)
        # A column the run does not estimate carries no information there:
        # on full-width multi-run designs each run's design block is zero
        # for the other runs' columns, which the pivoted QR reports as
        # aliased. This is a property of the design, not of the data scale.
        estimated = diag_xtxinv > 0.0
        estimated[list(proj.aliased_indices)] = False
        se2 = diag_xtxinv * r.sigma2[np.newaxis, :]  # (p,V)
        positive = estimated & (se2 > 0.0)
        exact = estimated & (se2 == 0.0)
        w = np.divide(1.0, se2, out=np.zeros_like(se2), where=positive)
        wsum += w
        wbeta += w * r.betas
        # A zero-variance run has infinite precision: in the limit it takes
        # all the weight, shared equally with any other zero-variance run
        # (fmrireg do_fixef, 1733791).
        n_exact += exact
        exact_beta += np.where(exact, r.betas, 0.0)

    beta_mean /= float(len(results))
    betas_pooled = np.divide(
        wbeta,
        wsum,
        out=beta_mean,
        where=wsum > 0.0,
    )
    betas_pooled = np.divide(
        exact_beta,
        n_exact,
        out=betas_pooled,
        where=n_exact > 0.0,
    )

    # Keep a pooled XtXinv for downstream contrasts/SE interfaces.
    # This follows the prior behavior based on summed per-run information.
    XtX_total = np.zeros(
        (p_dim, p_dim) if grams is None else (V, p_dim, p_dim), dtype=np.float64
    )
    rss_total = np.zeros(V, dtype=np.float64)
    dfres_total = 0.0

    for i, (r, proj) in enumerate(zip(results, projections, strict=True)):
        XtX_cached = getattr(proj, "XtX", None)
        if grams is not None:
            XtX_r = grams[i]
        elif XtX_cached is not None:
            XtX_r = np.asarray(XtX_cached, dtype=np.float64)
        else:
            try:
                XtX_r = np.linalg.inv(proj.XtXinv)
            except np.linalg.LinAlgError:
                XtX_r = np.linalg.pinv(proj.XtXinv)

        XtX_total += XtX_r
        rss_total += r.rss
        dfres_total += r.dfres

    try:
        XtXinv_total = np.linalg.inv(XtX_total)
    except np.linalg.LinAlgError:
        XtXinv_total = np.linalg.pinv(XtX_total)

    sigma2_pooled = rss_total / dfres_total if dfres_total > 0 else np.full(V, np.nan)

    return {
        "betas": betas_pooled,
        "sigma": np.sqrt(sigma2_pooled),
        "dfres": dfres_total,
        "XtXinv": XtXinv_total,
    }
