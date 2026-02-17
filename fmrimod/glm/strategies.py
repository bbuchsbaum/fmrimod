"""Fitting strategies for fMRI GLMs.

Implements runwise and chunkwise strategies for processing multi-run
fMRI datasets.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .solver import Projection, fast_preproject, fast_lm_matrix, LmResult
from .preprocess import (
    apply_censoring,
    apply_volume_weights,
    compute_dvars,
    dvars_weights,
    extract_nuisance_timeseries,
    soft_subspace_projection,
)
from ..model.config import FmriLmConfig


@contextmanager
def _maybe_limit_blas_threads(blas_threads: Optional[int]):
    """Limit BLAS threads within a context when threadpoolctl is available."""
    if blas_threads is None:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]
    except Exception:
        yield
        return
    with threadpool_limits(limits=int(blas_threads)):
        yield


def _projection_cache_key(
    X: NDArray[np.float64],
    compute_dtype: object,
) -> tuple:
    """Build a stable cache key for a design matrix projection."""
    Xc = np.ascontiguousarray(X, dtype=np.dtype(compute_dtype))
    h = hashlib.blake2b(digest_size=16)
    h.update(memoryview(Xc).cast("B"))
    return (Xc.shape, Xc.dtype.str, h.digest())


def _fit_one_run(
    model: object,
    config: FmriLmConfig,
    run_idx: int,
    needs_residuals: bool,
    compute_dtype: object,
) -> tuple[int, LmResult, Projection, Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """Fit one run and return all run-local outputs."""
    Y_r = model.dataset.get_data(run_idx)  # type: ignore[attr-defined]
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
    projection_cache: Optional[Dict[tuple, Projection]] = None,
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
    X_fit, Y_fit = _prepare_run_matrices(
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
        cache_key = _projection_cache_key(X_fit, compute_dtype)
        proj = projection_cache.get(cache_key)
    if proj is None:
        proj = fast_preproject(X_fit, compute_dtype=compute_dtype)
        if projection_cache is not None and cache_key is not None:
            projection_cache[cache_key] = proj
    result = fast_lm_matrix(
        X_fit,
        Y_fit,
        proj,
        return_fitted=return_fitted,
        compute_dtype=compute_dtype,
    )

    return result, proj, X_fit, Y_fit


def _prepare_run_matrices(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    config: FmriLmConfig,
    *,
    censor: Optional[NDArray[np.bool_]] = None,
    dataset: Optional[object] = None,
    run: Optional[int] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply censoring/weighting/subspace preprocessing for one run."""
    X_fit = X
    Y_fit = Y

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
                and hasattr(dataset, "n_timepoints")
                and weights.shape[0] != X.shape[0]
            ):
                run_lengths = [int(v) for v in getattr(dataset, "n_timepoints")]
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
                and hasattr(dataset, "n_timepoints")
                and nuisance.shape[0] != X.shape[0]
            ):
                run_lengths = [int(v) for v in getattr(dataset, "n_timepoints")]
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

            X_fit, Y_fit = soft_subspace_projection(
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
            X_fit, Y_fit = soft_subspace_projection(
                X_fit,
                Y_fit,
                nuisance,
                config.soft_subspace.lam,
            )

    return X_fit, Y_fit


def fit_runwise(
    model: object,  # FmriModel
    config: FmriLmConfig,
    n_jobs: int = 1,
    blas_threads: Optional[int] = None,
    compute_dtype: object = np.float64,
    cache_projections: bool = False,
) -> Dict:
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
    run_residuals: Optional[List[Optional[NDArray]]] = (
        [None] * n_runs if needs_residuals else None
    )
    run_X: Optional[List[Optional[NDArray]]] = (
        [None] * n_runs if needs_residuals else None
    )

    if n_workers == 1:
        proj_cache: Optional[Dict[tuple, Projection]] = {} if cache_projections else None
        for r in range(n_runs):
            # Get per-run data and design
            Y_r = model.dataset.get_data(r)  # type: ignore[attr-defined]
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
                    run_idx, result, proj, residual, X_used = fut.result()
                    run_results[run_idx] = result
                    run_projections[run_idx] = proj
                    if needs_residuals:
                        assert run_X is not None
                        assert run_residuals is not None
                        assert residual is not None
                        assert X_used is not None
                        run_X[run_idx] = X_used
                        run_residuals[run_idx] = residual

    # Narrow optional element types after fill.
    run_results_typed = [r for r in run_results if r is not None]
    run_proj_typed = [p for p in run_projections if p is not None]
    run_residuals_typed: Optional[List[NDArray]] = None
    run_x_typed: Optional[List[NDArray]] = None
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

    betas = np.empty((p_dim, n_voxels), dtype=np.float64)
    rss = np.empty(n_voxels, dtype=np.float64)
    sigma2 = np.empty(n_voxels, dtype=np.float64)

    def _solve_one(start: int, end: int) -> tuple[int, int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        y_chunk = Y_fit[:, start:end]
        res = fast_lm_matrix(
            X_fit,
            y_chunk,
            proj,
            return_fitted=False,
            compute_dtype=compute_dtype,
        )
        return start, end, res.betas, res.rss, res.sigma2

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
) -> Dict:
    """Fit GLM in voxel chunks (fmrireg-style chunkwise strategy).

    Notes
    -----
    This strategy supports the same per-run OLS preprocessing as
    :func:`fit_run_ols` (censoring, volume weights, soft-subspace).
    AR/robust post-processing remains runwise-only.
    """
    if config.ar.enabled:
        raise NotImplementedError("chunkwise engine does not yet support AR modeling")
    if config.robust.enabled:
        raise NotImplementedError("chunkwise engine does not yet support robust fitting")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    n_runs = model.n_runs  # type: ignore[attr-defined]
    n_workers = max(1, min(int(n_jobs), int(n_runs)))
    run_results: List[Optional[LmResult]] = [None] * n_runs
    run_projections: List[Optional[Projection]] = [None] * n_runs
    proj_cache: Optional[Dict[tuple, Projection]] = {} if (cache_projections and n_workers == 1) else None

    def _fit_run(r: int) -> tuple[int, LmResult, Projection]:
        Y_r = model.dataset.get_data(r)  # type: ignore[attr-defined]
        X_r = model.design_matrix_array(run=r)  # type: ignore[attr-defined]

        dataset = model.dataset  # type: ignore[attr-defined]
        censor_r = None
        if hasattr(dataset, "get_censor"):
            censor_r = dataset.get_censor(r)

        X_fit, Y_fit = _prepare_run_matrices(
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
            cache_key = _projection_cache_key(X_fit, compute_dtype)
            proj = proj_cache.get(cache_key)
        if proj is None:
            proj = fast_preproject(X_fit, compute_dtype=compute_dtype)
            if proj_cache is not None and cache_key is not None:
                proj_cache[cache_key] = proj

        result = _fit_chunked_lm(
            X_fit,
            Y_fit,
            proj,
            chunk_size=chunk_size,
            n_jobs=1,
            blas_threads=blas_threads,
            compute_dtype=compute_dtype,
        )
        return r, result, proj

    if n_workers == 1:
        for r in range(n_runs):
            run_idx, result, proj = _fit_run(r)
            run_results[run_idx] = result
            run_projections[run_idx] = proj
    else:
        with _maybe_limit_blas_threads(blas_threads):
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
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
) -> Dict:
    """Pool per-run OLS results via fixed-effects meta-analysis.

    Uses inverse-variance weighting to combine betas across runs.

    Parameters
    ----------
    results : list of LmResult
        Per-run regression results.
    projections : list of Projection
        Per-run projections.

    Returns
    -------
    dict
        Pooled ``betas``, ``sigma``, ``dfres``, ``XtXinv``.
    """
    if len(results) == 1:
        r = results[0]
        p = projections[0]
        return {
            "betas": r.betas,
            "sigma": np.sqrt(r.sigma2),
            "dfres": r.dfres,
            "XtXinv": p.XtXinv,
        }

    p_dim = results[0].betas.shape[0]
    V = results[0].betas.shape[1]

    # Inverse-variance beta pooling (streaming; avoids large temporary stacks).
    # se_{r,j,v}^2 = sigma2_{r,v} * XtXinv_{r,j,j}
    wsum = np.zeros((p_dim, V), dtype=np.float64)
    wbeta = np.zeros((p_dim, V), dtype=np.float64)
    beta_mean = np.zeros((p_dim, V), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for r, proj in zip(results, projections):
        beta_mean += r.betas
        diag_xtxinv = np.maximum(np.diag(proj.XtXinv), 0.0)[:, np.newaxis]  # (p,1)
        se2 = diag_xtxinv * r.sigma2[np.newaxis, :]  # (p,V)
        w = np.where(se2 > eps, 1.0 / se2, 0.0)
        wsum += w
        wbeta += w * r.betas

    beta_mean /= float(len(results))
    betas_pooled = np.divide(
        wbeta,
        wsum,
        out=beta_mean,
        where=wsum > 0.0,
    )

    # Keep a pooled XtXinv for downstream contrasts/SE interfaces.
    # This follows the prior behavior based on summed per-run information.
    XtX_total = np.zeros((p_dim, p_dim))
    rss_total = np.zeros(V)
    dfres_total = 0.0

    for r, proj in zip(results, projections):
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
