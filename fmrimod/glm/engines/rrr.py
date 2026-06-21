"""Reduced-rank regression fitting engine."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from ...dataset.data_access import get_run_data
from ...lowrank.rrr import ReducedRankConfig, ReducedRankResult, fit_reduced_rank
from ...model.config import FmriLmConfig
from ..engine import EngineResult, register_engine
from ..solver import fast_preproject

if TYPE_CHECKING:
    from ..fmri_lm import FmriModelLike


@register_engine
@register_engine(name="rrr_gls")
class ReducedRankEngine:
    """Reduced-rank GLM engine.

    By default, event-related columns are rank-constrained when the model
    exposes event-column metadata. Realized matrices and other minimal model
    objects fall back to constraining all columns.
    """

    name = "reduced_rank"

    def fit(
        self,
        model: "FmriModelLike",
        config: FmriLmConfig,
        *,
        rank: Optional[int] = None,
        rank_mode: str = "fixed",
        energy_keep: float = 0.99,
        rss_budget: Optional[float] = None,
        ridge: float = 0.0,
        se_mode: str = "conditional",
        bootstrap_n: int = 200,
        bootstrap_block_size: int = 1,
        bootstrap_seed: Optional[int] = None,
        target: str = "event",
        target_columns: Optional[Sequence[int | str]] = None,
        **kwargs: object,
    ) -> EngineResult:
        ncomp = kwargs.pop("ncomp", None)
        if rank is None and ncomp is not None:
            if isinstance(ncomp, bool) or not isinstance(ncomp, int):
                raise ValueError("ncomp must be a positive integer")
            rank = int(ncomp)
        if kwargs:
            extras = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected reduced-rank engine argument(s): {extras}")

        rr_config = ReducedRankConfig(
            rank=rank,
            rank_mode=cast(Any, rank_mode),
            energy_keep=energy_keep,
            rss_budget=rss_budget,
            ridge=ridge,
            se_mode=cast(Any, se_mode),
            bootstrap_n=bootstrap_n,
            bootstrap_block_size=bootstrap_block_size,
            bootstrap_seed=bootstrap_seed,
        )

        n_runs = model.n_runs
        if n_runs == 1:
            return self._fit_single(model, rr_config, target, target_columns)

        return self._fit_multirun(model, rr_config, target, target_columns)

    def _fit_single(
        self,
        model: "FmriModelLike",
        rr_config: ReducedRankConfig,
        target: str,
        target_columns: Optional[Sequence[int | str]],
    ) -> EngineResult:
        X = model.design_matrix_array(run=0)
        Y = get_run_data(cast(Any, model.dataset), 0)
        columns = self._resolve_target_columns(model, 0, X, target, target_columns)
        result = fit_reduced_rank(X, Y, rr_config, target_columns=columns)
        proj = fast_preproject(X)
        return EngineResult(
            betas=result.betas,
            sigma=np.sqrt(result.sigma2),
            dfres=result.dfres,
            XtXinv=proj.XtXinv,
            extra={
                "reduced_rank_config": rr_config,
                "reduced_rank_result": result,
            },
        )

    def _fit_multirun(
        self,
        model: "FmriModelLike",
        rr_config: ReducedRankConfig,
        target: str,
        target_columns: Optional[Sequence[int | str]],
    ) -> EngineResult:
        XtX_total: Optional[NDArray[np.float64]] = None
        XtXB_total: Optional[NDArray[np.float64]] = None
        rss_total: Optional[NDArray[np.float64]] = None
        dfres_total = 0.0
        run_results: list[ReducedRankResult] = []

        for run in range(model.n_runs):
            X_r = model.design_matrix_array(run=run)
            Y_r = get_run_data(cast(Any, model.dataset), run)
            run_config = rr_config
            if rr_config.bootstrap_seed is not None:
                run_config = replace(
                    rr_config,
                    bootstrap_seed=rr_config.bootstrap_seed + run,
                )
            columns = self._resolve_target_columns(
                model,
                run,
                X_r,
                target,
                target_columns,
            )
            result_r = fit_reduced_rank(X_r, Y_r, run_config, target_columns=columns)
            run_results.append(result_r)

            p, n_voxels = result_r.betas.shape
            if XtX_total is None:
                XtX_total = np.zeros((p, p), dtype=np.float64)
                XtXB_total = np.zeros((p, n_voxels), dtype=np.float64)
                rss_total = np.zeros(n_voxels, dtype=np.float64)

            assert (
                XtX_total is not None
                and XtXB_total is not None
                and rss_total is not None
            )
            proj_r = fast_preproject(X_r)
            try:
                XtX_r = np.linalg.inv(proj_r.XtXinv)
            except np.linalg.LinAlgError:
                XtX_r = np.linalg.pinv(proj_r.XtXinv)
            XtX_total += XtX_r
            XtXB_total += XtX_r @ result_r.betas
            rss_total += result_r.rss
            dfres_total += result_r.dfres

        assert (
            XtX_total is not None
            and XtXB_total is not None
            and rss_total is not None
        )
        try:
            XtXinv_total = np.linalg.inv(XtX_total)
        except np.linalg.LinAlgError:
            XtXinv_total = np.linalg.pinv(XtX_total)

        betas_pooled = XtXinv_total @ XtXB_total
        sigma_pooled = np.sqrt(rss_total / max(dfres_total, 1.0))
        return EngineResult(
            betas=betas_pooled,
            sigma=sigma_pooled,
            dfres=dfres_total,
            XtXinv=XtXinv_total,
            extra={
                "reduced_rank_config": rr_config,
                "reduced_rank_run_results": tuple(run_results),
            },
        )

    def _resolve_target_columns(
        self,
        model: "FmriModelLike",
        run: int,
        X: NDArray[np.float64],
        target: str,
        target_columns: Optional[Sequence[int | str]],
    ) -> tuple[int, ...] | None:
        if target_columns is not None:
            return _resolve_explicit_columns(model, run, X.shape[1], target_columns)
        if target == "all":
            return None
        if target != "event":
            raise ValueError("target must be 'event' or 'all'")

        event_indices = getattr(model, "event_column_indices", None)
        if event_indices is not None:
            out = tuple(int(i) for i in np.asarray(event_indices, dtype=np.intp))
            if out:
                return out

        n_event_columns = getattr(model, "n_event_columns", None)
        if isinstance(n_event_columns, int) and n_event_columns > 0:
            return tuple(range(n_event_columns))

        return None

    def preflight(self, model: "FmriModelLike", config: FmriLmConfig) -> None:
        if not hasattr(model, "dataset"):
            raise ValueError("Model must have a 'dataset' attribute")
        if not hasattr(model, "design_matrix_array"):
            raise ValueError("Model must provide 'design_matrix_array()'")


def _resolve_explicit_columns(
    model: "FmriModelLike",
    run: int,
    n_columns: int,
    target_columns: Sequence[int | str],
) -> tuple[int, ...]:
    if not target_columns:
        raise ValueError("target_columns must contain at least one column")

    if all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in target_columns
    ):
        return _validate_integer_columns(
            cast("Sequence[int]", target_columns),
            n_columns,
        )

    names = _design_column_names(model, run)
    if names is None:
        raise ValueError("named target_columns require model.design_matrix() columns")
    name_to_index = {name: idx for idx, name in enumerate(names)}

    out: list[int] = []
    for value in target_columns:
        if isinstance(value, bool):
            raise ValueError("target_columns must contain integers or strings")
        if isinstance(value, int):
            out.append(int(value))
            continue
        name = str(value)
        if name not in name_to_index:
            raise ValueError(f"target column {name!r} was not found")
        out.append(name_to_index[name])
    return _validate_integer_columns(out, n_columns)


def _validate_integer_columns(
    target_columns: Sequence[int],
    n_columns: int,
) -> tuple[int, ...]:
    out: list[int] = []
    for value in target_columns:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("target_columns must contain integers or strings")
        idx = int(value)
        if idx < 0 or idx >= n_columns:
            raise IndexError(f"target column index {idx} is out of range")
        out.append(idx)
    if len(set(out)) != len(out):
        raise ValueError("target_columns must not contain duplicates")
    return tuple(out)


def _design_column_names(
    model: "FmriModelLike",
    run: int,
) -> tuple[str, ...] | None:
    design_matrix = getattr(model, "design_matrix", None)
    if callable(design_matrix):
        frame = design_matrix(run=run)
        columns = getattr(frame, "columns", None)
        if columns is not None:
            return tuple(str(c) for c in columns)

    design_columns = getattr(model, "design_columns", None)
    if callable(design_columns):
        columns = design_columns()
        names = getattr(columns, "names", None)
        if names is not None:
            return tuple(str(c) for c in names)
        try:
            return tuple(str(c) for c in columns)
        except TypeError:
            return None

    return None
