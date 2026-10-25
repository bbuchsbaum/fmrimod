"""Sketch-based low-rank fitting engine.

Uses randomised sketching to reduce the temporal dimension before
solving, optionally combined with Nyström landmark extension for
spatial reduction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from fmrimod.lowrank.engine import SketchLmResult

    from ..fmri_lm import FmriModelLike

import numpy as np
from numpy.typing import NDArray

from ...dataset.data_access import get_run_data
from ...lowrank.engine import LowRankConfig, fit_sketched
from ...model.config import FmriLmConfig
from ..engine import EngineResult, register_engine
from ..solver import fast_preproject
from ._capability import require_unweighted_ols


@register_engine
class SketchEngine:
    """Low-rank sketch-based GLM engine.

    Keyword arguments are forwarded to :class:`~fmrimod.lowrank.engine.LowRankConfig`.

    Examples
    --------
    >>> result = fmri_lm(model, config, engine="sketch",
    ...                  sketch_kind="srht", sketch_ratio=0.5)
    """

    name = "sketch"

    def fit(
        self,
        model: "FmriModelLike",
        config: FmriLmConfig,
        *,
        sketch_kind: str = "gaussian",
        sketch_ratio: float = 0.5,
        sketch_size: Optional[int] = None,
        use_landmarks: bool = False,
        n_landmarks: int = 500,
        landmark_k: int = 6,
        landmark_method: str = "kmeans",
        ridge: float = 0.0,
        seed: Optional[int] = None,
        coords: Optional[NDArray[np.float64]] = None,
        method: Optional[str] = None,
        m: Optional[int] = None,
        iters: int = 0,
        **kwargs: object,
    ) -> EngineResult:
        if method is not None:
            sketch_kind = method
        if m is not None:
            sketch_size = int(m)
        if iters not in (0, None):
            raise NotImplementedError(
                "SketchEngine supports one-shot temporal sketching; "
                "iterative sketch refinements (iters > 0) are not implemented."
            )
        lr_config = LowRankConfig(
            sketch_kind=sketch_kind,
            sketch_ratio=sketch_ratio,
            sketch_size=sketch_size,
            use_landmarks=use_landmarks,
            n_landmarks=n_landmarks,
            landmark_k=landmark_k,
            landmark_method=landmark_method,
            ridge=ridge,
            seed=seed,
        )

        n_runs = model.n_runs
        if n_runs == 1:
            return self._fit_single(model, config, lr_config, coords)

        return self._fit_multirun(model, config, lr_config, coords)

    def _fit_single(
        self,
        model: "FmriModelLike",
        config: FmriLmConfig,
        lr_config: LowRankConfig,
        coords: Optional[NDArray[np.float64]],
    ) -> EngineResult:
        X = model.design_matrix_array(run=0)
        Y = get_run_data(cast(Any, model.dataset), 0)
        result = fit_sketched(X, Y, lr_config, coords=coords)

        return EngineResult(
            betas=result.betas,
            sigma=np.sqrt(result.sigma2),
            dfres=result.dfres,
            XtXinv=_cov_unscaled(result, X),
            extra={"lowrank_config": lr_config, "sketch": _sketch_info(result)},
        )

    def _fit_multirun(
        self,
        model: "FmriModelLike",
        config: FmriLmConfig,
        lr_config: LowRankConfig,
        coords: Optional[NDArray[np.float64]],
    ) -> EngineResult:
        """Pool sketch results across runs via inverse-variance weighting."""
        n_runs = model.n_runs
        p = None
        V = None

        XtX_total: Optional[NDArray[np.float64]] = None
        XtXB_total: Optional[NDArray[np.float64]] = None
        rss_total: Optional[NDArray[np.float64]] = None
        dfres_total = 0.0
        kappa_total = 0.0
        sketch_info = []

        for r in range(n_runs):
            X_r = model.design_matrix_array(run=r)
            Y_r = get_run_data(cast(Any, model.dataset), r)
            result_r = fit_sketched(X_r, Y_r, lr_config, coords=coords)

            if p is None:
                p = result_r.betas.shape[0]
                V = result_r.betas.shape[1]
                XtX_total = np.zeros((p, p))
                XtXB_total = np.zeros((p, V))
                rss_total = np.zeros(V)

            assert (
                XtX_total is not None
                and XtXB_total is not None
                and rss_total is not None
            )
            # Precision of this run's estimator (its sketch's conditional
            # covariance when sketched).
            cov_r = _cov_unscaled(result_r, X_r)
            try:
                XtX_r = np.linalg.inv(cov_r)
            except np.linalg.LinAlgError:
                XtX_r = np.linalg.pinv(cov_r)

            XtX_total += XtX_r
            XtXB_total += XtX_r @ result_r.betas
            rss_total += result_r.rss
            dfres_total += result_r.dfres
            kappa_total += (
                result_r.kappa if result_r.kappa is not None else result_r.dfres
            )
            sketch_info.append(_sketch_info(result_r))

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
        sigma_pooled = np.sqrt(rss_total / max(kappa_total, 1))

        return EngineResult(
            betas=betas_pooled,
            sigma=sigma_pooled,
            dfres=dfres_total,
            XtXinv=XtXinv_total,
            extra={"lowrank_config": lr_config, "sketch": sketch_info},
        )

    def preflight(self, model: "FmriModelLike", config: FmriLmConfig) -> None:
        if not hasattr(model, "dataset"):
            raise ValueError("Model must have a 'dataset' attribute")
        require_unweighted_ols(
            self.name,
            model,
            config,
            current_capability=("OLS on a sketched time axis",),
        )


def _cov_unscaled(result: SketchLmResult, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unscaled covariance of the estimator ``fit_sketched`` computed.

    A ridge solve has none of its own; it keeps the full-data ``(X'X)^-1``.
    """
    if result.cov_unscaled is not None:
        return result.cov_unscaled
    return fast_preproject(X).XtXinv


def _sketch_info(result: SketchLmResult) -> dict[str, object]:
    """Sketch metadata (fmrireg ``$sketch``): rows, ``tr(PK)`` and df."""
    return {"m": result.sketch_rows, "kappa": result.kappa, "df": result.dfres}
