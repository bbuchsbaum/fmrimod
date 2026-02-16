"""Sketch-based low-rank fitting engine.

Uses randomised sketching to reduce the temporal dimension before
solving, optionally combined with Nyström landmark extension for
spatial reduction.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..engine import EngineResult, register_engine
from ..solver import fast_preproject
from ...model.config import FmriLmConfig
from ...lowrank.engine import LowRankConfig, fit_sketched


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
        model: Any,
        config: FmriLmConfig,
        *,
        sketch_kind: str = "gaussian",
        sketch_ratio: float = 0.5,
        use_landmarks: bool = False,
        n_landmarks: int = 500,
        landmark_k: int = 6,
        landmark_method: str = "kmeans",
        ridge: float = 0.0,
        seed: Optional[int] = None,
        coords: Optional[NDArray[np.float64]] = None,
        **kwargs: Any,
    ) -> EngineResult:
        lr_config = LowRankConfig(
            sketch_kind=sketch_kind,
            sketch_ratio=sketch_ratio,
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
        model: Any,
        config: FmriLmConfig,
        lr_config: LowRankConfig,
        coords: Optional[NDArray[np.float64]],
    ) -> EngineResult:
        X = model.design_matrix_array(run=0)
        Y = model.dataset.get_data(0)
        result = fit_sketched(X, Y, lr_config, coords=coords)

        proj = fast_preproject(X)
        return EngineResult(
            betas=result.betas,
            sigma=np.sqrt(result.sigma2),
            dfres=result.dfres,
            XtXinv=proj.XtXinv,
            extra={"lowrank_config": lr_config},
        )

    def _fit_multirun(
        self,
        model: Any,
        config: FmriLmConfig,
        lr_config: LowRankConfig,
        coords: Optional[NDArray[np.float64]],
    ) -> EngineResult:
        """Pool sketch results across runs via inverse-variance weighting."""
        n_runs = model.n_runs
        p = None
        V = None

        XtX_total = None
        XtXB_total = None
        rss_total = None
        dfres_total = 0.0

        for r in range(n_runs):
            X_r = model.design_matrix_array(run=r)
            Y_r = model.dataset.get_data(r)
            result_r = fit_sketched(X_r, Y_r, lr_config, coords=coords)

            if p is None:
                p = result_r.betas.shape[0]
                V = result_r.betas.shape[1]
                XtX_total = np.zeros((p, p))
                XtXB_total = np.zeros((p, V))
                rss_total = np.zeros(V)

            proj_r = fast_preproject(X_r)
            try:
                XtX_r = np.linalg.inv(proj_r.XtXinv)
            except np.linalg.LinAlgError:
                XtX_r = np.linalg.pinv(proj_r.XtXinv)

            XtX_total += XtX_r
            XtXB_total += XtX_r @ result_r.betas
            rss_total += result_r.rss
            dfres_total += result_r.dfres

        try:
            XtXinv_total = np.linalg.inv(XtX_total)
        except np.linalg.LinAlgError:
            XtXinv_total = np.linalg.pinv(XtX_total)

        betas_pooled = XtXinv_total @ XtXB_total
        sigma_pooled = np.sqrt(rss_total / max(dfres_total, 1))

        return EngineResult(
            betas=betas_pooled,
            sigma=sigma_pooled,
            dfres=dfres_total,
            XtXinv=XtXinv_total,
            extra={"lowrank_config": lr_config},
        )

    def preflight(self, model: Any, config: FmriLmConfig) -> None:
        if not hasattr(model, "dataset"):
            raise ValueError("Model must have a 'dataset' attribute")
