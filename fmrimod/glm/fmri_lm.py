"""Main entry point for fMRI GLM fitting.

Provides ``fmri_lm()`` — the primary user-facing function — and the
:class:`FmriLm` result class that holds fitted coefficients, residual
information, and methods for computing contrasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..model.config import FmriLmConfig
from .solver import Projection
from .contrasts import (
    ContrastResult,
    contrast_t,
    contrast_f_vectorized,
)
from .effective_df import effective_df


@dataclass
class FmriLm:
    """Result of fitting a GLM to fMRI data.

    Holds beta coefficients, residual information, and pre-computed
    quantities needed for efficient contrast computation.

    Attributes
    ----------
    betas : NDArray
        Coefficient matrix, shape ``(p, V)`` where ``p`` is the number
        of design columns and ``V`` the number of voxels.
    sigma : NDArray
        Residual standard deviation per voxel, shape ``(V,)``.
    residual_df : float
        Residual degrees of freedom.
    XtXinv : NDArray
        ``(X'X)^{-1}`` matrix, shape ``(p, p)``.
    model : object
        The :class:`~fmrimod.model.FmriModel` that was fitted.
    config : FmriLmConfig
        Configuration used for fitting.
    contrasts : dict
        Dictionary of computed :class:`ContrastResult` objects.
    ar_params : NDArray or None
        Estimated AR parameters, shape ``(ar_order, V)`` or ``(ar_order,)``
        for global estimation.
    robust_weights : NDArray or None
        IRLS weights from robust fitting, shape ``(n, V)``.
    run_results : list
        Per-run fitting results (for diagnostics).
    projections : list
        Per-run :class:`Projection` objects.
    """

    betas: NDArray[np.float64]
    sigma: NDArray[np.float64]
    residual_df: float
    XtXinv: NDArray[np.float64]
    model: object
    config: FmriLmConfig
    contrasts: Dict[str, ContrastResult] = field(default_factory=dict)
    ar_params: Optional[NDArray[np.float64]] = None
    robust_weights: Optional[NDArray[np.float64]] = None
    run_results: Optional[List] = None
    projections: Optional[List[Projection]] = None

    # -- Accessors --

    def coef(self) -> NDArray[np.float64]:
        """Return the coefficient matrix ``(p, V)``."""
        return self.betas

    def se(self) -> NDArray[np.float64]:
        """Return standard errors for each coefficient, shape ``(p, V)``.

        ``SE_{j,v} = sigma_v * sqrt(XtXinv_{j,j})``
        """
        diag_XtXinv = np.diag(self.XtXinv)
        return self.sigma[np.newaxis, :] * np.sqrt(
            np.maximum(diag_XtXinv, 0.0)
        )[:, np.newaxis]

    def tstat(self) -> NDArray[np.float64]:
        """Return t-statistics for each coefficient, shape ``(p, V)``."""
        se_vals = self.se()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(se_vals > 1e-15, self.betas / se_vals, 0.0)

    @property
    def n_voxels(self) -> int:
        """Number of voxels."""
        return self.betas.shape[1]

    @property
    def n_coefficients(self) -> int:
        """Number of regression coefficients."""
        return self.betas.shape[0]

    # -- Contrast computation --

    def contrast(
        self,
        spec: Union[NDArray[np.float64], str, dict],
        name: Optional[str] = None,
    ) -> ContrastResult:
        """Compute a contrast on the fitted model.

        Parameters
        ----------
        spec : NDArray or str or dict
            Contrast specification.  Can be:
            - A 1-D vector for a t-contrast
            - A 2-D matrix for an F-contrast
            - A string name referring to a pre-defined contrast in
              the event model
            - A dict ``{"weights": array, "name": str}``
        name : str, optional
            Override contrast name.

        Returns
        -------
        ContrastResult
        """
        if isinstance(spec, str):
            # Look up from model's contrast weights
            cw = self.model.contrast_weights() if hasattr(self.model, "contrast_weights") else {}  # type: ignore[union-attr]
            if spec not in cw:
                raise KeyError(f"Unknown contrast name: {spec!r}")
            return self._compute_contrast(cw[spec], name=name or spec)

        if isinstance(spec, dict):
            if "weights" not in spec:
                raise ValueError("Contrast dict spec must contain 'weights'")
            weights = np.asarray(spec["weights"], dtype=np.float64)
            cname = name or spec.get("name", "contrast")
            return self._compute_contrast(weights, name=cname)

        weights = np.asarray(spec, dtype=np.float64)
        return self._compute_contrast(weights, name=name or "contrast")

    def _compute_contrast(
        self,
        weights: NDArray[np.float64],
        name: str,
    ) -> ContrastResult:
        """Dispatch to t or F contrast based on weight dimensions."""
        weights = np.atleast_1d(weights)
        if weights.ndim == 1:
            result = contrast_t(
                weights, self.betas, self.XtXinv, self.sigma,
                self.residual_df, name=name,
            )
        else:
            result = contrast_f_vectorized(
                weights, self.betas, self.XtXinv, self.sigma,
                self.residual_df, name=name,
            )
        self.contrasts[name] = result
        return result

    # -- Display --

    def __repr__(self) -> str:
        parts = [
            "FmriLm(",
            f"  n_coefficients={self.n_coefficients},",
            f"  n_voxels={self.n_voxels},",
            f"  residual_df={self.residual_df:.1f},",
            f"  config={self.config!r},",
            f"  contrasts={list(self.contrasts.keys())},",
            ")",
        ]
        return "\n".join(parts)


def fmri_lm(
    model: object,
    config: Optional[FmriLmConfig] = None,
    engine: str = "runwise",
    **engine_kwargs,
) -> FmriLm:
    """Fit a GLM to fMRI data.

    This is the main user-facing entry point.  It takes an
    :class:`~fmrimod.model.FmriModel` and an :class:`FmriLmConfig`
    and returns a fitted :class:`FmriLm` result object.

    The *engine* parameter selects the fitting strategy.  Built-in
    engines are ``"runwise"`` (default OLS with meta-analytic pooling),
    ``"chunkwise"`` (voxel-chunked OLS), and ``"sketch"``
    (randomised low-rank solver).  Third-party engines
    can be registered via Python entry points or
    :func:`~fmrimod.glm.engine.register_engine`.

    Parameters
    ----------
    model : FmriModel
        The model specification (design + data).
    config : FmriLmConfig, optional
        Fitting configuration.  Defaults to plain OLS.
    engine : str
        Name of the fitting engine (default ``"runwise"``).
        See :func:`~fmrimod.glm.engine.list_engines` for available
        engines.
    **engine_kwargs
        Additional keyword arguments forwarded to the engine's
        ``fit()`` method.

    Returns
    -------
    FmriLm
        Fitted model with coefficients, residuals, and contrast methods.

    Examples
    --------
    >>> from fmrimod.model import FmriModel, FmriLmConfig
    >>> from fmrimod.glm import fmri_lm
    >>>
    >>> result = fmri_lm(fmri_model)
    >>> result.betas.shape  # (n_columns, n_voxels)
    >>> t_result = result.contrast(np.array([1, -1, 0, 0]))

    Using the sketch engine::

        >>> result = fmri_lm(model, engine="sketch", sketch_ratio=0.5)

    Using the chunkwise engine::

        >>> result = fmri_lm(model, engine="chunkwise", chunk_size=5000)
    """
    from .engine import EngineResult, get_engine

    if config is None:
        config = FmriLmConfig()

    # Resolve and run the engine
    eng = get_engine(engine)
    eng.preflight(model, config)
    fit_result: EngineResult = eng.fit(model, config, **engine_kwargs)

    # Handle AR modeling if configured
    ar_params = fit_result.ar_params
    if ar_params is None and config.ar.enabled:
        from ..ar.integration import iterative_gls

        # Build a dict compatible with the legacy interface
        legacy = _engine_result_to_dict(fit_result)
        legacy, ar_params = iterative_gls(model, config, legacy)
        fit_result = _dict_to_engine_result(legacy, fit_result)

    # Handle robust fitting if configured
    robust_weights = fit_result.robust_weights
    if robust_weights is None and config.robust.enabled:
        from ..robust.irls import robust_refit

        legacy = _engine_result_to_dict(fit_result)
        legacy, robust_weights = robust_refit(model, config, legacy)
        fit_result = _dict_to_engine_result(legacy, fit_result)

    return FmriLm(
        betas=fit_result.betas,
        sigma=fit_result.sigma,
        residual_df=fit_result.dfres,
        XtXinv=fit_result.XtXinv,
        model=model,
        config=config,
        ar_params=ar_params,
        robust_weights=robust_weights,
        run_results=fit_result.run_results,
        projections=fit_result.projections,
    )


def _engine_result_to_dict(er: object) -> Dict:
    """Convert an EngineResult to the legacy dict format."""
    return {
        "betas": er.betas,  # type: ignore[attr-defined]
        "sigma": er.sigma,  # type: ignore[attr-defined]
        "dfres": er.dfres,  # type: ignore[attr-defined]
        "XtXinv": er.XtXinv,  # type: ignore[attr-defined]
        "projections": er.projections,  # type: ignore[attr-defined]
        "run_results": er.run_results,  # type: ignore[attr-defined]
        "residuals": er.residuals,  # type: ignore[attr-defined]
        "run_X": er.run_X,  # type: ignore[attr-defined]
    }


def _dict_to_engine_result(d: Dict, original: object) -> object:
    """Update an EngineResult from a legacy dict (after AR/robust)."""
    from .engine import EngineResult

    return EngineResult(
        betas=d["betas"],
        sigma=d["sigma"],
        dfres=d["dfres"],
        XtXinv=d["XtXinv"],
        projections=d.get("projections"),
        run_results=d.get("run_results"),
        residuals=d.get("residuals"),
        run_X=d.get("run_X"),
        ar_params=getattr(original, "ar_params", None),
        robust_weights=getattr(original, "robust_weights", None),
        extra=getattr(original, "extra", {}),
    )
