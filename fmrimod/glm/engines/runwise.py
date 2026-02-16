"""Run-wise OLS fitting engine.

The default engine: fits each run independently, then pools results
via fixed-effects meta-analysis.
"""

from __future__ import annotations

from typing import Any

from ..engine import EngineResult, register_engine
from ..strategies import fit_runwise
from ...model.config import FmriLmConfig


@register_engine
class RunwiseEngine:
    """Default engine: per-run OLS with meta-analytic pooling.

    This wraps :func:`~fmrimod.glm.strategies.fit_runwise` behind the
    :class:`~fmrimod.glm.engine.FittingEngine` protocol.
    """

    name = "runwise"

    def fit(
        self,
        model: Any,
        config: FmriLmConfig,
        **kwargs: Any,
    ) -> EngineResult:
        raw = fit_runwise(model, config)
        return EngineResult(
            betas=raw["betas"],
            sigma=raw["sigma"],
            dfres=raw["dfres"],
            XtXinv=raw["XtXinv"],
            projections=raw.get("projections"),
            run_results=raw.get("run_results"),
            residuals=raw.get("residuals"),
            run_X=raw.get("run_X"),
        )

    def preflight(self, model: Any, config: FmriLmConfig) -> None:
        """Validate that the model has data and a design matrix."""
        if not hasattr(model, "dataset"):
            raise ValueError("Model must have a 'dataset' attribute")
        if not hasattr(model, "design_matrix_array"):
            raise ValueError("Model must provide 'design_matrix_array()'")
