"""Single-design (concatenated) OLS fitting engine.

The default ``RunwiseEngine`` fits each run independently and pools the
per-run betas via inverse-variance weighting. That recovers the same
betas as a single concatenated OLS for orthogonal block-diagonal
designs, but the residual variance is estimated per-run — which is
the wrong denominator for cross-run contrast t/F statistics.

The concat engine stacks the per-run design and data, solves the
system once, and returns the textbook ``dfres = n - rank`` single
variance estimate. Use it when contrast vectors span runs (e.g.
"is A's effect in run 1 different from A's effect in run 2?").
"""

from __future__ import annotations

from typing import Any

from ..engine import EngineResult, register_engine
from ..strategies import fit_concat
from ...model.config import FmriLmConfig


@register_engine
class ConcatEngine:
    """Single-design OLS engine for multi-run concatenated analyses."""

    name = "concat"

    def fit(
        self,
        model: Any,
        config: FmriLmConfig,
        **kwargs: Any,
    ) -> EngineResult:
        raw = fit_concat(model, config, **kwargs)
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
        if not hasattr(model, "dataset"):
            raise ValueError("Model must have a 'dataset' attribute")
        if not hasattr(model, "design_matrix_array"):
            raise ValueError("Model must provide 'design_matrix_array()'")
