"""Chunkwise OLS fitting engine.

Processes voxel columns in chunks while sharing the same run design
projection to reduce peak memory and improve throughput on large datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..fmri_lm import FmriModelLike

from ...model.config import FmriLmConfig
from ..engine import EngineResult, register_engine
from ..strategies import fit_chunkwise


@register_engine
class ChunkwiseEngine:
    """Chunkwise engine: per-run chunked OLS + fixed-effects pooling."""

    name = "chunkwise"

    def fit(
        self,
        model: "FmriModelLike",
        config: FmriLmConfig,
        **kwargs: object,
    ) -> EngineResult:
        raw = fit_chunkwise(model, config, **kwargs)
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

    def preflight(self, model: "FmriModelLike", config: FmriLmConfig) -> None:
        """Validate that the model has data and a design matrix."""
        if not hasattr(model, "dataset"):
            raise ValueError("Model must have a 'dataset' attribute")
        if not hasattr(model, "design_matrix_array"):
            raise ValueError("Model must provide 'design_matrix_array()'")

