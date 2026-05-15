"""Matrix-first GLM entry points.

Typed Python entry points for fitting a GLM from (X, Y) matrices or from
OLS sufficient statistics, returning a full :class:`~fmrimod.glm.FmriLm`
result. These were historically housed in :mod:`fmrimod.glm.compat`
during the R-name migration; they are typed Python values, not
R-name shims, and live here as the matrix-first leaf of the spec/dataset
→ glm seam.

The ``_MatrixDataset`` / ``_MatrixModel`` helpers below adapt a raw
(X, Y) pair to the model-like interface ``fmri_lm`` consumes; they are
shared by :func:`fit_glm_on_transformed_series` in :mod:`fmrimod.glm.compat`
and intentionally remain private (leading underscore) until a typed seam
claims them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ..design.columns import DesignColumn, DesignColumns
from ..model.config import FmriLmConfig
from .fmri_lm import FmriLm, fmri_lm


class _MatrixDataset:
    def __init__(self, Y: NDArray[np.float64]) -> None:
        self._Y = Y
        self.n_timepoints = [int(Y.shape[0])]
        self.n_runs = 1

    def get_data(self, run: int = 0) -> NDArray[np.float64]:
        if run != 0:
            raise IndexError("matrix dataset has one run")
        return self._Y

    def get_censor(self, run: int = 0) -> None:
        return None


class _MatrixModel:
    def __init__(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        source: object = None,
    ) -> None:
        self._X = X
        self.dataset = _MatrixDataset(Y)
        self.n_runs = 1
        self._source = source

    def design_matrix_array(self, run: int = 0) -> NDArray[np.float64]:
        if run != 0:
            raise IndexError("matrix model has one run")
        return self._X

    def design_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(self._X, columns=self._column_names())

    def design_columns(self) -> DesignColumns:
        source_columns = getattr(self._source, "design_columns", None)
        if callable(source_columns):
            columns = source_columns()
            if isinstance(columns, DesignColumns):
                return columns
        return DesignColumns(
            tuple(
                DesignColumn(
                    name=name,
                    index=index,
                    role="matrix",
                    model_source="matrix",
                    provenance={
                        "role": "inferred",
                        "term": "missing",
                        "condition": "missing",
                        "level": "missing",
                        "basis_ix": "missing",
                        "basis_name": "missing",
                        "basis_total": "missing",
                    },
                )
                for index, name in enumerate(self._column_names())
            )
        )

    def _column_names(self) -> list[str]:
        dm = _source_design_matrix(self._source)
        if hasattr(dm, "columns") and len(dm.columns) == self._X.shape[1]:
            return [str(c) for c in dm.columns]
        return [f"coef_{idx}" for idx in range(self._X.shape[1])]

    def contrast_weights(self) -> Dict[str, NDArray[np.float64]]:
        if self._source is not None and hasattr(self._source, "contrast_weights"):
            return self._source.contrast_weights()
        return {}


def _source_design_matrix(source: object) -> object:
    if isinstance(source, pd.DataFrame):
        return source
    design_matrix = getattr(source, "design_matrix", None)
    if callable(design_matrix):
        try:
            return design_matrix()
        except TypeError:
            return design_matrix(run=0)
    return None


def _design_matrix_from_model(model: object) -> NDArray[np.float64]:
    if hasattr(model, "design_matrix_array"):
        return np.asarray(model.design_matrix_array(run=0), dtype=np.float64)
    if hasattr(model, "design_matrix"):
        return np.asarray(model.design_matrix(), dtype=np.float64)
    raise TypeError("model must provide design_matrix_array() or design_matrix()")


def fit_glm_from_matrix(
    X: ArrayLike,
    Y: ArrayLike,
    model: object = None,
    cfg: Optional[FmriLmConfig] = None,
    strategy: str = "external",
    engine: str = "runwise",
) -> FmriLm:
    """Fit a matrix-design GLM and return a full ``FmriLm`` result.

    Unlike :func:`fit_glm_from_suffstats`, this path keeps raw ``X`` and ``Y``
    available to the solver, so cancellation-prone RSS values can be recovered
    from explicit residuals.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr[:, np.newaxis]
    if X_arr.ndim != 2 or Y_arr.ndim != 2:
        raise ValueError("X and Y must be 2-D matrices")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    source = model if model is not None else X
    fit = fmri_lm(
        _MatrixModel(X_arr, Y_arr, source=source),
        cfg or FmriLmConfig(),
        engine=engine,
    )
    fit.strategy = strategy
    fit.engine = engine
    return fit


def fit_glm_from_suffstats(
    model: object,
    XtX: ArrayLike,
    XtS: ArrayLike,
    StS: ArrayLike,
    df: float,
    cfg: Optional[FmriLmConfig] = None,
    dataset: Optional[object] = None,
    strategy: str = "external",
    engine: str = "suffstats",
) -> FmriLm:
    """Build an ``FmriLm`` result from OLS sufficient statistics."""
    del dataset
    XtX_arr = np.asarray(XtX, dtype=np.float64)
    XtS_arr = np.asarray(XtS, dtype=np.float64)
    StS_arr = np.ravel(np.asarray(StS, dtype=np.float64))
    XtXinv = np.linalg.pinv(XtX_arr)
    betas = XtXinv @ XtS_arr
    sse = np.maximum(StS_arr - np.sum(betas * XtS_arr, axis=0), 0.0)
    sigma = np.sqrt(np.maximum(sse / float(df), np.finfo(np.float64).eps))
    fit = FmriLm(
        betas=betas,
        sigma=sigma,
        residual_df=float(df),
        XtXinv=XtXinv,
        model=model,
        config=cfg or FmriLmConfig(),
    )
    fit.strategy = strategy
    fit.engine = engine
    return fit


__all__ = [
    "fit_glm_from_matrix",
    "fit_glm_from_suffstats",
]
