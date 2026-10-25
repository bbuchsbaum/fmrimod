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

from collections.abc import Sequence
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ..design.columns import DesignColumn, DesignColumns
from ..model.config import FmriLmConfig
from .fmri_lm import FmriLm, fmri_lm


def _run_slices(run_lengths: Sequence[int]) -> list[slice]:
    stops = np.cumsum([0, *run_lengths])
    return [slice(int(a), int(b)) for a, b in zip(stops[:-1], stops[1:], strict=True)]


class _MatrixDataset:
    def __init__(
        self,
        Y: NDArray[np.float64],
        run_lengths: Sequence[int] | None = None,
    ) -> None:
        self._Y = Y
        lengths = [int(Y.shape[0])] if run_lengths is None else [int(n) for n in run_lengths]
        if sum(lengths) != Y.shape[0]:
            raise ValueError(
                f"run lengths {lengths} do not add up to the {Y.shape[0]} response rows"
            )
        self.n_timepoints = lengths
        self.run_lengths = lengths
        self.n_runs = len(lengths)
        self._slices = _run_slices(lengths)

    def get_data(self, run: int = 0) -> NDArray[np.float64]:
        if not 0 <= run < self.n_runs:
            raise IndexError(f"matrix dataset has {self.n_runs} run(s)")
        return self._Y[self._slices[run]]

    def get_censor(self, run: int = 0) -> None:
        return None


class _MatrixModel:
    def __init__(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        source: object = None,
        run_lengths: Sequence[int] | None = None,
    ) -> None:
        self._X = X
        self.dataset = _MatrixDataset(Y, run_lengths)
        self.n_runs = self.dataset.n_runs
        self._slices = self.dataset._slices
        self._source = source

    def design_matrix_array(self, run: int | None = 0) -> NDArray[np.float64]:
        """Run ``run``'s rows of the design (all rows when ``run`` is None)."""
        if run is None:
            return self._X
        if not 0 <= run < self.n_runs:
            raise IndexError(f"matrix model has {self.n_runs} run(s)")
        return self._X[self._slices[run]]

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
            return cast(Dict[str, NDArray[np.float64]], self._source.contrast_weights())
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
    return _design_and_runs_from_model(model)[0]


def _design_and_runs_from_model(
    model: object,
) -> tuple[NDArray[np.float64], list[int] | None]:
    """Full design of ``model`` and its per-run row counts.

    A multi-run model gives every run's design stacked, with the run
    lengths, so AR estimation and whitening restart at run boundaries as
    in ``fmri_lm`` (fmrireg d0f0aa1 ``run_indices``). A single-run model
    gives its design and ``None``.
    """
    if hasattr(model, "design_matrix_array"):
        n_runs = getattr(model, "n_runs", 1)
        if isinstance(n_runs, (int, np.integer)) and int(n_runs) > 1:
            blocks = [
                np.asarray(model.design_matrix_array(run=r), dtype=np.float64)
                for r in range(int(n_runs))
            ]
            return np.vstack(blocks), [int(b.shape[0]) for b in blocks]
        return np.asarray(model.design_matrix_array(run=0), dtype=np.float64), None
    if hasattr(model, "design_matrix"):
        return np.asarray(model.design_matrix(), dtype=np.float64), None
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
    cast(Any, fit).strategy = strategy
    cast(Any, fit).engine = engine
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
    sigma = np.sqrt(sse / float(df))
    fit = FmriLm(
        betas=betas,
        sigma=sigma,
        residual_df=float(df),
        XtXinv=XtXinv,
        model=cast(Any, model),
        config=cfg or FmriLmConfig(),
    )
    cast(Any, fit).strategy = strategy
    cast(Any, fit).engine = engine
    return fit


__all__ = [
    "fit_glm_from_matrix",
    "fit_glm_from_suffstats",
]
