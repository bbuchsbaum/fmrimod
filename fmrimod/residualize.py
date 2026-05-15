"""Residualization methods using QR decomposition."""
from __future__ import annotations

from functools import singledispatch
from typing import Any, Optional, Protocol, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class _DesignSource(Protocol):
    """Duck-typed view of EventModel / BaselineModel for residualize dispatch.

    Names the two attributes the model-specific dispatch implementations
    read off their input. Kept local so the typed seam does not have to
    import the concrete model classes at the top level.
    """

    design_matrix: Any
    column_names: Sequence[str]


@singledispatch
def residualize(
    x: object,
    data: object,
    cols: Optional[Sequence[Union[int, str]]] = None,
) -> NDArray[Any]:
    """Residualize data by projecting out a design matrix.

    Computes ordinary-least-squares residuals:
    ``residuals = data - Q @ Q.T @ data`` where ``Q`` comes from
    the thin QR decomposition of the design matrix.

    This is a generic function dispatching on the type of ``x``.

    Parameters
    ----------
    x : numpy.ndarray, pandas.DataFrame, EventModel, or BaselineModel
        Design matrix or model to project out. When an ``EventModel``
        or ``BaselineModel`` is passed, the design matrix is extracted
        automatically.
    data : array-like
        Observations to residualize. Shape ``(n_timepoints,)`` or
        ``(n_timepoints, n_signals)``.
    cols : list of int or list of str, optional
        Subset of design columns to use. String names are supported
        for DataFrame and EventModel inputs.

    Returns
    -------
    numpy.ndarray
        Residuals with the same shape as ``data``.

    Raises
    ------
    ValueError
        If the number of rows in ``x`` and ``data`` do not match.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
    >>> y = np.random.randn(100)
    >>> resid = residualize(X, y)
    >>> resid.shape
    (100, 1)

    See Also
    --------
    validate_contrasts : Validate contrast estimability.
    """
    raise NotImplementedError(f"residualize not implemented for {type(x)}")


def _residualize_with_matrix(X: object, Y: object) -> NDArray[Any]:
    """Compute OLS residuals of Y against design matrix X via QR.

    Parameters
    ----------
    X : numpy.ndarray
        Design matrix, shape ``(n, p)``.
    Y : numpy.ndarray
        Data matrix, shape ``(n, m)`` or ``(n,)``.

    Returns
    -------
    numpy.ndarray
        Residuals, shape ``(n, m)``.
    """
    X_arr: NDArray[Any] = np.asarray(X, dtype=float)
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    Y_arr: NDArray[Any] = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)

    if Y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError(
            f"Row mismatch: nrow(data)={Y_arr.shape[0]}, nrow(design)={X_arr.shape[0]}"
        )

    # QR decomposition and compute residuals
    Q, R = np.linalg.qr(X_arr, mode='reduced')
    # Residuals = Y - Q @ Q.T @ Y
    fitted = Q @ (Q.T @ Y_arr)
    return Y_arr - fitted


@residualize.register(np.ndarray)
def _residualize_matrix(
    x: NDArray[Any],
    data: object,
    cols: Optional[Sequence[Union[int, str]]] = None,
) -> NDArray[Any]:
    """Residualize data against a numpy array design matrix."""
    X = x
    if cols is not None:
        if isinstance(cols[0], str):
            raise ValueError("String column names not supported for numpy arrays")
        X = X[:, list(cols)]
    return _residualize_with_matrix(X, data)


@residualize.register(pd.DataFrame)
def _residualize_dataframe(
    x: pd.DataFrame,
    data: object,
    cols: Optional[Sequence[Union[int, str]]] = None,
) -> NDArray[Any]:
    """Residualize data against a DataFrame design matrix."""
    if cols is not None:
        X = x[list(cols)].values
    else:
        X = x.values
    return _residualize_with_matrix(X, data)


# Register EventModel and BaselineModel implementations
def _register_model_residualize() -> None:
    try:
        from .design.event_model import EventModel

        @residualize.register(EventModel)
        def _residualize_event_model(
            x: _DesignSource,
            data: object,
            cols: Optional[Sequence[Union[int, str]]] = None,
        ) -> NDArray[Any]:
            X = x.design_matrix
            if cols is not None:
                if isinstance(cols[0], str):
                    col_names = x.column_names
                    col_idx = [col_names.index(str(c)) for c in cols]
                    X = X[:, col_idx]
                else:
                    X = X[:, list(cols)]
            return _residualize_with_matrix(X, data)
    except ImportError:
        pass

    try:
        from .baseline.baseline_model import BaselineModel

        @residualize.register(BaselineModel)
        def _residualize_baseline_model(
            x: _DesignSource,
            data: object,
            cols: Optional[Sequence[Union[int, str]]] = None,
        ) -> NDArray[Any]:
            dm = x.design_matrix
            X = np.asarray(dm)
            if cols is not None:
                if isinstance(cols[0], str):
                    col_names = list(dm.columns) if isinstance(dm, pd.DataFrame) else [f"V{i}" for i in range(X.shape[1])]
                    col_idx = [col_names.index(str(c)) for c in cols]
                    X = X[:, col_idx]
                else:
                    X = X[:, list(cols)]
            return _residualize_with_matrix(X, data)
    except ImportError:
        pass

_register_model_residualize()
