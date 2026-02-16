"""Residualization methods using QR decomposition."""
import numpy as np
import pandas as pd
from functools import singledispatch


@singledispatch
def residualize(x, data, cols=None):
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


def _residualize_with_matrix(X, Y):
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
    X = np.asarray(X, dtype=float)
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"Row mismatch: nrow(data)={Y.shape[0]}, nrow(design)={X.shape[0]}")

    # QR decomposition and compute residuals
    Q, R = np.linalg.qr(X, mode='reduced')
    # Residuals = Y - Q @ Q.T @ Y
    fitted = Q @ (Q.T @ Y)
    return Y - fitted


@residualize.register(np.ndarray)
def _residualize_matrix(x, data, cols=None):
    """Residualize data against a numpy array design matrix."""
    X = x
    if cols is not None:
        if isinstance(cols[0], str):
            raise ValueError("String column names not supported for numpy arrays")
        X = X[:, cols]
    return _residualize_with_matrix(X, data)


@residualize.register(pd.DataFrame)
def _residualize_dataframe(x, data, cols=None):
    """Residualize data against a DataFrame design matrix."""
    if cols is not None:
        X = x[cols].values
    else:
        X = x.values
    return _residualize_with_matrix(X, data)


# Register EventModel and BaselineModel implementations
def _register_model_residualize():
    try:
        from .design.event_model import EventModel

        @residualize.register(EventModel)
        def _residualize_event_model(x, data, cols=None):
            X = x.design_matrix
            if cols is not None:
                if isinstance(cols[0], str):
                    col_names = x.column_names
                    col_idx = [col_names.index(c) for c in cols]
                    X = X[:, col_idx]
                else:
                    X = X[:, cols]
            return _residualize_with_matrix(X, data)
    except ImportError:
        pass

    try:
        from .baseline.baseline_model import BaselineModel

        @residualize.register(BaselineModel)
        def _residualize_baseline_model(x, data, cols=None):
            dm = x.design_matrix
            X = np.asarray(dm)
            if cols is not None:
                if isinstance(cols[0], str):
                    col_names = list(dm.columns) if isinstance(dm, pd.DataFrame) else [f"V{i}" for i in range(X.shape[1])]
                    col_idx = [col_names.index(c) for c in cols]
                    X = X[:, col_idx]
                else:
                    X = X[:, cols]
            return _residualize_with_matrix(X, data)
    except ImportError:
        pass

_register_model_residualize()
