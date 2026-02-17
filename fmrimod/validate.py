"""Contrast validation and collinearity checking."""
import numpy as np
import pandas as pd
from functools import singledispatch


def validate_contrasts(x, weights=None, tol=1e-8):
    """Validate contrast weights against a design matrix or event model.

    Performs several diagnostic checks on each contrast vector:

    - **Estimability**: whether the contrast lies in the column space
      of the design matrix.
    - **Sum-to-zero**: whether the weights sum to zero (within ``tol``).
    - **Intercept orthogonality**: whether the contrast has zero weight
      on intercept-like columns.
    - **Full rank** (F-contrasts only): whether the contrast matrix
      has full column rank.

    Parameters
    ----------
    x : EventModel, numpy.ndarray, or pandas.DataFrame
        Design matrix or event model containing a design matrix.
    weights : array-like, dict, or None
        Contrast weights to validate. Accepted formats:

        - ``None``: if ``x`` is an ``EventModel``, validates all
          attached contrasts.
        - ``numpy.ndarray`` or ``list``: a single contrast vector
          or matrix.
        - ``dict``: maps contrast names to weight vectors/matrices.
    tol : float, default=1e-8
        Numerical tolerance for zero checks.

    Returns
    -------
    pandas.DataFrame
        One row per contrast column with fields: ``name``, ``type``,
        ``estimable``, ``sum_to_zero``, ``orthogonal_to_intercept``,
        ``full_rank``, ``nonzero_weights``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.column_stack([np.ones(100), np.random.randn(100, 2)])
    >>> w = np.array([0, 1, -1])
    >>> result = validate_contrasts(X, weights=w)
    >>> result['estimable'].iloc[0]
    True

    See Also
    --------
    check_collinearity : Check design matrix for multicollinearity.
    """
    # Get design matrix
    from .design.event_model import EventModel

    if isinstance(x, EventModel):
        X = x.design_matrix
        col_names = x.column_names
    elif isinstance(x, pd.DataFrame):
        X = x.values
        col_names = list(x.columns)
    elif isinstance(x, np.ndarray):
        X = x
        col_names = [f"V{i+1}" for i in range(x.shape[1])]
    else:
        raise TypeError("x must be EventModel, DataFrame, or numpy array")

    # Find intercept-like columns
    intercept_names = {'(Intercept)', 'Intercept', 'constant', 'const'}
    intercept_idx = [i for i, name in enumerate(col_names) if name in intercept_names]

    # Gather weights
    wlist = {}
    if weights is None:
        if isinstance(x, EventModel):
            # Try to get attached contrasts
            try:
                from .contrast import contrast_weights as cw_func
                cw = cw_func(x)
                if cw:
                    for name, obj in cw.items():
                        W = obj.get('offset_weights', obj.get('weights'))
                        if W is not None:
                            W = np.atleast_2d(np.asarray(W, dtype=float))
                            if W.shape[0] == 1 and W.shape[1] > 1:
                                W = W.T  # Ensure column vector
                            wlist[name] = W
            except Exception:
                pass
            if not wlist:
                return pd.DataFrame(columns=['name', 'type', 'estimable', 'sum_to_zero',
                                           'orthogonal_to_intercept', 'full_rank', 'nonzero_weights'])
        else:
            raise ValueError("When x is a matrix, weights must be provided")
    elif isinstance(weights, dict):
        for name, w in weights.items():
            w = np.asarray(w, dtype=float)
            if w.ndim == 1:
                w = w.reshape(-1, 1)
            wlist[name] = w
    elif isinstance(weights, np.ndarray):
        w = weights.astype(float)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
        wlist['contrast'] = w
    elif isinstance(weights, list):
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
        wlist['contrast'] = w
    else:
        raise TypeError("weights must be array, dict, or list")

    # Helper: check estimability
    def is_estimable(X, cvec):
        if len(cvec) != X.shape[1]:
            return False
        rX = np.linalg.matrix_rank(X)
        rXa = np.linalg.matrix_rank(np.vstack([X, cvec.reshape(1, -1)]))
        return rXa == rX

    # Validate each contrast
    rows = []
    for nm, W in wlist.items():
        if W.shape[0] != X.shape[1]:
            raise ValueError(
                f"Contrast '{nm}' has {W.shape[0]} weights but design matrix has "
                f"{X.shape[1]} columns"
            )

        ctype = "F" if W.shape[1] > 1 else "t"

        for j in range(W.shape[1]):
            cvec = W[:, j]
            col_name = nm if W.shape[1] == 1 else f"{nm}#{j+1}"

            estimable = is_estimable(X, cvec)
            sum_to_zero = abs(np.sum(cvec)) < tol
            orth_int = all(abs(cvec[i]) < tol for i in intercept_idx) if intercept_idx else True
            nonzero = int(np.sum(np.abs(cvec) > tol))

            full_rank = None
            if ctype == "F":
                full_rank = np.linalg.matrix_rank(W) == W.shape[1]

            rows.append({
                'name': col_name,
                'type': ctype,
                'estimable': estimable,
                'sum_to_zero': sum_to_zero,
                'orthogonal_to_intercept': orth_int,
                'full_rank': full_rank,
                'nonzero_weights': nonzero,
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values('name').reset_index(drop=True)
    return result


def check_collinearity(X, threshold=0.9):
    """Check a design matrix for multicollinearity.

    Computes the pairwise Pearson correlation between all non-intercept,
    non-zero-variance columns and flags pairs whose absolute correlation
    exceeds the threshold.

    Parameters
    ----------
    X : EventModel, numpy.ndarray, or pandas.DataFrame
        Design matrix to check. Intercept-like columns (named
        ``'Intercept'``, ``'constant'``, etc.) are automatically
        excluded.
    threshold : float, default=0.9
        Absolute correlation above which a pair is flagged as
        collinear.

    Returns
    -------
    dict
        ``{'ok': bool, 'pairs': DataFrame}`` where ``ok`` is True
        when no pair exceeds the threshold. The ``pairs`` DataFrame
        has columns ``regressor_1``, ``regressor_2``, and ``r``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.column_stack([np.arange(100), np.arange(100) + 0.1])
    >>> result = check_collinearity(X, threshold=0.99)
    >>> result['ok']
    False

    See Also
    --------
    validate_contrasts : Validate contrast estimability.
    """
    from .design.event_model import EventModel

    if isinstance(X, EventModel):
        col_names = X.column_names
        X = X.design_matrix
    elif isinstance(X, pd.DataFrame):
        col_names = list(X.columns)
        X = X.values
    elif isinstance(X, np.ndarray):
        col_names = [f"V{i+1}" for i in range(X.shape[1])]
    else:
        raise TypeError("X must be EventModel, DataFrame, or numpy array")

    # Drop intercept-like columns
    intercept_names = {'(Intercept)', 'Intercept', 'constant', 'const'}
    keep = [i for i, name in enumerate(col_names) if name not in intercept_names]

    if len(keep) == 0:
        return {'ok': True, 'pairs': pd.DataFrame()}

    Xk = X[:, keep]
    keep_names = [col_names[i] for i in keep]

    # Drop zero-variance columns
    variances = np.var(Xk, axis=0)
    valid = np.isfinite(variances) & (variances > 0)
    Xk = Xk[:, valid]
    keep_names = [n for n, v in zip(keep_names, valid) if v]

    if Xk.shape[1] < 2:
        return {'ok': True, 'pairs': pd.DataFrame()}

    # Compute correlation matrix
    with np.errstate(invalid='ignore'):
        C = np.corrcoef(Xk, rowvar=False)
    np.fill_diagonal(C, 0)

    # Find pairs above threshold
    idx = np.argwhere(np.abs(C) > threshold)
    # Keep only upper triangle
    idx = idx[idx[:, 0] < idx[:, 1]]

    if len(idx) == 0:
        return {'ok': True, 'pairs': pd.DataFrame()}

    pairs = pd.DataFrame({
        'regressor_1': [keep_names[i] for i, _ in idx],
        'regressor_2': [keep_names[j] for _, j in idx],
        'r': [C[i, j] for i, j in idx],
    })

    return {'ok': False, 'pairs': pairs}
