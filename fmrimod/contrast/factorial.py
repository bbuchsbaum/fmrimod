"""Fast factorial contrast generators.

These helpers mirror fmrireg's matrix generators:

- ``generate_main_effect_contrast(des, factor)``
- ``generate_interaction_contrast(des, factors)``

They return matrices with shape ``(n_cells, n_contrasts)``. Row ``r`` holds
the weights of the cell in row ``r`` of ``des`` (for an ``expand.grid``-style
design the first factor varies fastest); selected factors are
difference-coded and non-selected factors are grand-mean coded. To apply a
matrix to the columns of a crossed design term, list the cells in ``des`` in
that term's column order (fmrimod crossed terms vary the *last* factor
fastest).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..types import Array


def _coerce_design(des: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    """Coerce input design to a DataFrame while preserving column order."""
    if isinstance(des, pd.DataFrame):
        out = des.copy()
    else:
        out = pd.DataFrame(des)
    if out.shape[1] == 0:
        raise ValueError("des must contain at least one factor column")
    return out


def _factor(series: pd.Series[Any]) -> pd.Categorical:
    """Factor view of a column: its categories, or sorted levels as R ``factor()``."""
    return pd.Categorical(series)


def _difference_block(n_levels: int) -> Array:
    """Create R-equivalent difference coding block: ``-t(diff(diag(L)))``."""
    if n_levels < 2:
        return np.zeros((n_levels, 0), dtype=float)
    return -np.diff(np.eye(n_levels), axis=0).T


def _as_factor_list(factors: str | Sequence[str]) -> list[str]:
    """Normalize factor input to a list."""
    if isinstance(factors, str):
        return [factors]
    return [str(f) for f in factors]


def generate_interaction_contrast(
    des: pd.DataFrame | dict[str, Any],
    factors: str | Sequence[str],
) -> Array:
    """Generate factorial interaction/main-effect contrast matrices.

    Parameters
    ----------
    des : pandas.DataFrame or mapping
        One column per factor, listing every combination of factor levels
        exactly once. Row ``r`` of the result is the cell in row ``r`` of
        ``des``. Categorical columns keep their category order; other columns
        use sorted levels (as R ``factor()``).
    factors : str or sequence of str
        Factor names to difference-code. Passing a single factor reproduces a
        main-effect matrix.

    Returns
    -------
    Array
        Contrast matrix of shape ``(prod(levels), prod(levels_i - 1))`` for
        selected factors; columns put the first selected factor's contrast
        index fastest.

    Raises
    ------
    ValueError
        If ``des`` is not a complete factorial (a cell missing or repeated).
    """
    design = _coerce_design(des)
    factor_list = _as_factor_list(factors)
    if len(factor_list) == 0:
        raise ValueError("factors must contain at least one factor name")

    missing = [f for f in factor_list if f not in design.columns]
    if missing:
        raise ValueError(f"factors not found in des: {missing}")

    factors_by_column = [_factor(design[name]) for name in design.columns]
    codes = np.column_stack([f.codes for f in factors_by_column])
    n_cells = int(np.prod([len(f.categories) for f in factors_by_column], dtype=int))
    if (
        design.shape[0] != n_cells
        or np.any(codes < 0)
        or np.unique(codes, axis=0).shape[0] != n_cells
    ):
        raise ValueError(
            "des must list every combination of factor levels exactly once "
            "(e.g. an expand.grid of the levels)"
        )

    # Row r is the weight of the cell in row r of ``des``: a row-wise
    # Kronecker product of the per-factor blocks indexed by that row's level
    # of each factor (fmrireg 91bb402). Columns put the first factor's
    # contrast index fastest.
    selected = set(factor_list)
    c_matrix = np.ones((design.shape[0], 1), dtype=float)
    for name, factor, code in zip(design.columns, factors_by_column, codes.T, strict=True):
        n_levels = len(factor.categories)
        block = (
            _difference_block(n_levels)
            if name in selected
            else np.ones((n_levels, 1), dtype=float)
        )
        rows = block[code]
        c_matrix = (rows[:, :, np.newaxis] * c_matrix[:, np.newaxis, :]).reshape(
            design.shape[0], -1
        )
    return np.asarray(c_matrix, dtype=float)


def generate_main_effect_contrast(
    des: pd.DataFrame | dict[str, Any],
    factor: str | Sequence[str],
) -> Array:
    """Generate a main-effect contrast matrix for a single factor."""
    if isinstance(factor, str):
        factor_name = factor
    else:
        items = list(factor)
        if len(items) != 1:
            raise ValueError("main-effect contrast expects exactly one factor name")
        factor_name = str(items[0])
    return generate_interaction_contrast(des, factor_name)
