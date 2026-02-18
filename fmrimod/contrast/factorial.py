"""Fast factorial contrast generators.

These helpers mirror fmrireg's matrix generators:

- ``generate_main_effect_contrast(des, factor)``
- ``generate_interaction_contrast(des, factors)``

They return matrices with shape ``(n_cells, n_contrasts)`` where rows encode
factorial cells (Kronecker order by column position in ``des``), selected
factors are difference-coded, and non-selected factors are grand-mean coded.
"""

from __future__ import annotations

from functools import reduce
from typing import Sequence

import numpy as np
import pandas as pd

from ..types import Array


def _coerce_design(des: pd.DataFrame | dict) -> pd.DataFrame:
    """Coerce input design to a DataFrame while preserving column order."""
    if isinstance(des, pd.DataFrame):
        out = des.copy()
    else:
        out = pd.DataFrame(des)
    if out.shape[1] == 0:
        raise ValueError("des must contain at least one factor column")
    return out


def _nlevels(series: pd.Series) -> int:
    """Count levels for a factor-like series."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return int(len(series.cat.categories))
    return int(len(pd.unique(series.dropna())))


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
    des: pd.DataFrame | dict,
    factors: str | Sequence[str],
) -> Array:
    """Generate factorial interaction/main-effect contrast matrices.

    Parameters
    ----------
    des : pandas.DataFrame or mapping
        One column per factor.
    factors : str or sequence of str
        Factor names to difference-code. Passing a single factor reproduces a
        main-effect matrix.

    Returns
    -------
    Array
        Contrast matrix of shape ``(prod(levels), prod(levels_i - 1))`` for
        selected factors.
    """
    design = _coerce_design(des)
    factor_list = _as_factor_list(factors)
    if len(factor_list) == 0:
        raise ValueError("factors must contain at least one factor name")

    missing = [f for f in factor_list if f not in design.columns]
    if missing:
        raise ValueError(f"factors not found in des: {missing}")

    fac_names = list(design.columns)
    fac_set = set(factor_list)
    levels = [_nlevels(design[name]) for name in fac_names]

    blocks = [
        _difference_block(levels[i]) if fac_names[i] in fac_set else np.ones((levels[i], 1), dtype=float)
        for i in range(len(fac_names))
    ]
    c_matrix = reduce(np.kron, blocks)

    n_cells = int(np.prod(levels, dtype=int))
    if c_matrix.shape[0] != n_cells:
        raise RuntimeError(
            f"generated matrix has {c_matrix.shape[0]} rows, expected {n_cells}"
        )
    return np.asarray(c_matrix, dtype=float)


def generate_main_effect_contrast(
    des: pd.DataFrame | dict,
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
