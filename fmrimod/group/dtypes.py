"""Dtype policy for native group-analysis arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

GROUP_FLOAT_DTYPE = np.float64
GROUP_INDEX_DTYPE = np.intp


def as_group_float_array(value: object, *, copy: bool = False) -> NDArray[np.float64]:
    """Coerce numeric values to the internal group-analysis floating dtype."""
    arr = np.asarray(value, dtype=GROUP_FLOAT_DTYPE)
    return arr.copy() if copy else arr


def as_group_index_array(value: object, *, copy: bool = False) -> NDArray[np.intp]:
    """Coerce integer index values to the internal group-analysis index dtype."""
    arr = np.asarray(value, dtype=GROUP_INDEX_DTYPE)
    return arr.copy() if copy else arr
