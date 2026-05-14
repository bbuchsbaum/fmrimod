"""HRF library generation from parameter grids."""

from __future__ import annotations

from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .core import HRF, bind_basis


def hrf_library(
    fun: Callable[..., HRF],
    pgrid: Union[pd.DataFrame, Dict[str, list]],
    **kwargs
) -> HRF:
    """Generate an HRF library from a parameter grid.
    
    Applies a base HRF generating function to each row of a parameter grid,
    creating a combined HRF object representing the library.
    
    Args:
        fun: Function that generates an HRF given parameters
        pgrid: Parameter grid as DataFrame or dict
        **kwargs: Additional arguments passed to fun
        
    Returns:
        Combined HRF object representing the library
        
    Examples:
        >>> # Create library of gamma HRFs with varying parameters
        >>> param_grid = pd.DataFrame({
        ...     'shape': [4, 6, 8],
        ...     'rate': [0.9, 1.0, 1.1]
        ... })
        >>> from fmrimod.hrf.generators import gamma_generator
        >>> gamma_lib = hrf_library(gamma_generator, param_grid)
        >>> 
        >>> # Create library with fixed and varying parameters
        >>> import numpy as np
        >>> param_grid2 = {'lag': [0, 2, 4]}
        >>> from fmrimod.hrf.generators import gen_hrf
        >>> from fmrimod.hrf import get_hrf
        >>> lagged_lib = hrf_library(
        ...     lambda lag: gen_hrf(get_hrf("spmg1"), lag=lag),
        ...     param_grid2
        ... )
    """
    # Convert dict to DataFrame if needed
    if isinstance(pgrid, dict):
        # Create all combinations if multiple parameters
        import itertools
        keys = list(pgrid.keys())
        values = [pgrid[k] if isinstance(pgrid[k], list) else [pgrid[k]] for k in keys]
        combinations = list(itertools.product(*values))
        pgrid = pd.DataFrame(combinations, columns=keys)
    
    # Check for duplicate parameter names with kwargs
    param_names = set(pgrid.columns)
    kwarg_names = set(kwargs.keys())
    duplicates = param_names & kwarg_names
    if duplicates:
        raise ValueError(
            f"Duplicate parameter names in pgrid and kwargs: {duplicates}"
        )
    
    # Generate HRF for each parameter combination
    hrfs = []
    for idx, row in pgrid.iterrows():
        # Combine row parameters with kwargs
        params = dict(row)
        params.update(kwargs)
        
        # Generate HRF
        hrf = fun(**params)
        hrfs.append(hrf)
    
    # Combine into single HRF
    if len(hrfs) == 1:
        return hrfs[0]
    else:
        return bind_basis(*hrfs)  # Unpack list


def gen_hrf_library(*args, **kwargs) -> HRF:
    """R-compatible alias for :func:`hrf_library`."""
    return hrf_library(*args, **kwargs)
