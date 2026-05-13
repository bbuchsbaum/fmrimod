"""Design matrix creation for fMRI GLM analysis."""

from __future__ import annotations

from typing import List, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from .convolution import ConvolutionMethod
from .core import Regressor, RegressorSet


@overload
def regressor_design(
    regressors: Union[Regressor, RegressorSet, List[Regressor]],
    grid: ArrayLike,
    precision: float = ...,
    method: ConvolutionMethod = ...,
    sparse: Literal[False] = ...,
    include_intercept: bool = ...,
    column_names: None = ...,
) -> NDArray[np.float64]: ...


@overload
def regressor_design(
    regressors: Union[Regressor, RegressorSet, List[Regressor]],
    grid: ArrayLike,
    precision: float = ...,
    method: ConvolutionMethod = ...,
    sparse: Literal[False] = ...,
    include_intercept: bool = ...,
    *,
    column_names: List[str],
) -> pd.DataFrame: ...


@overload
def regressor_design(
    regressors: Union[Regressor, RegressorSet, List[Regressor]],
    grid: ArrayLike,
    precision: float = ...,
    method: ConvolutionMethod = ...,
    *,
    sparse: Literal[True],
    include_intercept: bool = ...,
    column_names: Optional[List[str]] = ...,
) -> scipy.sparse.csr_matrix: ...


def regressor_design(
    regressors: Union[Regressor, RegressorSet, List[Regressor]],
    grid: ArrayLike,
    precision: float = 0.33,
    method: ConvolutionMethod = "conv",
    sparse: bool = False,
    include_intercept: bool = False,
    column_names: Optional[List[str]] = None
) -> Union[NDArray[np.float64], scipy.sparse.csr_matrix, pd.DataFrame]:
    """Create a design matrix from regressors.
    
    This function evaluates one or more regressors at specified time points
    to create a design matrix suitable for GLM analysis.
    
    Args:
        regressors: Single regressor, regressor set, or list of regressors
        grid: Time points at which to evaluate (seconds)
        precision: Temporal precision for convolution
        method: Evaluation method ('conv', 'fft', 'direct')
        sparse: Whether to return sparse matrix
        include_intercept: Whether to add an intercept column
        column_names: Optional column names for DataFrame output
        
    Returns:
        Design matrix as array, sparse matrix, or DataFrame
        
    Examples:
        >>> # Single regressor
        >>> reg = regressor(onsets=[10, 30, 50], duration=2)
        >>> times = np.arange(0, 80, 0.1)
        >>> design = regressor_design(reg, times)
        
        >>> # Multiple conditions
        >>> onsets = [10, 20, 30, 40, 50, 60]
        >>> conditions = ['A', 'B', 'C', 'A', 'B', 'C']
        >>> rset = regressor_set(onsets, conditions)
        >>> design = regressor_design(rset, times, column_names=['CondA', 'CondB', 'CondC'])
    """
    grid = np.asarray(grid, dtype=np.float64)
    
    # Handle different input types
    if isinstance(regressors, Regressor):
        # Single regressor
        design = regressors.evaluate(grid, precision, method, sparse=False)
        if design.ndim == 1:
            design = design[:, np.newaxis]
        n_cols = design.shape[1]
        
    elif isinstance(regressors, RegressorSet):
        # Regressor set
        design = regressors.evaluate(grid, precision, method, sparse=False)
        n_cols = design.shape[1]
        
    elif isinstance(regressors, list):
        # List of regressors
        results = []
        for reg in regressors:
            if not isinstance(reg, Regressor):
                raise TypeError(f"Expected Regressor, got {type(reg)}")
            result = reg.evaluate(grid, precision, method, sparse=False)
            if result.ndim == 1:
                result = result[:, np.newaxis]
            results.append(result)
        design = np.hstack(results)
        n_cols = design.shape[1]
        
    else:
        raise TypeError(
            f"regressors must be Regressor, RegressorSet, or list of Regressors, "
            f"got {type(regressors)}"
        )
    
    # Add intercept if requested
    if include_intercept:
        intercept = np.ones((len(grid), 1))
        design = np.hstack([intercept, design])
        n_cols += 1
    
    # Convert to sparse if requested
    if sparse:
        return csr_matrix(design)
    
    # Convert to DataFrame if column names provided
    if column_names is not None:
        if len(column_names) != n_cols:
            raise ValueError(
                f"Number of column names ({len(column_names)}) "
                f"doesn't match number of columns ({n_cols})"
            )
        return pd.DataFrame(design, columns=column_names, index=grid)
    
    return design
