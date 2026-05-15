"""Design matrix convolution functionality.

This module provides :func:`convolve_design` for convolving an existing
design matrix (or ``EventModel``) with an HRF, and
:func:`convolve_regressors` for convolving a dictionary of named
regressors. These are useful when you have a pre-built matrix and want
to apply (or re-apply) HRF convolution as a post-processing step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

from .design.event_model import EventModel
from .dispatch import get_hrf
from .types import Array, HRFProtocol


def convolve_design(
    design_matrix: Union[Array, pd.DataFrame, EventModel],
    hrf: Union[str, Dict[str, object], HRFProtocol] = "spmg1",
    sampling_rate: Optional[float] = None,
    time_points: Optional[Array] = None,
    column_names: Optional[List[str]] = None,
    normalize: bool = False,
    **hrf_kwargs: object
) -> Union[Array, pd.DataFrame]:
    """Convolve a design matrix with a hemodynamic response function.

    This function takes an existing design matrix and convolves each column
    with the specified HRF. Useful for applying HRF convolution to matrices
    that were constructed without convolution, or for applying different
    HRFs to different parts of a design.

    Parameters
    ----------
    design_matrix : array-like, DataFrame, or EventModel
        The design matrix to convolve. Can be:
        - NumPy array (n_timepoints x n_regressors)
        - pandas DataFrame with regressors as columns
        - EventModel (will extract design matrix)
    hrf : str, dict, or HRF object, default="spmg1"
        HRF specification:
        - String name: "spmg1", "spmg2", "spmg3", "gamma", etc.
        - Dict with 'name' and parameters
        - HRF object from fmrimod
    sampling_rate : float, optional
        Sampling rate in seconds. Required if design_matrix is array.
        Extracted automatically from DataFrame or EventModel.
    time_points : array-like, optional
        Time points for the design matrix rows. If None, generated
        from sampling_rate.
    column_names : list of str, optional
        Names for the columns (used when returning DataFrame).
        Extracted automatically from DataFrame or EventModel.
    normalize : bool, default False
        If True, peak-normalize each column after convolution so that
        max(abs(col)) == 1.
    **hrf_kwargs
        Additional keyword arguments passed to HRF function
    
    Returns
    -------
    array-like or DataFrame
        Convolved design matrix. Returns same type as input when possible.
    
    Examples
    --------
    >>> # Convolve existing design matrix with SPM canonical HRF
    >>> X_conv = convolve_design(X, hrf='spmg1', sampling_rate=2.0)
    >>> 
    >>> # Use custom HRF parameters
    >>> X_conv = convolve_design(X, hrf='gamma', shape=6, scale=1)
    >>> 
    >>> # Apply to EventModel
    >>> model_conv = convolve_design(model, hrf='spmg3')
    
    Notes
    -----
    The convolution is performed using the same method as in EventModel,
    ensuring consistency across the package. Zero-padding is used to
    maintain the original matrix dimensions.
    """
    # Extract design matrix and metadata based on input type
    if isinstance(design_matrix, EventModel):
        X = design_matrix.design_matrix
        if sampling_rate is None:
            sampling_rate = design_matrix.tr
        if time_points is None:
            time_points = design_matrix.sampling_points
        if column_names is None:
            column_names = design_matrix.column_names
        return_dataframe = True
    elif isinstance(design_matrix, pd.DataFrame):
        X = design_matrix.values
        if column_names is None:
            column_names = list(design_matrix.columns)
        if time_points is None and sampling_rate is None:
            raise ValueError(
                "Either sampling_rate or time_points must be provided for DataFrame input"
            )
        return_dataframe = True
    else:
        # Assume numpy array or similar
        X = np.asarray(design_matrix)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(
                f"Design matrix must be 1D or 2D, got {X.ndim}D"
            )
        if sampling_rate is None and time_points is None:
            raise ValueError(
                "Either sampling_rate or time_points must be provided for array input"
            )
        return_dataframe = False
    
    # Generate time points if needed
    n_timepoints = X.shape[0]
    if time_points is None:
        time_points = np.arange(n_timepoints) * sampling_rate
    else:
        time_points = np.asarray(time_points)
        if len(time_points) != n_timepoints:
            raise ValueError(
                f"time_points length ({len(time_points)}) must match "
                f"design matrix rows ({n_timepoints})"
            )
    
    # Get HRF function
    hrf_func = get_hrf(hrf, **hrf_kwargs)
    
    # Create HRF time points - HRF is typically evaluated from 0 to ~30s
    if sampling_rate is not None:
        hrf_duration = 30.0  # seconds
        hrf_timepoints = np.arange(0, hrf_duration, sampling_rate)
    else:
        # Use time_points spacing
        dt = np.median(np.diff(time_points))
        hrf_duration = 30.0
        hrf_timepoints = np.arange(0, hrf_duration, dt)
    
    # Evaluate HRF
    hrf_values = hrf_func.evaluate(hrf_timepoints)
    
    # Handle multi-dimensional HRF (e.g., with derivatives)
    if hrf_values.ndim > 1:
        # Use only the main HRF (first column)
        hrf_values = hrf_values[:, 0]
    
    # Normalize HRF to unit area (preserve total signal)
    hrf_sum: float = float(np.sum(hrf_values))
    if abs(hrf_sum) > 1e-10:
        hrf_values = hrf_values / hrf_sum
    else:
        # If sum is too close to zero, just use max normalization
        hrf_max: float = float(np.max(np.abs(hrf_values)))
        if hrf_max > 0:
            hrf_values = hrf_values / hrf_max
    
    # Convolve each column
    n_cols = X.shape[1]
    X_convolved = np.zeros_like(X)
    
    for i in range(n_cols):
        # Get column
        col = X[:, i]
        
        # Convolve with HRF - use 'full' mode for proper causal convolution
        convolved_full = np.convolve(col, hrf_values, mode='full')
        
        # Trim to original length (keep only the first n_timepoints)
        # This ensures causality - HRF response follows the stimulus
        convolved = convolved_full[:n_timepoints]
        
        # Store result
        X_convolved[:, i] = convolved

    # Peak-normalize each column if requested
    if normalize:
        for i in range(n_cols):
            mx: float = float(np.max(np.abs(X_convolved[:, i])))
            if mx > 0:
                X_convolved[:, i] = X_convolved[:, i] / mx

    # Return appropriate type
    if return_dataframe:
        return pd.DataFrame(
            X_convolved,
            columns=column_names if column_names else [f"V{i+1}" for i in range(n_cols)]
        )
    else:
        return X_convolved


def convolve_regressors(
    regressors: Dict[str, Array],
    hrf: Union[str, Dict[str, object], HRFProtocol] = "spmg1",
    sampling_rate: float = 1.0,
    **hrf_kwargs: object
) -> Dict[str, Array]:
    """Convolve a dictionary of regressors with an HRF.
    
    Convenience function for convolving multiple named regressors
    stored in a dictionary.
    
    Parameters
    ----------
    regressors : dict
        Dictionary mapping regressor names to time series arrays
    hrf : str, dict, or HRF object, default="spmg1"
        HRF specification
    sampling_rate : float, default=1.0
        Sampling rate in seconds
    **hrf_kwargs
        Additional HRF parameters
    
    Returns
    -------
    dict
        Dictionary with same keys, convolved regressors as values
    
    Examples
    --------
    >>> regressors = {
    ...     'visual': visual_regressor,
    ...     'motor': motor_regressor
    ... }
    >>> conv_regressors = convolve_regressors(regressors, hrf='spmg1', sampling_rate=2.0)
    """
    convolved = {}
    
    forwarded = cast("dict[str, Any]", hrf_kwargs)
    for name, regressor in regressors.items():
        # Ensure 2D for convolve_design
        reg_array = np.asarray(regressor)
        if reg_array.ndim == 1:
            reg_array = reg_array.reshape(-1, 1)

        # Convolve
        conv_array = cast(
            "np.ndarray[Any, Any]",
            convolve_design(
                reg_array,
                hrf=hrf,
                sampling_rate=sampling_rate,
                **forwarded,
            ),
        )

        # Store with original shape
        if regressor.ndim == 1:
            convolved[name] = conv_array.ravel()
        else:
            convolved[name] = conv_array
    
    return convolved