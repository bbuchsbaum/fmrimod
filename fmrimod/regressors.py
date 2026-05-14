"""Regressor extraction functionality.

This module provides functions for extracting regressors from event models
and design matrices.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Pattern, Union

import numpy as np
import pandas as pd

from ._warnings import suppress_fmrimod_warnings
from .design.event_model import EventModel
from .types import Array


def regressors(
    model,
    term: Optional[Union[str, Pattern, List[str]]] = None,
    include_baseline: bool = True,
    as_dict: bool = False,
    hrf=None,
    sampling_frame=None,
    precision: float = 0.3,
) -> Union[Array, pd.DataFrame, Dict[str, Array]]:
    """Extract regressors from an event model or event term.

    This function extracts specific regressors or groups of regressors
    from an EventModel's design matrix, or convolves a single EventTerm
    with an HRF and sampling frame.

    Parameters
    ----------
    model : EventModel or EventTerm
        The event model containing the design matrix, or an EventTerm
        to convolve with an HRF
    term : str, regex pattern, list of str, optional
        Term name(s) or pattern to match columns (EventModel only):
        - None: return all regressors
        - str: return columns matching this term name
        - regex Pattern: return columns matching pattern
        - list of str: return columns for these terms
    include_baseline : bool, default=True
        Whether to include baseline/nuisance regressors
    as_dict : bool, default=False
        If True, return dict mapping column names to arrays.
        If False, return DataFrame or array.
    hrf : HRF object or str, optional
        HRF to use for convolution (EventTerm only). Defaults to SPM canonical.
    sampling_frame : SamplingFrame, optional
        Sampling frame for convolution (required for EventTerm)
    precision : float, default=0.3
        Temporal precision for HRF convolution (EventTerm only)

    Returns
    -------
    array-like, DataFrame, or dict
        Extracted regressors. Format depends on as_dict parameter.

    Examples
    --------
    >>> # Get all regressors as DataFrame
    >>> all_regs = regressors(model)
    >>>
    >>> # Get specific term's regressors
    >>> visual_regs = regressors(model, term='visual')
    >>>
    >>> # Get multiple terms
    >>> task_regs = regressors(model, term=['visual', 'motor'])
    >>>
    >>> # Use regex pattern
    >>> import re
    >>> condition_regs = regressors(model, term=re.compile(r'cond_.*'))
    >>>
    >>> # Get as dictionary
    >>> reg_dict = regressors(model, as_dict=True)
    >>>
    >>> # Exclude baseline regressors
    >>> event_regs = regressors(model, include_baseline=False)
    """
    # Dispatch on EventTerm
    from .events.term import EventTerm
    if isinstance(model, EventTerm):
        return _regressors_event_term(
            model, hrf=hrf, sampling_frame=sampling_frame, precision=precision
        )

    # Get full design matrix and column names
    X = model.design_matrix
    col_names = model.column_names
    n_cols = len(col_names)
    
    # Get column_indices from model if available (maps term names to col indices)
    model_col_indices = getattr(model, 'column_indices', {}) or {}

    # Determine which columns to include
    if term is None:
        # Include all columns
        if include_baseline:
            col_indices = list(range(n_cols))
        else:
            # Exclude baseline columns
            col_indices = []
            for i, name in enumerate(col_names):
                if not any(pattern in name.lower() for pattern in
                          ['baseline', 'drift', 'intercept', 'block', 'nuisance']):
                    col_indices.append(i)
    else:
        col_indices = []

        # Convert term to list for uniform processing
        if isinstance(term, str):
            terms = [term]
        elif isinstance(term, Pattern):
            terms = [term]
        elif isinstance(term, list):
            terms = term
        else:
            raise ValueError(
                f"term must be str, regex pattern, or list, got {type(term)}"
            )

        # Find matching columns for each term
        for t in terms:
            if isinstance(t, str):
                # First try exact match in column_indices
                if t in model_col_indices:
                    col_indices.extend(model_col_indices[t])
                else:
                    # Fall back to regex matching against column names
                    pattern = re.compile(rf"^{re.escape(t)}(\[|_|\.|$)")
                    matches = [i for i, name in enumerate(col_names)
                              if pattern.match(name)]
                    col_indices.extend(matches)
            elif isinstance(t, Pattern):
                # Regex pattern matching
                matches = [i for i, name in enumerate(col_names)
                          if t.search(name)]
                col_indices.extend(matches)
            else:
                raise ValueError(
                    f"Term items must be str or regex pattern, got {type(t)}"
                )

        # Remove duplicates while preserving order
        seen = set()
        col_indices = [i for i in col_indices if not (i in seen or seen.add(i))]

        # Apply baseline filter if needed
        if not include_baseline:
            filtered_indices = []
            for i in col_indices:
                name = col_names[i]
                if not any(pattern in name.lower() for pattern in
                          ['baseline', 'drift', 'intercept', 'block', 'nuisance']):
                    filtered_indices.append(i)
            col_indices = filtered_indices
    
    # Check if any columns were found
    if not col_indices:
        if term is not None:
            raise ValueError(f"No columns found matching term(s): {term}")
        else:
            raise ValueError("No non-baseline columns found")
    
    # Extract selected columns
    selected_X = X[:, col_indices]
    selected_names = [col_names[i] for i in col_indices]
    
    # Return in requested format
    if as_dict:
        # Return dictionary mapping names to column vectors
        result = {}
        for i, name in enumerate(selected_names):
            result[name] = selected_X[:, i]
        return result
    else:
        # Return DataFrame
        return pd.DataFrame(selected_X, columns=selected_names)


def regressor_names(
    model: EventModel,
    term: Optional[Union[str, Pattern, List[str]]] = None,
    include_baseline: bool = True
) -> List[str]:
    """Get names of regressors matching criteria.
    
    Convenience function to get just the names without extracting
    the actual regressor values.
    
    Parameters
    ----------
    model : EventModel
        The event model
    term : str, regex pattern, list of str, optional
        Term name(s) or pattern to match
    include_baseline : bool, default=True
        Whether to include baseline/nuisance regressors
    
    Returns
    -------
    list of str
        Names of matching regressors
    
    Examples
    --------
    >>> # Get all regressor names
    >>> all_names = regressor_names(model)
    >>> 
    >>> # Get names for specific term
    >>> visual_names = regressor_names(model, term='visual')
    """
    # Use regressors() to get the selection, then extract names
    reg_df = regressors(model, term=term, include_baseline=include_baseline, as_dict=False)
    return list(reg_df.columns)


def term_regressors(
    model: EventModel,
    as_dict: bool = True
) -> Dict[str, Union[Array, pd.DataFrame]]:
    """Extract regressors grouped by term.
    
    This function groups regressors by their originating term,
    making it easy to work with all regressors from each term
    separately.
    
    Parameters
    ----------
    model : EventModel
        The event model
    as_dict : bool, default=True
        If True, return dict of arrays for each term.
        If False, return dict of DataFrames for each term.
    
    Returns
    -------
    dict
        Dictionary mapping term names to their regressors
    
    Examples
    --------
    >>> # Get regressors grouped by term
    >>> term_dict = term_regressors(model)
    >>> visual_regs = term_dict['visual']  # All visual regressors
    >>> 
    >>> # Get as DataFrames
    >>> term_dfs = term_regressors(model, as_dict=False)
    """
    result = {}
    X = model.design_matrix
    col_names = model.column_names
    model_col_indices = getattr(model, 'column_indices', {}) or {}

    if model_col_indices:
        # Use model's column_indices (term name -> list of column indices)
        for term_name, indices in model_col_indices.items():
            if not indices:
                continue
            selected_names = [col_names[i] for i in indices if i < len(col_names)]
            selected_X = X[:, indices]

            if as_dict:
                term_dict = {}
                for i, name in enumerate(selected_names):
                    term_dict[name] = selected_X[:, i]
                result[term_name] = term_dict
            else:
                result[term_name] = pd.DataFrame(selected_X, columns=selected_names)
    else:
        # Fallback: parse column names
        term_names_set = set()
        for col_name in col_names:
            match = re.match(r'^([^[_.\d]+)', col_name)
            if match:
                term_name = match.group(1)
                if term_name.lower() not in ['baseline', 'drift', 'intercept', 'block', 'nuisance']:
                    term_names_set.add(term_name)

        for term_name in sorted(term_names_set):
            try:
                term_regs = regressors(model, term=term_name, include_baseline=False, as_dict=as_dict)
                if as_dict and term_regs:
                    result[term_name] = term_regs
                elif not as_dict and not term_regs.empty:
                    result[term_name] = term_regs
            except ValueError:
                continue

    return result


def _regressors_event_term(event_term, hrf=None, sampling_frame=None, precision=0.3):
    """Convolve a single event term with HRF and sampling frame.

    Parameters
    ----------
    event_term : EventTerm
        The event term to convolve
    hrf : HRF object or str, optional
        HRF to use. Defaults to SPM canonical.
    sampling_frame : SamplingFrame, optional
        Sampling frame defining the time grid. Required.
    precision : float, default=0.3
        Temporal precision for convolution

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per regressor
    """
    if sampling_frame is None:
        raise ValueError("sampling_frame is required for EventTerm regressors")

    try:
        with suppress_fmrimod_warnings():
            from . import regressor as _regressor_module
            from .hrf import library as _hrf_library
    except ImportError as err:
        raise ImportError(
            "fmrimod HRF library is required for EventTerm regressor convolution"
        ) from err

    # Resolve HRF
    if hrf is None:
        hrf_obj = _hrf_library.SPM_CANONICAL
    elif isinstance(hrf, str):
        from .dispatch import get_hrf as _get_hrf
        hrf_obj = _get_hrf(hrf)
    else:
        hrf_obj = hrf

    # Get sampling grid
    if hasattr(sampling_frame, 'samples'):
        grid = np.asarray(sampling_frame.samples)
    else:
        grid = np.asarray(sampling_frame)

    # Get event info
    if not event_term.events:
        return pd.DataFrame()

    event = event_term.events[0]
    onsets_arr = np.array(event.onsets)
    durations_arr = np.array(event.durations)

    # Get the design matrix for this term (indicator/value columns)
    dm = event_term.design_matrix(onsets_arr)
    col_names = event_term.get_column_names()

    # Ensure dm is 2D
    if dm.ndim == 1:
        dm = dm.reshape(-1, 1)

    # Convolve each column with the HRF
    n_basis = hrf_obj.nbasis if hasattr(hrf_obj, 'nbasis') else 1
    n_dm_cols = dm.shape[1]

    all_columns = []
    all_names = []

    for j in range(n_dm_cols):
        amplitudes = dm[:, j]
        # Create regressor and evaluate on grid
        reg = _regressor_module.regressor(
            onsets=onsets_arr,
            hrf=hrf_obj,
            duration=durations_arr,
            amplitude=amplitudes,
        )
        reg_arr = reg.evaluate(grid, precision=precision)
        reg_arr = np.asarray(reg_arr)
        if reg_arr.ndim == 1:
            reg_arr = reg_arr.reshape(-1, 1)

        for b in range(reg_arr.shape[1]):
            all_columns.append(reg_arr[:, b])
            if n_basis > 1:
                all_names.append(f"{col_names[j]}_b{b+1}")
            else:
                all_names.append(col_names[j])

    if not all_columns:
        return pd.DataFrame()

    result = np.column_stack(all_columns)
    return pd.DataFrame(result, columns=all_names)
