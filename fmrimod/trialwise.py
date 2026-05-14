"""Single-trial (LSA-style) regressor functionality for fMRI design matrices.

This module provides the :func:`trialwise` function which creates one
regressor per trial in the event model, optionally appending a grand-mean
column. This is the standard approach for Least-Squares All (LSA)
single-trial estimation used in representational similarity analysis (RSA)
and multi-voxel pattern analysis (MVPA).
"""

from typing import Any, Optional, Union

import numpy as np

from .events import EventFactor
from .formula import Term


def trialwise(
    basis: Union[str, object] = "spmg1",
    lag: float = 0.0,
    nbasis: int = 1,
    add_sum: bool = False,
    label: str = "trial",
    durations: Optional[Any] = None,
    normalize: bool = False,
) -> Term:
    """Generate one regressor per trial (plus an optional grand-mean column).

    Use it only on the RHS of an event-model formula:

        event_model("onset ~ trialwise(basis='spmg1', add_sum=True)", data)

    This creates individual regressors for each trial/event in the data,
    allowing for single-trial analysis. Each trial gets its own column
    in the design matrix.

    Parameters
    ----------
    basis : str or HRF object, default="spmg1"
        The hemodynamic response function or name of pre-supplied function.
        Options: "gamma", "spmg1", "spmg2", "spmg3", "bspline", "gaussian"
    lag : float, default=0.0
        Temporal offset in seconds added to onset before convolution
    nbasis : int, default=1
        Number of basis functions (for HRFs that support variable bases)
    add_sum : bool, default=False
        If True, append a column that is the average of all trialwise columns.
        This serves as a conventional main effect regressor.
    label : str, default="trial"
        Term label/prefix for the generated columns
    durations : optional
        Duration specification (column name, float, or array). Passed to hrf().
    normalize : bool, default=False
        Peak-normalize each regressor column. Passed to hrf().

    Returns
    -------
    Term
        A Term object configured for single-trial regressors

    Examples
    --------
    >>> # Basic single-trial model
    >>> model = event_model("onset ~ trialwise()", data)
    >>>
    >>> # Single-trial with grand mean
    >>> model = event_model("onset ~ trialwise(add_sum=True)", data)
    >>>
    >>> # Custom HRF for single trials
    >>> model = event_model("onset ~ trialwise(basis='spmg3')", data)
    >>>
    >>> # With durations and normalization
    >>> model = event_model("onset ~ trialwise(durations='duration', normalize=True)", data)
    >>>
    >>> # Combined with other terms
    >>> model = event_model("onset ~ hrf(condition) + trialwise()", data)

    Notes
    -----
    The trialwise() function works by creating a special EventFactor that
    generates unique levels for each trial. This is then processed through
    the standard HRF convolution pipeline.

    When add_sum=True, an additional column is added that represents the
    average activation across all trials, useful for group-level analyses.
    """
    # Create a special term that will be expanded during model construction
    # We use a placeholder event that will be replaced with trial indices
    term = Term(
        events="_trialwise_placeholder_",
        hrf=basis,
        name=label
    )

    # Add special attributes for trialwise processing
    term._is_trialwise = True
    term._add_sum = add_sum
    term._trialwise_label = label
    term._lag = lag
    term._nbasis = nbasis

    # New parameters - store in _kwargs
    if not hasattr(term, '_kwargs') or term._kwargs is None:
        term._kwargs = {}
    if durations is not None:
        term._kwargs['durations'] = durations
    if normalize:
        term._kwargs['normalize'] = True

    return term


def _create_trial_factor(n: int, onsets: Optional[np.ndarray] = None) -> EventFactor:
    """Create an EventFactor with a unique level per trial.

    This is an internal helper used by ``EventModel`` when expanding
    ``trialwise()`` terms. Each trial receives a zero-padded numeric
    label (e.g., ``"01"``, ``"02"``, ...) as its factor level.

    Parameters
    ----------
    n : int
        Number of trials.
    onsets : array-like, optional
        Onset times for the trials. If ``None``, integer indices
        ``0, 1, ..., n-1`` are used.

    Returns
    -------
    EventFactor
        Factor with ``n`` levels named ``"001"``, ``"002"``, etc.
        (zero-padded to the width of ``n``).
    """
    # Determine padding for nice formatting
    pad = len(str(n))
    
    # Create trial labels with zero-padding
    trial_labels = [f"{i+1:0{pad}d}" for i in range(n)]
    
    # Create EventFactor with these levels
    # Each trial gets its own unique level
    levels = trial_labels
    
    # If no onsets provided, use indices
    if onsets is None:
        onsets = np.arange(n)
    
    return EventFactor(
        name="trial",
        onsets=onsets,
        values=trial_labels,  # Each trial gets its own value
        levels=levels
    )