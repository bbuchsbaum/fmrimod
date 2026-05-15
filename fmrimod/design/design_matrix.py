"""Design matrix extraction generic function.

This module implements the design_matrix() generic function that extracts
design matrices from various objects in a consistent way.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ..design.event_model import EventModel
from ..types import Array


@singledispatch
def design_matrix(x: Any, **kwargs: Any) -> Array:
    """Extract or construct a design matrix.

    This is a :func:`functools.singledispatch` generic function: ``x``
    is intentionally ``Any`` because the dispatcher inspects its
    runtime type and routes to one of several typed registrations
    (``EventModel``, :class:`BaselineModel`, :class:`BaselineTerm`,
    :class:`EventTerm`). The registered implementations carry their
    own typed signatures.

    Parameters
    ----------
    x : object
        Object to extract design matrix from
    **kwargs
        Additional arguments depending on object type
    
    Returns
    -------
    Array or pd.DataFrame
        Design matrix
    
    Raises
    ------
    NotImplementedError
        If no method exists for the object type
    
    Examples
    --------
    >>> # From EventModel
    >>> model = event_model("condition + rating", data=df, tr=2.0, n_scans=100)
    >>> X = design_matrix(model)
    >>> 
    >>> # From baseline model (when implemented)
    >>> baseline = baseline_model(poly(3), sampling_info)
    >>> X_baseline = design_matrix(baseline)
    """
    raise NotImplementedError(
        f"No design_matrix method for type {type(x).__name__}"
    )


@design_matrix.register(EventModel)
def _(
    x: EventModel,
    blockid: Optional[Union[int, "list[int]"]] = None,
    **kwargs: object
) -> Array:
    """Extract design matrix from EventModel.

    Parameters
    ----------
    x : EventModel
        Event model object
    blockid : int or list of int, optional
        Block ID(s) to extract (1-based to match R convention).
        If None, returns full matrix.
    **kwargs
        Additional arguments (ignored)

    Returns
    -------
    Array
        Design matrix, possibly subset by block
    """
    # Get the full design matrix
    dm = x.design_matrix

    if blockid is None:
        return dm

    # Build timepoint-to-block mapping from sampling frame
    sf = x.sampling_info

    if hasattr(sf, 'blockids'):
        # fmrimod blockids can be 0-based or 1-based depending on source.
        # Normalize to 1-based ids expected by the user-facing API.
        timepoint_blockids = np.asarray(sf.blockids)
        if timepoint_blockids.size > 0 and timepoint_blockids.min() == 0:
            timepoint_blockids = timepoint_blockids + 1
    elif hasattr(sf, 'blocklens'):
        # Build from blocklens (1-based)
        timepoint_blockids = []
        for i, blen in enumerate(sf.blocklens):
            timepoint_blockids.extend([i + 1] * int(blen))
        timepoint_blockids = np.array(timepoint_blockids)
    else:
        # Single block - all timepoints belong to block 1
        timepoint_blockids = np.ones(dm.shape[0], dtype=int)

    # Convert blockid to list if single value
    if not isinstance(blockid, (list, tuple)):
        blockid = [blockid]

    # Find rows belonging to requested blocks
    keep_rows = np.isin(timepoint_blockids, blockid)

    if not np.any(keep_rows):
        import warnings
        warnings.warn(f"Specified blockid(s) {blockid} not found.")
        # Return empty array with same number of columns
        return dm[0:0, :]

    return dm[keep_rows, :]


# Register baseline model
def _register_baseline_model() -> None:
    """Register design_matrix method for baseline_model."""
    from ..baseline import BaselineModel
    
    @design_matrix.register(BaselineModel)
    def _(
        x: BaselineModel,
        blockid: Optional[Union[int, "list[int]"]] = None,
        allrows: bool = False,
        **kwargs: object
    ) -> Array:
        """Extract design matrix from BaselineModel.
        
        Parameters
        ----------
        x : BaselineModel
            Baseline model object
        blockid : int or list of int, optional
            Block ID(s) to extract. If None, returns full matrix.
        allrows : bool, default=False
            If True, return all rows but only columns for specified blocks.
        **kwargs
            Additional arguments (ignored)
        
        Returns
        -------
        Array
            Design matrix, possibly subset by block
        """
        if blockid is None:
            # Return full design matrix
            return x.design_matrix
        
        # Get matrices from each term for specified blocks
        matrices = []
        
        for term_name in ['drift', 'block', 'nuisance']:
            if term_name in x.terms and x.terms[term_name] is not None:
                term = x.terms[term_name]
                term_matrix = term.get_block_matrix(blockid, allrows)
                
                if not term_matrix.empty:
                    matrices.append(term_matrix.values)
        
        if matrices:
            return np.hstack(matrices)
        else:
            # Return empty matrix
            if allrows and hasattr(x.sampling_frame, 'n_scans'):
                n_rows = x.sampling_frame.n_scans
            else:
                n_rows = 0
            return np.zeros((n_rows, 0))

# Call registration
_register_baseline_model()


def _register_baseline_term() -> None:
    """Register design_matrix method for BaselineTerm."""
    from ..baseline.baseline_term import BaselineTerm

    @design_matrix.register(BaselineTerm)
    def _(x: BaselineTerm, **kwargs: object) -> Array:
        """Extract design matrix from BaselineTerm.

        Parameters
        ----------
        x : BaselineTerm
            Baseline term object
        **kwargs
            Additional arguments (ignored)

        Returns
        -------
        Array
            Design matrix for this term
        """
        mat = x.design_matrix
        if isinstance(mat, pd.DataFrame):
            return mat.values
        return mat

_register_baseline_term()


def _register_event_term() -> None:
    """Register design_matrix method for EventTerm."""
    from ..events.term import EventTerm

    @design_matrix.register(EventTerm)
    def _(x: EventTerm, **kwargs: object) -> Array:
        """Extract design matrix from EventTerm.

        For EventTerm, returns the unconvolved indicator/design matrix.
        This is the raw event coding before HRF convolution.

        Parameters
        ----------
        x : EventTerm
            Event term object
        **kwargs
            Additional arguments (passed to design_matrix method)

        Returns
        -------
        Array
            Design matrix for this term
        """
        # EventTerm.design_matrix() requires sampling_points parameter
        # If not provided in kwargs, we cannot compute it
        if 'sampling_points' not in kwargs:
            raise ValueError(
                "EventTerm.design_matrix() requires 'sampling_points' parameter"
            )
        return x.design_matrix(kwargs['sampling_points'])

_register_event_term()


# Re-export for convenience
__all__ = ['design_matrix']
