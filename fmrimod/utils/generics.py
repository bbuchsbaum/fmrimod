"""Generic utility functions for extracting information from model objects."""

from functools import singledispatch
from typing import Any, List, Union

import numpy as np
import pandas as pd


@singledispatch
def blockids(x, **kwargs) -> Union[List[Any], np.ndarray]:
    """Extract block IDs from an object.
    
    This is a generic function that extracts block/run identifiers from
    various objects in the fmridesign framework.
    
    Parameters
    ----------
    x : object
        Object to extract block IDs from
    **kwargs
        Additional arguments for specific methods
    
    Returns
    -------
    array-like
        Block identifiers
    
    Examples
    --------
    >>> from fmrimod import EventModel
    >>> model = EventModel(...)  # doctest: +SKIP
    >>> blockids(model)  # doctest: +SKIP
    array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    """
    raise NotImplementedError(f"blockids not implemented for {type(x)}")


@singledispatch
def blocklens(x, **kwargs) -> Union[List[int], np.ndarray]:
    """Extract block lengths from an object.
    
    This is a generic function that extracts the lengths of blocks/runs
    from various objects in the fmridesign framework.
    
    Parameters
    ----------
    x : object
        Object to extract block lengths from
    **kwargs
        Additional arguments for specific methods
    
    Returns
    -------
    array-like
        Block lengths (number of scans per block)
    
    Examples
    --------
    >>> from fmrimod import EventModel
    >>> model = EventModel(...)  # doctest: +SKIP
    >>> blocklens(model)  # doctest: +SKIP
    array([100, 100, 100])
    """
    raise NotImplementedError(f"blocklens not implemented for {type(x)}")


@singledispatch
def term_names(x, **kwargs) -> List[str]:
    """Extract term names from an object.
    
    This is a generic function that extracts the names of terms
    (predictors) from various model objects.
    
    Parameters
    ----------
    x : object
        Object to extract term names from
    **kwargs
        Additional arguments for specific methods
    
    Returns
    -------
    list of str
        Term names
    
    Examples
    --------
    >>> from fmrimod import EventModel
    >>> model = EventModel(...)  # doctest: +SKIP
    >>> term_names(model)  # doctest: +SKIP
    ['condition', 'response_time', 'condition:response_time']
    """
    raise NotImplementedError(f"term_names not implemented for {type(x)}")


@singledispatch
def longnames(x, **kwargs) -> List[str]:
    """Extract long (descriptive) names from an object.
    
    This is a generic function that extracts verbose, human-readable names
    from various objects in the fmridesign framework. Long names typically
    include full descriptions with basis function indices.
    
    Parameters
    ----------
    x : object
        Object to extract long names from
    **kwargs
        Additional arguments for specific methods
    
    Returns
    -------
    list of str
        Long descriptive names
    
    Examples
    --------
    >>> from fmrimod import EventModel
    >>> model = EventModel(...)  # doctest: +SKIP
    >>> longnames(model)  # doctest: +SKIP
    ['condition[face].basis01', 'condition[face].basis02', 
     'condition[house].basis01', 'condition[house].basis02']
    """
    raise NotImplementedError(f"longnames not implemented for {type(x)}")


@singledispatch 
def shortnames(x, **kwargs) -> List[str]:
    """Extract short (abbreviated) names from an object.
    
    This is a generic function that extracts concise names from various 
    objects in the fmridesign framework. Short names are typically more
    compact versions suitable for display or AFNI compatibility.
    
    Parameters
    ----------
    x : object
        Object to extract short names from
    **kwargs
        Additional arguments for specific methods
    
    Returns
    -------
    list of str
        Short abbreviated names
    
    Examples
    --------
    >>> from fmrimod import EventModel
    >>> model = EventModel(...)  # doctest: +SKIP  
    >>> shortnames(model)  # doctest: +SKIP
    ['cond.face.b01', 'cond.face.b02',
     'cond.house.b01', 'cond.house.b02']
    """
    raise NotImplementedError(f"shortnames not implemented for {type(x)}")


# Register implementations for common types

@blockids.register(list)
def _blockids_list(x: list, **kwargs) -> list:
    """Extract block IDs from a list (assumes it's already block IDs)."""
    return x


@blockids.register(np.ndarray)
def _blockids_array(x: np.ndarray, **kwargs) -> np.ndarray:
    """Extract block IDs from an array (assumes it's already block IDs)."""
    return x


# Register EventModel implementations when available
def _register_event_model():
    """Register EventModel implementations after imports are resolved."""
    try:
        from ..design.event_model import EventModel
        
        @blockids.register(EventModel)
        def _blockids_event_model(model: EventModel, **kwargs):
            """Extract block IDs from an EventModel."""
            return model.blockids
        
        @blocklens.register(EventModel)
        def _blocklens_event_model(model: EventModel, **kwargs):
            """Extract block lengths from an EventModel."""
            if hasattr(model, 'sampling_frame') and model.sampling_frame is not None:
                # Get from sampling frame
                if hasattr(model.sampling_frame, 'blocklens'):
                    return model.sampling_frame.blocklens
                elif hasattr(model.sampling_frame, 'block_lengths'):
                    return model.sampling_frame.block_lengths
            # Fallback: compute from blockids
            if hasattr(model, 'blockids'):
                bids = np.array(model.blockids)
                unique_blocks = np.unique(bids)
                return np.array([np.sum(bids == b) for b in unique_blocks])
            raise ValueError("Cannot determine block lengths from EventModel")
        
        @term_names.register(EventModel)
        def _term_names_event_model(model: EventModel, **kwargs):
            """Extract term names from an EventModel."""
            if hasattr(model, 'terms'):
                return [term.name for term in model.terms if hasattr(term, 'name')]
            return []
            
    except ImportError:
        pass


# Register BaselineModel implementations when available
def _register_baseline_model():
    """Register BaselineModel implementations after imports are resolved."""
    try:
        from ..baseline.baseline_model import BaselineModel
        
        @term_names.register(BaselineModel)
        def _term_names_baseline_model(model: BaselineModel, **kwargs):
            """Extract term names from a BaselineModel."""
            if hasattr(model, 'terms'):
                names = []
                for term in model.terms:
                    if hasattr(term, 'varname'):
                        names.append(term.varname)
                    elif hasattr(term, 'name'):
                        names.append(term.name)
                return names
            return []
            
    except ImportError:
        pass


# Register SamplingFrame implementations when available
def _register_sampling_frame():
    """Register SamplingFrame implementations after imports are resolved."""
    from .._warnings import suppress_fmrimod_warnings

    try:
        # Import the actual SamplingFrame class from the sampling module
        with suppress_fmrimod_warnings():
            from ..sampling import SamplingFrame as _SamplingFrameClass
        
        @blocklens.register(_SamplingFrameClass)
        def _blocklens_sampling_frame(sframe: _SamplingFrameClass, **kwargs):
            """Extract block lengths from a SamplingFrame."""
            if hasattr(sframe, 'blocklens'):
                return sframe.blocklens
            elif hasattr(sframe, 'block_lengths'):
                return sframe.block_lengths
            elif hasattr(sframe, '_blocklens'):
                return sframe._blocklens
            else:
                raise ValueError("SamplingFrame has no block length information")
        
        @blockids.register(_SamplingFrameClass)
        def _blockids_sampling_frame(sframe: _SamplingFrameClass, **kwargs):
            """Extract block IDs from a SamplingFrame."""
            # Generate block IDs from block lengths
            blocklens = _blocklens_sampling_frame(sframe)
            blockids = []
            for i, length in enumerate(blocklens):
                blockids.extend([i + 1] * length)  # 1-indexed like R
            return np.array(blockids)
            
    except ImportError:
        pass


# Call registration functions at module load
_register_event_model()
_register_baseline_model()
_register_sampling_frame()


# New generic functions for feature parity

@singledispatch
def cells(x, drop_empty: bool = True, **kwargs) -> pd.DataFrame:
    """Extract cells (factor-level combinations) from an object.

    For categorical events or terms, returns a DataFrame listing all
    possible combinations of factor levels. For continuous events,
    returns a single row representing the variable.

    This is a generic function that dispatches based on the type of
    ``x``. Implementations are registered for ``EventFactor``,
    ``EventVariable``, ``EventTerm``, and related types.

    Parameters
    ----------
    x : object
        Object to extract cells from (e.g., ``EventTerm``,
        ``EventFactor``).
    drop_empty : bool, default=True
        Whether to drop cells with zero observations.
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    pandas.DataFrame
        DataFrame where each row is a cell and columns are
        factor names. Has a ``count`` attribute when available.

    See Also
    --------
    conditions : Extract condition name strings.
    levels : Extract factor levels.
    """
    raise NotImplementedError(f"cells not implemented for {type(x)}")


@singledispatch
def conditions(x, drop_empty: bool = True, expand_basis: bool = False, **kwargs) -> List[str]:
    """Extract condition names from an object.

    Generates descriptive name strings for all conditions represented
    by the object. For categorical events, creates ``'factor.level'``
    tokens; for continuous events, returns the variable name; for
    interaction terms, returns cross-product tokens.

    Parameters
    ----------
    x : object
        Object to extract conditions from (e.g., ``EventTerm``,
        ``EventFactor``).
    drop_empty : bool, default=True
        Whether to exclude conditions with no observations.
    expand_basis : bool, default=False
        Whether to expand names with basis function suffixes
        (``_b01``, ``_b02``, ...) for multi-basis HRFs.
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    list of str
        Condition name strings.

    See Also
    --------
    cells : Extract factor-level combinations as DataFrame.
    columns : Extract design matrix column names.
    """
    raise NotImplementedError(f"conditions not implemented for {type(x)}")


def _display_condition_name(canonical: Any) -> Any:
    """Convert canonical condition tags to compact display names."""
    if canonical is None:
        return None
    parts = str(canonical).split("_")
    labels = [part.split(".")[-1] for part in parts]
    return ":".join(labels)


@singledispatch
def condition_map(
    x,
    drop_empty: bool = True,
    expand_basis: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Map compact display condition names to canonical condition names.

    This mirrors the fmridesign ``condition_map()`` generic while keeping the
    result in a pandas DataFrame. Event terms return ``display`` and
    ``canonical`` columns. Event models add ``term`` and ``column_name``.
    """
    raise NotImplementedError(f"condition_map not implemented for {type(x)}")


@singledispatch
def onsets(x, **kwargs) -> Union[np.ndarray, List[float]]:
    """Extract event onset times from an object.

    Returns the onset time of each event in seconds relative to
    the start of the experiment.

    Parameters
    ----------
    x : object
        Object to extract onsets from (e.g., ``EventFactor``,
        ``EventVariable``, ``EventTerm``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    numpy.ndarray
        Event onset times in seconds.

    See Also
    --------
    durations : Extract event durations.
    """
    raise NotImplementedError(f"onsets not implemented for {type(x)}")


@singledispatch
def durations(x, **kwargs) -> Union[np.ndarray, List[float]]:
    """Extract event durations from an object.

    Returns the duration of each event in seconds. A duration of 0
    indicates an instantaneous (impulse) event.

    Parameters
    ----------
    x : object
        Object to extract durations from (e.g., ``EventFactor``,
        ``EventVariable``, ``EventTerm``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    numpy.ndarray
        Event durations in seconds.

    See Also
    --------
    onsets : Extract event onset times.
    """
    raise NotImplementedError(f"durations not implemented for {type(x)}")


@singledispatch
def elements(x, what: str = "values", transformed: bool = True, **kwargs) -> Any:
    """Extract elements (values or labels) from an event object.

    Parameters
    ----------
    x : object
        Object to extract elements from (e.g., ``EventFactor``,
        ``EventVariable``).
    what : {'values', 'labels'}, default='values'
        Type of elements to extract. ``'values'`` returns the
        per-event data; ``'labels'`` returns level or column names.
    transformed : bool, default=True
        Whether to return transformed (basis-applied) values.
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    array-like or list of str
        Extracted elements.

    See Also
    --------
    labels : Extract labels directly.
    levels : Extract factor levels.
    """
    raise NotImplementedError(f"elements not implemented for {type(x)}")


@singledispatch
def labels(x, **kwargs) -> List[str]:
    """Extract labels from an event object.

    For categorical events, returns the factor levels. For continuous
    events, returns the variable name. For terms, returns labels
    from all component events.

    Parameters
    ----------
    x : object
        Object to extract labels from (e.g., ``EventFactor``,
        ``EventTerm``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    list of str
        Labels.

    See Also
    --------
    levels : Extract factor levels (None for continuous).
    conditions : Extract condition name strings.
    """
    raise TypeError(f"labels not implemented for {type(x)}")


@singledispatch
def levels(x, **kwargs) -> Union[List[str], None]:
    """Extract factor levels from an object.

    Returns the ordered list of unique levels for categorical
    events. Returns None for continuous events.

    Parameters
    ----------
    x : object
        Object to extract levels from (e.g., ``EventFactor``,
        ``EventTerm``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    list of str or None
        Factor levels if categorical, None if continuous.

    See Also
    --------
    labels : Extract labels (always returns strings).
    is_categorical : Check if an object is categorical.
    """
    raise NotImplementedError(f"levels not implemented for {type(x)}")


@singledispatch
def columns(x, **kwargs) -> List[str]:
    """Extract design matrix column names from an object.

    Returns the names that would appear as column headers in the
    design matrix produced by this object. For factors, these are
    ``'factor.level'`` tokens; for continuous variables, the
    variable name; for models, all column names.

    Parameters
    ----------
    x : object
        Object to extract columns from (e.g., ``EventFactor``,
        ``EventModel``, ``numpy.ndarray``, ``pandas.DataFrame``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    list of str
        Column names.

    See Also
    --------
    conditions : Extract condition names (may differ from columns
        when basis expansion is involved).
    """
    raise TypeError(f"columns not implemented for {type(x)}")


@singledispatch
def nbasis(x, **kwargs) -> int:
    """Extract the number of basis functions from an object.

    For events with multi-basis HRFs (e.g., SPMG3), returns the
    number of basis functions per condition. For simple events,
    returns 1. For terms, returns the product of nbasis across
    component events.

    Parameters
    ----------
    x : object
        Object to query (e.g., ``EventFactor``, ``EventTerm``,
        ``ParametricBasis``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    int
        Number of basis functions.
    """
    raise NotImplementedError(f"nbasis not implemented for {type(x)}")


@singledispatch
def is_categorical(x, **kwargs) -> bool:
    """Check if an object represents categorical (factor) data.

    Parameters
    ----------
    x : object
        Object to check (e.g., ``EventFactor``, ``EventTerm``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    bool
        True if the object is categorical, False otherwise.

    See Also
    --------
    is_continuous : Check for continuous data.
    levels : Extract factor levels.
    """
    raise NotImplementedError(f"is_categorical not implemented for {type(x)}")


@singledispatch
def is_continuous(x, **kwargs) -> bool:
    """Check if an object represents continuous (numeric) data.

    Parameters
    ----------
    x : object
        Object to check (e.g., ``EventVariable``, ``EventTerm``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    bool
        True if the object is continuous, False otherwise.

    See Also
    --------
    is_categorical : Check for categorical data.
    """
    raise NotImplementedError(f"is_continuous not implemented for {type(x)}")


@singledispatch
def event_terms(x, **kwargs) -> List[Any]:
    """Extract event terms from a model.

    Returns the list of ``EventTerm`` objects that define the
    task-related regressors in the model.

    Parameters
    ----------
    x : object
        Model object to extract terms from (typically
        ``EventModel``).
    **kwargs
        Additional arguments for specific implementations.

    Returns
    -------
    list of EventTerm
        Event terms in model order.

    See Also
    --------
    term_names : Extract term name strings.
    """
    raise NotImplementedError(f"event_terms not implemented for {type(x)}")


@singledispatch
def construct(x, *args, **kwargs) -> Any:
    """Construct a model component from a specification object.

    Materializes a specification (e.g., ``NuisanceSpec``,
    ``BaselineSpec``) into the concrete object it describes,
    given a sampling frame and any additional context.

    Parameters
    ----------
    x : object
        Specification object to construct (e.g., ``NuisanceSpec``,
        ``BlockSpec``, ``BaselineSpec``).
    *args
        Positional arguments for construction (typically a
        sampling frame).
    **kwargs
        Keyword arguments for construction.

    Returns
    -------
    object
        Constructed component (e.g., nuisance data,
        ``BaselineModel``).
    """
    raise NotImplementedError(f"construct not implemented for {type(x)}")


def evaluate(x, *args, **kwargs):
    """Evaluate an object with an ``evaluate`` method or callable interface.

    This is a lightweight compatibility helper for the R ``evaluate()`` generic.
    It intentionally delegates to the object's Python method rather than
    introducing a parallel dispatch system.
    """
    method = getattr(x, "evaluate", None)
    if callable(method):
        return method(*args, **kwargs)
    if callable(x):
        return x(*args, **kwargs)
    raise NotImplementedError(f"evaluate not implemented for {type(x)}")


def acquisition_onsets(x, **kwargs) -> np.ndarray:
    """Return global fMRI acquisition onset times from a sampling frame."""
    val = getattr(x, "acquisition_onsets", None)
    if callable(val):
        return np.asarray(val(**kwargs))
    if val is not None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unexpected}")
        return np.asarray(val)
    raise NotImplementedError(f"acquisition_onsets not implemented for {type(x)}")


def amplitudes(x, **kwargs) -> np.ndarray:
    """Return event amplitudes from a regressor-like object."""
    method = getattr(x, "amplitudes", None)
    if callable(method):
        return np.asarray(method(**kwargs))
    val = getattr(x, "amplitude", None)
    if val is not None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unexpected}")
        return np.asarray(val)
    raise NotImplementedError(f"amplitudes not implemented for {type(x)}")


def samples(x, global_time: bool = True, **kwargs) -> np.ndarray:
    """Return sampling times from a sampling-frame-like object.

    Parameters
    ----------
    x : object
        Object exposing ``sample_times`` or ``samples``.
    global_time : bool, default True
        Whether to return global scan times. The R-style keyword ``global`` is
        accepted through ``**kwargs``.
    """
    blockids = kwargs.pop("blockids", None)
    if "global" in kwargs:
        global_time = bool(kwargs.pop("global"))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unexpected}")
    if hasattr(x, "sample_times"):
        return np.asarray(x.sample_times(global_time=global_time, blockids=blockids))
    if blockids is not None:
        raise NotImplementedError(f"samples block filtering not implemented for {type(x)}")
    val = getattr(x, "samples", None)
    if callable(val):
        return np.asarray(val())
    if val is not None and global_time:
        return np.asarray(val)
    raise NotImplementedError(f"samples not implemented for {type(x)}")


def global_onsets(x, onsets, blockids, **kwargs) -> np.ndarray:
    """Convert block-local onsets to global experiment onsets."""
    method = getattr(x, "global_onsets", None)
    if callable(method):
        return np.asarray(method(onsets, blockids, **kwargs))
    raise NotImplementedError(f"global_onsets not implemented for {type(x)}")


def shift(x, shift_amount=None, **kwargs):
    """Shift a regressor-like object by a temporal offset."""
    if shift_amount is None and "offset" in kwargs:
        shift_amount = kwargs.pop("offset")
    if shift_amount is None:
        raise TypeError("shift requires `shift_amount` or `offset`")
    method = getattr(x, "shift", None)
    if callable(method):
        return method(shift_amount, **kwargs)
    raise NotImplementedError(f"shift not implemented for {type(x)}")


# Register EventTerm implementations after generics are defined
def _register_event_term():
    """Register EventTerm implementations after imports are resolved."""
    try:
        from ..events.cells import cells_event_term, conditions_event_term
        from ..events.term import EventTerm
        
        @cells.register(EventTerm)
        def _cells_event_term(term: EventTerm, drop_empty: bool = True, **kwargs):
            """Extract cells from an EventTerm."""
            return cells_event_term(term, drop_empty)
        
        @conditions.register(EventTerm)
        def _conditions_event_term(term: EventTerm, drop_empty: bool = True, 
                                   expand_basis: bool = False, **kwargs):
            """Extract conditions from an EventTerm."""
            return conditions_event_term(term, drop_empty, expand_basis)

        @condition_map.register(EventTerm)
        def _condition_map_event_term(term: EventTerm, drop_empty: bool = True,
                                      expand_basis: bool = False, **kwargs):
            """Map EventTerm display labels to canonical condition names."""
            canonical = conditions_event_term(term, drop_empty, expand_basis)
            display = [_display_condition_name(cond) for cond in canonical]
            return pd.DataFrame({
                "display": display,
                "canonical": canonical,
            })
            
    except ImportError:
        pass

_register_event_term()


# Register event implementations for extraction functions
def _register_event_extractors():
    """Register implementations for event extraction functions."""
    try:
        from ..events.basis import EventBasis
        from ..events.factor import EventFactor
        from ..events.matrix import EventMatrix
        from ..events.term import EventTerm
        from ..events.variable import EventVariable
        
        # Onsets implementations
        @onsets.register(EventFactor)
        def _onsets_event_factor(event: EventFactor, **kwargs):
            """Extract onsets from EventFactor."""
            return np.array(event.onsets)
        
        @onsets.register(EventVariable)
        def _onsets_event_variable(event: EventVariable, **kwargs):
            """Extract onsets from EventVariable."""
            return np.array(event.onsets)
        
        @onsets.register(EventMatrix)
        def _onsets_event_matrix(event: EventMatrix, **kwargs):
            """Extract onsets from EventMatrix."""
            return np.array(event.onsets)
        
        @onsets.register(EventBasis)
        def _onsets_event_basis(event: EventBasis, **kwargs):
            """Extract onsets from EventBasis."""
            return np.array(event.onsets)
        
        @onsets.register(EventTerm)
        def _onsets_event_term(term: EventTerm, **kwargs):
            """Extract onsets from EventTerm."""
            # Get onsets from first event (all should have same onsets)
            if term.events:
                return np.array(term.events[0].onsets)
            return np.array([])
        
        # Durations implementations
        @durations.register(EventFactor)
        def _durations_event_factor(event: EventFactor, **kwargs):
            """Extract durations from EventFactor."""
            return np.array(event.durations)
        
        @durations.register(EventVariable)
        def _durations_event_variable(event: EventVariable, **kwargs):
            """Extract durations from EventVariable."""
            return np.array(event.durations)
        
        @durations.register(EventMatrix)
        def _durations_event_matrix(event: EventMatrix, **kwargs):
            """Extract durations from EventMatrix."""
            return np.array(event.durations)
        
        @durations.register(EventBasis)
        def _durations_event_basis(event: EventBasis, **kwargs):
            """Extract durations from EventBasis."""
            return np.array(event.durations)
        
        @durations.register(EventTerm)
        def _durations_event_term(term: EventTerm, **kwargs):
            """Extract durations from EventTerm."""
            # Get durations from first event (all should have same durations)
            if term.events:
                return np.array(term.events[0].durations)
            return np.array([])
        
        # Elements implementations
        @elements.register(EventFactor)
        def _elements_event_factor(event: EventFactor, what: str = "values", 
                                  transformed: bool = True, **kwargs):
            """Extract elements from EventFactor."""
            if what == "labels":
                return event.levels
            else:  # values
                return event.values
        
        @elements.register(EventVariable)
        def _elements_event_variable(event: EventVariable, what: str = "values", 
                                    transformed: bool = True, **kwargs):
            """Extract elements from EventVariable."""
            if what == "labels":
                return [event.name]
            else:  # values
                return event.values
        
        @elements.register(EventMatrix)
        def _elements_event_matrix(event: EventMatrix, what: str = "values", 
                                  transformed: bool = True, **kwargs):
            """Extract elements from EventMatrix."""
            if what == "labels":
                return event.column_names
            else:  # values
                return event.values
        
        @elements.register(EventBasis)
        def _elements_event_basis(event: EventBasis, what: str = "values", 
                                 transformed: bool = True, **kwargs):
            """Extract elements from EventBasis."""
            if what == "labels":
                return event.basis_names
            else:  # values
                return event.values
            
    except ImportError:
        pass

_register_event_extractors()


# Register type checking implementations  
def _register_type_checkers():
    """Register implementations for type checking functions."""
    try:
        from ..events.basis import EventBasis
        from ..events.factor import EventFactor
        from ..events.matrix import EventMatrix
        from ..events.term import EventTerm
        from ..events.variable import EventVariable
        
        # is_categorical implementations
        @is_categorical.register(EventFactor)
        def _is_categorical_event_factor(event: EventFactor, **kwargs):
            """Check if EventFactor is categorical."""
            return True
        
        @is_categorical.register(EventVariable)
        def _is_categorical_event_variable(event: EventVariable, **kwargs):
            """Check if EventVariable is categorical."""
            return False
        
        @is_categorical.register(EventMatrix)
        def _is_categorical_event_matrix(event: EventMatrix, **kwargs):
            """Check if EventMatrix is categorical."""
            return False
        
        @is_categorical.register(EventBasis)
        def _is_categorical_event_basis(event: EventBasis, **kwargs):
            """Check if EventBasis is categorical."""
            return False
        
        @is_categorical.register(EventTerm)
        def _is_categorical_event_term(term: EventTerm, **kwargs):
            """Check if EventTerm is categorical."""
            return term.is_categorical
        
        # is_continuous implementations
        @is_continuous.register(EventFactor)
        def _is_continuous_event_factor(event: EventFactor, **kwargs):
            """Check if EventFactor is continuous."""
            return False
        
        @is_continuous.register(EventVariable)
        def _is_continuous_event_variable(event: EventVariable, **kwargs):
            """Check if EventVariable is continuous."""
            return True
        
        @is_continuous.register(EventMatrix)
        def _is_continuous_event_matrix(event: EventMatrix, **kwargs):
            """Check if EventMatrix is continuous."""
            return True
        
        @is_continuous.register(EventBasis)
        def _is_continuous_event_basis(event: EventBasis, **kwargs):
            """Check if EventBasis is continuous."""
            return True
        
        @is_continuous.register(EventTerm)
        def _is_continuous_event_term(term: EventTerm, **kwargs):
            """Check if EventTerm is continuous."""
            return term.is_continuous
            
    except ImportError:
        pass

_register_type_checkers()


# Register cells/conditions implementations for event classes
def _register_event_cells_conditions():
    """Register cells and conditions implementations for event classes."""
    try:
        from ..events.basis import EventBasis
        from ..events.cells import cells_event_term, conditions_event_term
        from ..events.factor import EventFactor
        from ..events.matrix import EventMatrix
        from ..events.variable import EventVariable
        
        # For EventFactor - wrap in EventTerm to use existing logic
        @cells.register(EventFactor)
        def _cells_event_factor(event: EventFactor, drop_empty: bool = True, **kwargs):
            """Extract cells from EventFactor."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return cells_event_term(term, drop_empty)
        
        @conditions.register(EventFactor)
        def _conditions_event_factor(event: EventFactor, drop_empty: bool = True,
                                    expand_basis: bool = False, **kwargs):
            """Extract conditions from EventFactor."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return conditions_event_term(term, drop_empty, expand_basis)

        @condition_map.register(EventFactor)
        def _condition_map_event_factor(event: EventFactor, drop_empty: bool = True,
                                        expand_basis: bool = False, **kwargs):
            """Map EventFactor display labels to canonical condition names."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return condition_map(term, drop_empty=drop_empty, expand_basis=expand_basis)
        
        # For EventVariable - wrap in EventTerm
        @cells.register(EventVariable)
        def _cells_event_variable(event: EventVariable, drop_empty: bool = True, **kwargs):
            """Extract cells from EventVariable."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return cells_event_term(term, drop_empty)
        
        @conditions.register(EventVariable)
        def _conditions_event_variable(event: EventVariable, drop_empty: bool = True,
                                      expand_basis: bool = False, **kwargs):
            """Extract conditions from EventVariable."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return conditions_event_term(term, drop_empty, expand_basis)

        @condition_map.register(EventVariable)
        def _condition_map_event_variable(event: EventVariable, drop_empty: bool = True,
                                          expand_basis: bool = False, **kwargs):
            """Map EventVariable display labels to canonical condition names."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return condition_map(term, drop_empty=drop_empty, expand_basis=expand_basis)
        
        # For EventMatrix - wrap in EventTerm
        @cells.register(EventMatrix)
        def _cells_event_matrix(event: EventMatrix, drop_empty: bool = True, **kwargs):
            """Extract cells from EventMatrix."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return cells_event_term(term, drop_empty)
        
        @conditions.register(EventMatrix)
        def _conditions_event_matrix(event: EventMatrix, drop_empty: bool = True,
                                    expand_basis: bool = False, **kwargs):
            """Extract conditions from EventMatrix."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return conditions_event_term(term, drop_empty, expand_basis)

        @condition_map.register(EventMatrix)
        def _condition_map_event_matrix(event: EventMatrix, drop_empty: bool = True,
                                        expand_basis: bool = False, **kwargs):
            """Map EventMatrix display labels to canonical condition names."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return condition_map(term, drop_empty=drop_empty, expand_basis=expand_basis)
        
        # For EventBasis - wrap in EventTerm
        @cells.register(EventBasis)
        def _cells_event_basis(event: EventBasis, drop_empty: bool = True, **kwargs):
            """Extract cells from EventBasis."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return cells_event_term(term, drop_empty)
        
        @conditions.register(EventBasis)
        def _conditions_event_basis(event: EventBasis, drop_empty: bool = True,
                                   expand_basis: bool = False, **kwargs):
            """Extract conditions from EventBasis."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return conditions_event_term(term, drop_empty, expand_basis)

        @condition_map.register(EventBasis)
        def _condition_map_event_basis(event: EventBasis, drop_empty: bool = True,
                                       expand_basis: bool = False, **kwargs):
            """Map EventBasis display labels to canonical condition names."""
            from ..events.term import EventTerm
            term = EventTerm([event])
            return condition_map(term, drop_empty=drop_empty, expand_basis=expand_basis)
            
    except ImportError:
        pass

_register_event_cells_conditions()


# Register event_terms implementations
def _register_event_terms():
    """Register event_terms implementations for models."""
    try:
        from ..design.event_model import EventModel
        
        @event_terms.register(EventModel)
        def _event_terms_event_model(model: EventModel, **kwargs):
            """Extract event terms from EventModel."""
            # EventModel stores terms internally (cached after design_matrix build)
            if model._event_terms is not None:
                return model._event_terms
            elif hasattr(model, 'terms'):
                # Try to create event terms via the model's method
                return model._create_event_terms()
            else:
                return []
            
    except ImportError:
        pass

_register_event_terms()


# Register labels and levels implementations
def _register_labels_levels():
    """Register labels and levels implementations for events."""
    try:
        from ..events.basis import EventBasis
        from ..events.factor import EventFactor
        from ..events.matrix import EventMatrix
        from ..events.term import EventTerm
        from ..events.variable import EventVariable
        
        # labels implementations
        @labels.register(EventFactor)
        def _labels_event_factor(event: EventFactor, **kwargs):
            """Extract labels from EventFactor."""
            return list(event.levels)
        
        @labels.register(EventVariable)
        def _labels_event_variable(event: EventVariable, **kwargs):
            """Extract labels from EventVariable."""
            return [event.name]
        
        @labels.register(EventMatrix)
        def _labels_event_matrix(event: EventMatrix, **kwargs):
            """Extract labels from EventMatrix."""
            return list(event.column_names)
        
        @labels.register(EventBasis)
        def _labels_event_basis(event: EventBasis, **kwargs):
            """Extract labels from EventBasis."""
            return event.basis_names
        
        @labels.register(EventTerm)
        def _labels_event_term(term: EventTerm, **kwargs):
            """Extract labels from EventTerm."""
            all_labels = []
            for event in term.events:
                event_labels = labels(event)
                all_labels.extend(event_labels)
            return all_labels
        
        # levels implementations
        @levels.register(EventFactor)
        def _levels_event_factor(event: EventFactor, **kwargs):
            """Extract levels from EventFactor."""
            return list(event.levels)
        
        @levels.register(EventVariable)
        def _levels_event_variable(event: EventVariable, **kwargs):
            """Extract levels from EventVariable."""
            return None  # Continuous variables have no levels
        
        @levels.register(EventMatrix)
        def _levels_event_matrix(event: EventMatrix, **kwargs):
            """Extract levels from EventMatrix."""
            return None  # Matrix events have no levels
        
        @levels.register(EventBasis)
        def _levels_event_basis(event: EventBasis, **kwargs):
            """Extract levels from EventBasis."""
            return None  # Basis events have no levels
        
        @levels.register(EventTerm)
        def _levels_event_term(term: EventTerm, **kwargs):
            """Extract levels from EventTerm."""
            if term.is_categorical:
                # For purely categorical terms, return level combinations
                level_lists = []
                for event in term.events:
                    if hasattr(event, 'levels'):
                        level_lists.append(list(event.levels))
                
                if len(level_lists) == 1:
                    return level_lists[0]
                else:
                    # Return combinations for interaction terms
                    from itertools import product
                    combinations = list(product(*level_lists))
                    return [':'.join(combo) for combo in combinations]
            return None
            
    except ImportError:
        pass

_register_labels_levels()


# Register columns implementations
def _register_columns():
    """Register columns implementations for various objects."""
    try:
        from ..design.event_model import EventModel
        from ..events.basis import EventBasis
        from ..events.factor import EventFactor
        from ..events.matrix import EventMatrix
        from ..events.term import EventTerm
        from ..events.variable import EventVariable

        @columns.register(EventFactor)
        def _columns_event_factor(event: EventFactor, **kwargs):
            """Extract columns from EventFactor."""
            return [f"{event.name}.{level}" for level in event.levels]

        @columns.register(EventVariable)
        def _columns_event_variable(event: EventVariable, **kwargs):
            """Extract columns from EventVariable."""
            return [event.name]

        @columns.register(EventMatrix)
        def _columns_event_matrix(event: EventMatrix, **kwargs):
            """Extract columns from EventMatrix."""
            if hasattr(event, 'column_names'):
                return list(event.column_names)
            return [f"V{i+1}" for i in range(event.n_columns)]

        @columns.register(EventBasis)
        def _columns_event_basis(event: EventBasis, **kwargs):
            """Extract columns from EventBasis."""
            if hasattr(event, 'basis_names'):
                return list(event.basis_names)
            return [event.name]

        @columns.register(EventTerm)
        def _columns_event_term(term: EventTerm, **kwargs):
            """Extract columns from EventTerm."""
            return term.get_column_names()

        @columns.register(EventModel)
        def _columns_event_model(model: EventModel, **kwargs):
            """Extract columns from EventModel."""
            return model.column_names

        # Register longnames and shortnames for EventModel
        @longnames.register(EventModel)
        def _longnames_event_model(model: EventModel, **kwargs):
            """Extract long names from EventModel."""
            return model.longnames()

        @shortnames.register(EventModel)
        def _shortnames_event_model(model: EventModel, **kwargs):
            """Extract short names from EventModel."""
            return model.shortnames()

    except ImportError:
        pass

    # Baseline model - separate try/except to not fail if not available
    try:
        from ..baseline.baseline_model import BaselineModel

        @columns.register(BaselineModel)
        def _columns_baseline_model(model: BaselineModel, **kwargs):
            """Extract columns from BaselineModel."""
            if hasattr(model, "column_names"):
                return list(model.column_names)
            if hasattr(model, 'colnames'):
                return list(model.colnames)
            return []
    except ImportError:
        pass

    # For numpy arrays
    @columns.register(np.ndarray)
    def _columns_array(arr: np.ndarray, **kwargs):
        """Extract columns from numpy array."""
        if arr.ndim == 1:
            return ["V1"]
        elif arr.ndim == 2:
            return [f"V{i+1}" for i in range(arr.shape[1])]
        else:
            raise ValueError(f"Cannot extract columns from {arr.ndim}D array")

    # For pandas DataFrames
    @columns.register(pd.DataFrame)
    def _columns_dataframe(df: pd.DataFrame, **kwargs):
        """Extract columns from DataFrame."""
        return list(df.columns)

_register_columns()


# Register construct implementations for specs
def _register_construct():
    """Register construct implementations for spec objects."""
    try:
        from ..baseline.specs import BlockSpec, NuisanceSpec

        @construct.register(NuisanceSpec)
        def _construct_nuisance_spec(spec: NuisanceSpec, sampling_frame, **kwargs):
            """Construct nuisance variable from spec."""
            return spec.data

        @construct.register(BlockSpec)
        def _construct_block_spec(spec: BlockSpec, sampling_frame, **kwargs):
            """Construct block variable from spec."""
            return spec.label
    except ImportError:
        pass

    try:
        from ..baseline.baseline_model import BaselineSpec

        @construct.register(BaselineSpec)
        def _construct_baseline_spec(spec: BaselineSpec, sampling_frame, **kwargs):
            """Construct baseline model from spec."""
            from ..baseline import baseline_model
            return baseline_model(
                spec.basis,
                degree=spec.degree,
                sframe=sampling_frame,
                **kwargs
            )
            
    except ImportError:
        pass

_register_construct()


# Register nbasis implementations
def _register_nbasis():
    """Register nbasis implementations for various objects."""
    try:
        from ..basis.base import ParametricBasis
        from ..events.basis import EventBasis
        from ..events.factor import EventFactor
        from ..events.matrix import EventMatrix
        from ..events.term import EventTerm
        from ..events.variable import EventVariable

        @nbasis.register(EventFactor)
        def _nbasis_event_factor(event: EventFactor, **kwargs):
            """Extract nbasis from EventFactor."""
            # Factors have nlevels - 1 degrees of freedom (contrast coding)
            return max(0, len(event.levels) - 1)

        @nbasis.register(EventVariable)
        def _nbasis_event_variable(event: EventVariable, **kwargs):
            """Extract nbasis from EventVariable."""
            return 1  # Continuous variables have 1 basis function

        @nbasis.register(EventMatrix)
        def _nbasis_event_matrix(event: EventMatrix, **kwargs):
            """Extract nbasis from EventMatrix."""
            if hasattr(event, 'n_columns'):
                return event.n_columns
            elif hasattr(event, 'values') and event.values is not None:
                return event.values.shape[1] if event.values.ndim > 1 else 1
            return 1

        @nbasis.register(EventBasis)
        def _nbasis_event_basis(event: EventBasis, **kwargs):
            """Extract nbasis from EventBasis."""
            if hasattr(event, 'n_basis'):
                return event.n_basis
            elif hasattr(event, 'basis_matrix') and event.basis_matrix is not None:
                return event.basis_matrix.shape[1]
            return 1

        @nbasis.register(EventTerm)
        def _nbasis_event_term(term: EventTerm, **kwargs):
            """Extract nbasis from EventTerm."""
            # Compute product of nbasis across events in the term
            result = 1
            for event in term.events:
                result *= nbasis(event)
            return result

        @nbasis.register(ParametricBasis)
        def _nbasis_parametric_basis(basis_obj: ParametricBasis, **kwargs):
            """Extract nbasis from ParametricBasis object."""
            if hasattr(basis_obj, 'nbasis'):
                return basis_obj.nbasis
            elif hasattr(basis_obj, 'n_basis'):
                return basis_obj.n_basis
            elif hasattr(basis_obj, 'degree'):
                return basis_obj.degree + 1
            else:
                # Try to evaluate at a dummy point
                try:
                    result = basis_obj.evaluate(np.array([0.5]))
                    return result.shape[1] if result.ndim > 1 else 1
                except Exception:
                    return 1
            
    except ImportError:
        pass

_register_nbasis()


# --- New generics: events, event_conditions, contrasts ---

@singledispatch
def events(x, drop_empty: bool = False, **kwargs) -> pd.DataFrame:
    """Return canonical event table with onset, duration, condition columns.

    This generic extracts a tidy event table from model objects,
    suitable for inspection, plotting, or round-tripping into new
    models.

    Parameters
    ----------
    x : EventTerm or EventModel
        Object to extract events from.
    drop_empty : bool, default False
        If True, remove rows whose condition label is empty or None.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``onset``, ``duration``, ``condition``
        (and optionally ``block`` for multi-run models).

    Raises
    ------
    NotImplementedError
        If no implementation is registered for the type of ``x``.

    See Also
    --------
    event_conditions : Return only the per-event condition labels.
    onsets : Extract onset times.
    durations : Extract durations.
    conditions : Extract unique condition labels.
    """
    raise NotImplementedError(f"events() not implemented for {type(x)}")


@singledispatch
def event_conditions(x, drop_empty: bool = False, **kwargs) -> np.ndarray:
    """Return per-event condition assignments.

    Unlike :func:`conditions`, which returns the unique set of
    condition labels, this function returns one label per event
    (i.e., the same length as the onset vector).

    Parameters
    ----------
    x : EventTerm or EventModel
        Object to extract condition assignments from.
    drop_empty : bool, default False
        If True, omit events whose condition label is empty or None.

    Returns
    -------
    numpy.ndarray
        1-D object array of condition labels, one per event.

    Raises
    ------
    NotImplementedError
        If no implementation is registered for the type of ``x``.

    See Also
    --------
    conditions : Return unique condition labels.
    events : Return full event table with onsets, durations, and conditions.
    """
    raise NotImplementedError(f"event_conditions() not implemented for {type(x)}")


@singledispatch
def contrasts(x, **kwargs):
    """Retrieve contrast specifications from an object.

    Unlike :func:`~fmrimod.contrast.contrast_weights.contrast_weights`,
    which computes numeric weight vectors, this function returns the
    raw :class:`~fmrimod.contrast.contrast_spec.ContrastSpec` objects
    attached to terms or models.

    Parameters
    ----------
    x : EventTerm or EventModel
        Object to retrieve contrast specifications from.

    Returns
    -------
    dict or list
        Mapping of contrast names to
        :class:`~fmrimod.contrast.contrast_spec.ContrastSpec` objects,
        or a list of specs when names are not available.

    Raises
    ------
    NotImplementedError
        If no implementation is registered for the type of ``x``.

    See Also
    --------
    conditions : Return unique condition labels.
    fmrimod.contrast.contrast_weights.contrast_weights :
        Compute numeric contrast weight vectors.
    fmrimod.validate.validate_contrasts :
        Validate contrast estimability.
    """
    raise NotImplementedError(f"contrasts() not implemented for {type(x)}")


def _register_events_event_conditions_contrasts():
    """Register events, event_conditions, and contrasts implementations."""
    try:
        from ..events.term import EventTerm
        from ..naming import level_token, make_cond_tag

        @events.register(EventTerm)
        def _events_event_term(x, drop_empty=False, **kwargs):
            """Extract canonical event table from EventTerm."""
            ons = np.array(x.events[0].onsets) if x.events else np.array([])

            dur = np.array(x.events[0].durations) if x.events else np.array([])
            if len(dur) == 0:
                dur = np.zeros(len(ons))
            elif len(dur) != len(ons):
                dur = np.resize(dur, len(ons))

            cond = event_conditions(x, drop_empty=drop_empty)

            return pd.DataFrame({
                'onset': ons,
                'duration': dur,
                'condition': cond,
            })

        @event_conditions.register(EventTerm)
        def _event_conditions_event_term(x, drop_empty=False, **kwargs):
            """Return condition labels for each event in the term."""
            n_events = len(x.events[0].onsets) if x.events else 0

            categorical_events = [e for e in x.events if e.event_type == "categorical"]

            # Continuous-only or intercept-only terms
            if not categorical_events:
                levels_out = ["condition"]
                if n_events > 0:
                    return np.array([levels_out[0]] * n_events, dtype=object)
                return np.array([], dtype=object)

            # Build level tokens per categorical event
            from itertools import product as iter_product
            level_tokens_list = []
            for ev in categorical_events:
                tokens = [level_token(ev.name, lev) for lev in ev.levels]
                level_tokens_list.append(tokens)

            # Canonical condition levels via product
            if len(level_tokens_list) == 1:
                cond_levels = level_tokens_list[0]
            else:
                combos = list(iter_product(*level_tokens_list))
                cond_levels = [make_cond_tag(list(combo)) for combo in combos]

            if not cond_levels:
                return np.array([], dtype=object)

            # Per-event tokens
            per_event_tokens = []
            for ev in categorical_events:
                tok = [level_token(ev.name, lev) for lev in ev.levels]
                vals = ev.values
                event_labels = []
                for v in vals:
                    try:
                        idx = list(ev.levels).index(v)
                        event_labels.append(tok[idx])
                    except (ValueError, IndexError):
                        event_labels.append(None)
                per_event_tokens.append(event_labels)

            # Combine per-event tokens into condition labels
            event_cond_labels = []
            for i in range(n_events):
                parts = []
                for pet in per_event_tokens:
                    if i < len(pet) and pet[i] is not None:
                        parts.append(pet[i])
                if not parts:
                    event_cond_labels.append(None)
                elif len(parts) == 1:
                    event_cond_labels.append(parts[0])
                else:
                    event_cond_labels.append(make_cond_tag(parts))

            # Map to canonical levels (get integer ids)
            ids = []
            for label in event_cond_labels:
                if label in cond_levels:
                    ids.append(cond_levels.index(label) + 1)  # 1-indexed
                else:
                    ids.append(None)

            if not drop_empty:
                result = []
                for idx in ids:
                    if idx is not None:
                        result.append(cond_levels[idx - 1])
                    else:
                        result.append(None)
                return np.array(result, dtype=object)
            else:
                # Drop empty: only keep levels that are present
                present = sorted(set(i for i in ids if i is not None))
                if not present:
                    return np.array([None] * len(ids), dtype=object)
                lev_drop = [cond_levels[p - 1] for p in present]
                result = []
                for idx in ids:
                    if idx is not None and idx in present:
                        new_idx = present.index(idx)
                        result.append(lev_drop[new_idx])
                    else:
                        result.append(None)
                return np.array(result, dtype=object)

    except ImportError:
        pass

    try:
        from ..design.event_model import EventModel

        @condition_map.register(EventModel)
        def _condition_map_event_model(x, drop_empty=True, expand_basis=False, **kwargs):
            """Map EventModel display/canonical names to design column names."""
            rows = []
            event_term_list = x._create_event_terms()
            column_names = list(getattr(x, "column_names", []) or [])
            column_indices = getattr(x, "column_indices", {}) or {}

            for term_obj, event_term in zip(x.terms, event_term_list):
                term_name = getattr(term_obj, "name", getattr(event_term, "name", "term"))
                term_map = condition_map(
                    event_term,
                    drop_empty=drop_empty,
                    expand_basis=expand_basis,
                    **kwargs,
                )
                term_cols = [
                    column_names[idx]
                    for idx in column_indices.get(term_name, [])
                    if idx < len(column_names)
                ]
                if len(term_cols) == len(term_map):
                    mapped_cols = term_cols
                else:
                    mapped_cols = [None] * len(term_map)
                for i, row in term_map.reset_index(drop=True).iterrows():
                    rows.append({
                        "term": term_name,
                        "display": row["display"],
                        "canonical": row["canonical"],
                        "column_name": mapped_cols[i],
                    })

            return pd.DataFrame(
                rows,
                columns=["term", "display", "canonical", "column_name"],
            )

        @events.register(EventModel)
        def _events_event_model(x, drop_empty=False, **kwargs):
            """Extract events from all terms in model."""
            all_events = []
            event_term_list = x._create_event_terms()
            for et in event_term_list:
                term_events = events(et, drop_empty=drop_empty)
                term_events['term'] = et.name
                all_events.append(term_events)
            if all_events:
                return pd.concat(all_events, ignore_index=True)
            return pd.DataFrame(columns=['onset', 'duration', 'condition', 'term'])

        @contrasts.register(EventModel)
        def _contrasts_event_model(x, **kwargs):
            """Collect contrast specifications from all terms."""
            all_contrasts = {}
            for term in x.terms:
                term_contrasts = None
                # Check _kwargs for contrasts
                if hasattr(term, '_kwargs') and 'contrasts' in term._kwargs:
                    term_contrasts = term._kwargs['contrasts']
                elif hasattr(term, 'contrasts') and term.contrasts is not None:
                    term_contrasts = term.contrasts

                if term_contrasts is None:
                    continue

                if isinstance(term_contrasts, dict):
                    # Prefix keys with term name
                    for cname, cspec in term_contrasts.items():
                        all_contrasts[f"{term.name}.{cname}"] = cspec
                elif isinstance(term_contrasts, list):
                    for c in term_contrasts:
                        cname = getattr(c, 'name', str(c))
                        all_contrasts[f"{term.name}.{cname}"] = c
                else:
                    # Single contrast object
                    cname = getattr(term_contrasts, 'name', 'contrast')
                    all_contrasts[f"{term.name}.{cname}"] = term_contrasts
            return all_contrasts

    except ImportError:
        pass

_register_events_event_conditions_contrasts()
