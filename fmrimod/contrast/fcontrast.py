"""F-contrast matrices for testing main effects and interactions."""

from __future__ import annotations

import itertools
from functools import reduce, singledispatch
from typing import Any, Dict, Optional

import numpy as np

from ..design.event_model import EventModel

# Need actual imports for singledispatch registration
from ..events.factor import EventFactor
from ..events.term import EventTerm
from ..events.variable import EventVariable
from ..types import Array


@singledispatch
def Fcontrasts(x: Any, max_inter: int = 4, **kwargs: Any) -> Dict[str, Array]:
    """Generate F-contrast matrices for testing main effects and interactions.

    This is a :func:`functools.singledispatch` generic function: ``x``
    is intentionally ``Any`` because the dispatcher routes on the
    runtime type (registered for :class:`EventTerm`, :class:`EventModel`,
    and the categorical event variants). The registered implementations
    carry their own typed signatures.

    F-contrasts are used to test hypotheses about multiple parameters
    simultaneously. For categorical variables, this generates contrast
    matrices for main effects and interactions up to a specified order.

    Parameters
    ----------
    x : object
        Object to generate F-contrasts for (event_term, event_model, etc.)
    max_inter : int, optional
        Maximum interaction order to include (default: 4)
    **kwargs
        Additional arguments passed to specific methods
    
    Returns
    -------
    Dict[str, Array]
        Dictionary mapping contrast names to contrast matrices
    
    Examples
    --------
    >>> # F-contrasts for a single categorical event
    >>> event = EventFactor('condition', onsets, ['A', 'B', 'C'])
    >>> fcon = Fcontrasts(event)
    >>> fcon.keys()
    dict_keys(['condition'])
    >>> 
    >>> # F-contrasts for interaction term
    >>> term = EventTerm([cond_event, group_event])
    >>> fcon = Fcontrasts(term)
    >>> fcon.keys()
    dict_keys(['condition', 'group', 'condition:group'])
    >>> 
    >>> # F-contrasts for full model
    >>> model = event_model("condition * group", data=df)
    >>> fcon = Fcontrasts(model)
    """
    raise NotImplementedError(f"Fcontrasts not implemented for {type(x)}")


def _contrast_sum_matrix(n: int) -> Array:
    """Create sum-to-zero contrast matrix.
    
    Parameters
    ----------
    n : int
        Number of levels
    
    Returns
    -------
    Array
        Contrast matrix of shape (n, n-1)
    """
    if n < 2:
        raise ValueError("Need at least 2 levels for contrasts")
    
    # Create sum contrast matrix
    con = np.eye(n)[:, :-1]  # Drop last column
    # Last row is negative sum of others
    con[-1, :] = -1
    
    return con


def _unit_vector(n: int) -> Array:
    """Create unit vector (column of ones).
    
    Parameters
    ----------
    n : int
        Length of vector
    
    Returns
    -------
    Array
        Column vector of ones
    """
    return np.ones((n, 1))


@Fcontrasts.register(EventVariable)
def _fcontrasts_event_variable(event: EventVariable, **kwargs: object) -> Dict[str, Array]:
    """Continuous events have no F-contrasts.
    
    Parameters
    ----------
    event : EventVariable
        Continuous event
    
    Returns
    -------
    Dict[str, Array]
        Empty dictionary (no F-contrasts for continuous)
    """
    return {}


@Fcontrasts.register(EventFactor)
def _fcontrasts_event_factor(event: EventFactor, **kwargs: object) -> Dict[str, Array]:
    """Generate F-contrasts for categorical event.
    
    Parameters
    ----------
    event : EventFactor
        Categorical event
    
    Returns
    -------
    Dict[str, Array]
        Single contrast matrix for the factor
    """
    if event.event_type != 'categorical':
        return {}
    
    n_levels = len(event.levels)
    if n_levels < 2:
        return {}
    
    # Generate contrast matrix
    contrast_matrix = _contrast_sum_matrix(n_levels)
    
    # Add row names (level names)
    return {event.name: contrast_matrix}


@Fcontrasts.register(EventTerm)
def _fcontrasts_event_term(
    term: EventTerm, max_inter: int = 4, **kwargs: object
) -> Dict[str, Array]:
    """Generate F-contrasts for event term.
    
    Generates contrast matrices for main effects and interactions
    of categorical variables in the term.
    
    Parameters
    ----------
    term : EventTerm
        Event term containing one or more events
    max_inter : int, optional
        Maximum interaction order (default: 4)
    
    Returns
    -------
    Dict[str, Array]
        Contrast matrices for main effects and interactions
    """
    # Filter for categorical events only
    cat_events = [ev for ev in term.events if ev.event_type == 'categorical']
    
    if not cat_events:
        return {}
    
    # Create C (unit) and D (contrast) matrices for each categorical event
    C_matrices = {}
    D_matrices = {}
    
    for ev in cat_events:
        n_levels = len(ev.levels)
        C_matrices[ev.name] = _unit_vector(n_levels)
        D_matrices[ev.name] = _contrast_sum_matrix(n_levels)
    
    # Get expected row names in Kronecker order
    level_lists = [ev.levels for ev in cat_events]
    level_grid = list(itertools.product(*level_lists))
    cat_cond_names = [':'.join(combo) for combo in level_grid]
    expected_rows = len(level_grid)
    
    # Compute main effects matrices
    main_effects = {}
    for i, ev in enumerate(cat_events):
        # Start with C matrices for all events
        mat_list = [C_matrices[e.name] for e in cat_events]
        # Replace the i-th with D matrix
        mat_list[i] = D_matrices[ev.name]
        
        # Kronecker product
        result = reduce(np.kron, mat_list)
        main_effects[ev.name] = result
    
    # Start with main effects
    all_contrasts = main_effects.copy()
    
    # Compute interaction effects if requested
    if len(cat_events) > 1:
        # Generate all combinations of 2 or more events, up to max_inter
        for k in range(2, min(len(cat_events) + 1, max_inter + 1)):
            for combo in itertools.combinations(range(len(cat_events)), k):
                # Start with C matrices
                mat_list = [C_matrices[e.name] for e in cat_events]
                # Replace selected indices with D matrices
                for idx in combo:
                    mat_list[idx] = D_matrices[cat_events[idx].name]
                
                # Kronecker product
                result = reduce(np.kron, mat_list)
                
                # Create interaction name
                inter_name = ':'.join(cat_events[idx].name for idx in combo)
                all_contrasts[inter_name] = result
    
    return all_contrasts


@Fcontrasts.register(EventModel)
def _fcontrasts_event_model(model: EventModel, **kwargs: object) -> Dict[str, Array]:
    """Generate F-contrasts for event model.
    
    Computes F-contrasts for each term and maps them to the
    full design matrix space.
    
    Parameters
    ----------
    model : EventModel
        Event model
    
    Returns
    -------
    Dict[str, Array]  
        F-contrast matrices mapped to full design matrix
    """
    from ..utils import term_indices
    
    # Get column indices for each term
    tind = term_indices(model)
    ncond = model.design_matrix.shape[1]
    
    all_fcontrasts = {}
    
    # Process each term
    for i, term in enumerate(model.terms):
        term_name = term.name
        term_indices_vec = tind.get(term_name, [])
        
        if not term_indices_vec:
            continue
        
        # Get event terms (if available)
        if hasattr(model, '_event_terms') and model._event_terms:
            event_term = model._event_terms[i]
            
            # Calculate F-contrasts for the term
            fcon_local = Fcontrasts(event_term)
            
            if fcon_local:
                # Map each local contrast to full design matrix space
                for con_name, con_matrix in fcon_local.items():
                    # Create full-size contrast matrix
                    out = np.zeros((ncond, con_matrix.shape[1]))
                    
                    # Get column names for this term
                    term_cols = [model.column_names[idx] for idx in term_indices_vec]
                    
                    # Check dimensions
                    if con_matrix.shape[0] != len(term_indices_vec):
                        # Skip if dimensions don't match
                        continue
                    
                    # Map contrast to full matrix
                    out[term_indices_vec, :] = con_matrix
                    
                    # Create full contrast name
                    full_name = f"{term_name}#{con_name}"
                    all_fcontrasts[full_name] = out
    
    return all_fcontrasts


@singledispatch
def plot_Fcontrasts(x,
                    figsize: Optional[tuple] = None,
                    cmap: str = 'RdBu_r') -> None:
    """Plot F-contrast matrices as heatmaps.

    This is a generic function that dispatches based on the type of x.

    Parameters
    ----------
    x : Dict[str, Array] or EventModel
        F-contrast matrices from Fcontrasts() or an EventModel
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap name (default: 'RdBu_r')

    Examples
    --------
    >>> # From dict
    >>> fcon = Fcontrasts(model)
    >>> plot_Fcontrasts(fcon)
    >>>
    >>> # From EventModel directly
    >>> model = event_model("condition * group", data=df)
    >>> plot_Fcontrasts(model)
    """
    raise NotImplementedError(f"plot_Fcontrasts not implemented for {type(x)}")


@plot_Fcontrasts.register(dict)
def _plot_fcontrasts_dict(fcontrasts: Dict[str, Array],
                         figsize: Optional[tuple] = None,
                         cmap: str = 'RdBu_r') -> None:
    """Plot F-contrast matrices from dictionary.

    Parameters
    ----------
    fcontrasts : Dict[str, Array]
        F-contrast matrices from Fcontrasts()
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap name (default: 'RdBu_r')
    """
    import matplotlib.pyplot as plt

    n_contrasts = len(fcontrasts)
    if n_contrasts == 0:
        raise ValueError("No F-contrasts to plot")

    # Determine subplot layout
    ncols = min(3, n_contrasts)
    nrows = (n_contrasts + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Hide unused subplots
    for i in range(n_contrasts, len(axes)):
        axes[i].set_visible(False)

    # Plot each contrast
    for i, (name, matrix) in enumerate(fcontrasts.items()):
        ax = axes[i]

        # Determine color scale
        vmax = np.abs(matrix).max()
        if vmax == 0:
            vmax = 1

        # Plot heatmap
        im = ax.imshow(matrix.T, aspect='auto', cmap=cmap,
                       vmin=-vmax, vmax=vmax)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Labels
        ax.set_title(name)
        ax.set_xlabel('Design Matrix Columns')
        ax.set_ylabel('Contrast Columns')

        # Add grid
        ax.set_xticks(np.arange(matrix.shape[0]) - 0.5, minor=True)
        ax.set_yticks(np.arange(matrix.shape[1]) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if plt.isinteractive():
        plt.show()


def _plot_fcontrasts_event_model(model: EventModel,
                                figsize: Optional[tuple] = None,
                                cmap: str = 'RdBu_r') -> None:
    """Plot F-contrasts for an EventModel.

    Parameters
    ----------
    model : EventModel
        Event model to compute and plot F-contrasts for
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap name (default: 'RdBu_r')
    """
    # Get F-contrasts from the model
    fcon = Fcontrasts(model)

    if not fcon:
        raise ValueError("No F-contrasts could be generated from this model")

    # Delegate to dict plotting function
    return _plot_fcontrasts_dict(fcon, figsize=figsize, cmap=cmap)


# Register the EventModel implementation
# We do this outside the function to avoid circular imports
def _register_event_model_fcontrasts():
    """Register EventModel implementation after imports are resolved."""
    try:
        from ..design.event_model import EventModel
        plot_Fcontrasts.register(EventModel)(_plot_fcontrasts_event_model)
    except ImportError:
        pass


# Call registration at module load time
_register_event_model_fcontrasts()
