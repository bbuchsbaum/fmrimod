"""F-contrast matrices for testing main effects and interactions."""

from __future__ import annotations

import itertools
import warnings
from functools import reduce, singledispatch
from typing import Any, Dict, Optional, cast

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
    
    n_levels = len(cast(Any, event).levels)
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

    # Rows follow the term's cells: the first factor varies slowest, which is
    # what reducing the factor list with np.kron produces.
    n_levels = [len(cast(Any, ev).levels) for ev in cat_events]
    names = [cast(Any, ev).name for ev in cat_events]
    C_matrices = [_unit_vector(n) for n in n_levels]
    # A single-level factor is constant: it enters every contrast through its
    # 1 x 1 margin and has no effect of its own (R fmridesign f63135b).
    testable = [i for i, n in enumerate(n_levels) if n >= 2]
    if not testable:
        return {}

    def _effect(ix: tuple[int, ...]) -> Array:
        mat_list = list(C_matrices)
        for idx in ix:
            mat_list[idx] = _contrast_sum_matrix(n_levels[idx])
        return cast(Array, reduce(np.kron, mat_list))

    all_contrasts = {names[i]: _effect((i,)) for i in testable}
    for k in range(2, min(len(testable), max_inter) + 1):
        for combo in itertools.combinations(testable, k):
            all_contrasts[':'.join(names[idx] for idx in combo)] = _effect(combo)

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
        term_indices_vec = tind.get(term_name or "", [])
        
        if not term_indices_vec:
            continue
        
        # Get event terms (if available)
        if hasattr(model, '_event_terms') and model._event_terms:
            event_term = model._event_terms[i]
            
            # Calculate F-contrasts for the term
            fcon_local = Fcontrasts(event_term)
            
            if fcon_local:
                nb = _term_nbasis(model, term_indices_vec)
                # Map each local contrast to full design matrix space
                for con_name, con_matrix in fcon_local.items():
                    # A multi-basis HRF has nb condition-major columns per
                    # cell; test the effect jointly in every basis function
                    # (R fmridesign 4415504: kronecker(C, diag(nb))).
                    if nb > 1:
                        con_matrix = np.kron(con_matrix, np.eye(nb))

                    if con_matrix.shape[0] != len(term_indices_vec):
                        warnings.warn(
                            f"F-contrast '{con_name}' for term '{term_name}' has "
                            f"{con_matrix.shape[0]} rows, but the term has "
                            f"{len(term_indices_vec)} design columns. Skipping.",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue

                    out = np.zeros((ncond, con_matrix.shape[1]))
                    out[term_indices_vec, :] = con_matrix
                    all_fcontrasts[f"{term_name}#{con_name}"] = out

    return all_fcontrasts


def _term_nbasis(model: EventModel, term_indices_vec: list[int]) -> int:
    """Basis functions per cell of a term, read from its column facts."""
    facts = model.column_facts
    totals = {facts[idx].get("basis_total") for idx in term_indices_vec}
    if len(totals) != 1:
        return 1
    total = totals.pop()
    return int(total) if total else 1


@singledispatch
def plot_Fcontrasts(x: Any,
                    figsize: Optional[tuple[float, float]] = None,
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
                         figsize: Optional[tuple[float, float]] = None,
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
                                figsize: Optional[tuple[float, float]] = None,
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
def _register_event_model_fcontrasts() -> None:
    """Register EventModel implementation after imports are resolved."""
    try:
        from ..design.event_model import EventModel
        cast(Any, plot_Fcontrasts).register(EventModel)(_plot_fcontrasts_event_model)
    except ImportError:
        pass


# Call registration at module load time
_register_event_model_fcontrasts()
