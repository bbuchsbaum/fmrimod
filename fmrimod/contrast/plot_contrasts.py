"""Visualization functions for contrasts."""

from typing import Dict, Optional, List, Tuple, Union, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import singledispatch

from ..types import Array

if TYPE_CHECKING:
    from ..design.event_model import EventModel


@singledispatch
def plot_contrasts(x, 
                  absolute_limits: bool = False,
                  rotate_x_text: bool = True,
                  scale_mode: str = "auto",
                  figsize: Optional[Tuple[float, float]] = None,
                  cmap: Optional[str] = None,
                  **kwargs):
    """Plot contrast weights as a heatmap.
    
    This is a generic function that dispatches based on the type of x.
    
    Parameters
    ----------
    x : object
        Object containing contrast information (EventModel, dict, etc.)
    absolute_limits : bool, optional
        If True, fix color scale at (-1, 1). If False, use data range.
    rotate_x_text : bool, optional
        If True, rotate x-axis labels for readability.
    scale_mode : {'auto', 'diverging', 'one_sided'}, optional
        Color scale mode. 'auto' chooses based on data.
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap name. If None, chosen based on scale_mode.
    **kwargs
        Additional arguments passed to plotting functions
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    raise NotImplementedError(f"plot_contrasts not implemented for {type(x)}")


@plot_contrasts.register(dict)
def _plot_contrasts_dict(contrast_dict: Dict[str, Array],
                        absolute_limits: bool = False,
                        rotate_x_text: bool = True,
                        scale_mode: str = "auto",
                        figsize: Optional[Tuple[float, float]] = None,
                        cmap: Optional[str] = None,
                        **kwargs):
    """Plot contrasts from a dictionary of contrast weights.
    
    Parameters
    ----------
    contrast_dict : dict
        Dictionary mapping contrast names to weight matrices.
        Each value should be a 2D array where rows are design matrix
        columns and columns are contrast components.
    absolute_limits : bool, optional
        If True, fix color scale at (-1, 1). If False, use data range.
    rotate_x_text : bool, optional
        If True, rotate x-axis labels for readability.
    scale_mode : {'auto', 'diverging', 'one_sided'}, optional
        Color scale mode. 'auto' chooses based on data.
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap name. If None, chosen based on scale_mode.
    **kwargs
        Additional arguments (e.g., edgecolor for tiles)
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if not contrast_dict:
        raise ValueError("No contrasts provided")
    
    # Build the big matrix of all contrasts
    contrast_rows = []
    row_labels = []
    
    # Determine number of regressors from first contrast
    first_key = next(iter(contrast_dict))
    n_regressors = contrast_dict[first_key].shape[0]
    
    for name, weights in contrast_dict.items():
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        
        if weights.shape[0] != n_regressors:
            raise ValueError(f"Contrast '{name}' has {weights.shape[0]} rows, "
                           f"expected {n_regressors}")
        
        # Add each column of the contrast matrix
        for col_idx in range(weights.shape[1]):
            contrast_rows.append(weights[:, col_idx])
            if weights.shape[1] > 1:
                row_labels.append(f"{name}_component{col_idx + 1}")
            else:
                row_labels.append(name)
    
    # Stack into matrix
    contrast_matrix = np.vstack(contrast_rows)
    
    # Determine scale mode
    if scale_mode == "auto":
        has_negative = np.any(contrast_matrix < 0)
        scale_mode = "diverging" if has_negative else "one_sided"
    
    # Determine color limits
    vmin = np.min(contrast_matrix)
    vmax = np.max(contrast_matrix)
    
    if absolute_limits:
        if scale_mode == "diverging":
            vmin, vmax = -1, 1
        else:
            vmin, vmax = 0, 1
    
    # Choose colormap
    if cmap is None:
        if scale_mode == "diverging":
            cmap = "RdBu_r"
        else:
            cmap = "Reds"
    
    # Create figure
    if figsize is None:
        # Adjust size based on number of contrasts and regressors
        width = max(8, n_regressors * 0.3)
        height = max(6, len(row_labels) * 0.3)
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(contrast_matrix, 
                   aspect='auto',
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_regressors))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    
    # Try to get regressor names from the contrast dict attributes
    regressor_names = None
    for weights in contrast_dict.values():
        if hasattr(weights, 'regressor_names'):
            regressor_names = weights.regressor_names
            break
    
    if regressor_names is None:
        regressor_names = [f"Reg{i+1}" for i in range(n_regressors)]
    
    ax.set_xticklabels(regressor_names)
    
    # Rotate x labels if requested
    if rotate_x_text:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Contrast Weight')
    
    # Add grid
    ax.set_xticks(np.arange(n_regressors + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels
    ax.set_xlabel('Design Matrix Columns')
    ax.set_ylabel('Contrasts')
    ax.set_title('Contrast Weights')
    
    # Apply any additional kwargs to the plot
    if 'edgecolor' in kwargs:
        for spine in ax.spines.values():
            spine.set_edgecolor(kwargs['edgecolor'])
    
    plt.tight_layout()
    
    return fig


def plot_contrasts_event_model(model: 'EventModel',
                             absolute_limits: bool = False,
                             rotate_x_text: bool = True,
                             scale_mode: str = "auto",
                             figsize: Optional[Tuple[float, float]] = None,
                             cmap: Optional[str] = None,
                             **kwargs):
    """Plot contrasts for an EventModel.
    
    Parameters
    ----------
    model : EventModel
        Event model with defined contrasts
    absolute_limits : bool, optional
        If True, fix color scale at (-1, 1). If False, use data range.
    rotate_x_text : bool, optional
        If True, rotate x-axis labels for readability.
    scale_mode : {'auto', 'diverging', 'one_sided'}, optional
        Color scale mode. 'auto' chooses based on data.
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap name. If None, chosen based on scale_mode.
    **kwargs
        Additional arguments
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    from ..contrast import contrast_weights
    
    # Get contrast weights from the model
    cws = contrast_weights(model)
    
    if not cws:
        raise ValueError("No contrasts defined in this event model")
    
    # Flatten the nested structure into a single dict
    flat_contrasts = {}
    regressor_names = model.design_matrix.columns if hasattr(model.design_matrix, 'columns') else None
    
    for term_name, term_contrasts in cws.items():
        if term_contrasts is None:
            continue
            
        for contrast_name, contrast_obj in term_contrasts.items():
            if contrast_obj is None:
                continue
                
            # Get the offset weights (full design matrix space)
            weights = getattr(contrast_obj, 'offset_weights', 
                            getattr(contrast_obj, 'weights', None))
            
            if weights is None:
                continue
            
            # Store with full name
            full_name = f"{term_name}.{contrast_name}"
            flat_contrasts[full_name] = weights
            
            # Try to attach regressor names if available
            if regressor_names is not None and not hasattr(weights, 'regressor_names'):
                weights.regressor_names = regressor_names
    
    # Use the dict plotting function
    return _plot_contrasts_dict(flat_contrasts,
                              absolute_limits=absolute_limits,
                              rotate_x_text=rotate_x_text,
                              scale_mode=scale_mode,
                              figsize=figsize,
                              cmap=cmap,
                              **kwargs)


# Register the EventModel implementation
# We do this outside the function to avoid circular imports
def _register_event_model():
    """Register EventModel implementation after imports are resolved."""
    try:
        from ..design.event_model import EventModel
        plot_contrasts.register(EventModel)(plot_contrasts_event_model)
    except ImportError:
        pass


# Call registration at module load time
_register_event_model()