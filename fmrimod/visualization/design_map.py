"""Design matrix visualization functions.

This module provides functions for visualizing design matrices as heatmaps,
correlation matrices, and time series plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ..design.event_model import EventModel


def design_map(
    model,
    block_separators: bool = True,
    rotate_x_text: bool = True,
    fill_midpoint: Optional[float] = None,
    fill_limits: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    **kwargs: object
) -> tuple[Figure, Axes]:
    """Visualize design matrix as a heatmap.

    Creates a heatmap visualization of the design matrix showing the
    intensity of each regressor at each time point.

    Parameters
    ----------
    model : EventModel or BaselineModel
        Event model or baseline model containing the design matrix to visualize
    block_separators : bool, default=True
        Whether to draw separators between blocks/runs if available
    rotate_x_text : bool, default=True
        Whether to rotate x-axis labels for better readability
    fill_midpoint : float, optional
        Midpoint for diverging color scale. If None, uses continuous scale
    fill_limits : tuple of float, optional
        (min, max) limits for color scale. If None, uses data range
    figsize : tuple of float, optional
        Figure size as (width, height). If None, automatically determined
    cmap : str, optional
        Colormap name. If None, uses 'RdBu_r' for diverging or 'viridis' for continuous
    **kwargs
        Additional arguments passed to matplotlib's imshow
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> # Basic usage
    >>> fig, ax = design_map(model)
    >>> 
    >>> # With custom colormap and size
    >>> fig, ax = design_map(model, figsize=(12, 8), cmap='coolwarm')
    >>> 
    >>> # Focus on specific value range
    >>> fig, ax = design_map(model, fill_limits=(-2, 2))
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )

    # Handle BaselineModel
    from ..baseline.baseline_model import BaselineModel
    if isinstance(model, BaselineModel):
        # Create a simple wrapper
        class _BaselineWrapper:
            def __init__(self, bm):
                self.design_matrix = bm.design_matrix
                dm = bm.design_matrix
                # Get column names from terms
                cols = []
                for tn in ['drift', 'block', 'nuisance']:
                    if tn in bm.terms and bm.terms[tn] is not None:
                        mat = bm.terms[tn].design_matrix
                        if isinstance(mat, pd.DataFrame):
                            cols.extend(list(mat.columns))
                        else:
                            cols.extend([f"{tn}_{i}" for i in range(mat.shape[1] if mat.ndim > 1 else 1)])
                self.column_names = cols if cols else [f"col_{i}" for i in range(dm.shape[1])]
                self.name = "BaselineModel"
                self.sampling_points = np.arange(dm.shape[0])
                # Copy block_ids if available
                if hasattr(bm.sampling_frame, 'blockids'):
                    self.block_ids = bm.sampling_frame.blockids
        model = _BaselineWrapper(model)

    # Get design matrix
    X = model.design_matrix
    n_scans, n_regressors = X.shape
    
    # Determine figure size if not provided
    if figsize is None:
        # Scale based on matrix dimensions
        width = max(8, min(20, n_regressors * 0.3))
        height = max(6, min(15, n_scans * 0.05))
        figsize = (width, height)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color scale
    if fill_limits is None:
        vmin, vmax = X.min(), X.max()
    else:
        vmin, vmax = fill_limits
    
    # Choose colormap
    if cmap is None:
        if fill_midpoint is not None:
            cmap = 'RdBu_r'  # Red-Blue reversed (blue=negative, red=positive)
        else:
            cmap = 'viridis'
    
    # Plot heatmap
    if fill_midpoint is not None:
        # Diverging colormap centered at midpoint
        # Normalize to make midpoint at center
        norm_vmin = vmin - fill_midpoint
        norm_vmax = vmax - fill_midpoint
        norm_max = max(abs(norm_vmin), abs(norm_vmax))
        im = ax.imshow(
            X,
            cmap=cmap,
            aspect='auto',
            vmin=fill_midpoint - norm_max,
            vmax=fill_midpoint + norm_max,
            **kwargs
        )
    else:
        # Continuous colormap
        im = ax.imshow(
            X,
            cmap=cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
    
    # Add block separators if requested and block IDs are available
    if block_separators and hasattr(model, 'block_ids'):
        block_ids = model.block_ids
        # Find block boundaries
        block_changes = np.where(np.diff(block_ids) != 0)[0] + 1
        
        # Draw horizontal lines at block boundaries
        for idx in block_changes:
            ax.axhline(y=idx - 0.5, color='white', linewidth=2, linestyle='-')
    
    # Set ticks and labels
    ax.set_xticks(range(n_regressors))
    ax.set_xticklabels(model.column_names, rotation=45 if rotate_x_text else 0, ha='right' if rotate_x_text else 'center')
    
    # Set y-axis to show scan numbers
    # Show subset of y-ticks if too many scans
    if n_scans > 50:
        y_tick_spacing = max(10, n_scans // 10)
        y_ticks = range(0, n_scans, y_tick_spacing)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(i+1) for i in y_ticks])
    else:
        ax.set_yticks(range(0, n_scans, max(1, n_scans // 20)))
        ax.set_yticklabels([str(i+1) for i in ax.get_yticks()])
    
    # Labels
    ax.set_xlabel('Regressors', fontsize=12)
    ax.set_ylabel('Scan Number', fontsize=12)
    ax.set_title(f'Design Matrix: {model.name}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax


def correlation_map(
    model,
    rotate_x_text: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = 'RdBu_r',
    annot: bool = False,
    fmt: str = '.2f',
    **kwargs: object
) -> tuple[Figure, Axes]:
    """Visualize correlation matrix of design matrix regressors.

    Creates a heatmap showing the correlation between all pairs of
    regressors in the design matrix.

    Parameters
    ----------
    model : EventModel or BaselineModel
        Event model or baseline model containing the design matrix
    rotate_x_text : bool, default=True
        Whether to rotate x-axis labels
    figsize : tuple of float, optional
        Figure size as (width, height). If None, automatically determined
    cmap : str, default='RdBu_r'
        Colormap name (diverging colormap recommended)
    annot : bool, default=False
        Whether to annotate cells with correlation values
    fmt : str, default='.2f'
        Format string for annotations
    **kwargs
        Additional arguments passed to seaborn's heatmap
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> # Basic correlation map
    >>> fig, ax = correlation_map(model)
    >>> 
    >>> # With correlation values shown
    >>> fig, ax = correlation_map(model, annot=True, figsize=(10, 8))
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )

    # Handle BaselineModel
    from ..baseline.baseline_model import BaselineModel
    if isinstance(model, BaselineModel):
        # Create a simple wrapper
        class _BaselineWrapper:
            def __init__(self, bm):
                self.design_matrix = bm.design_matrix
                dm = bm.design_matrix
                # Get column names from terms
                cols = []
                for tn in ['drift', 'block', 'nuisance']:
                    if tn in bm.terms and bm.terms[tn] is not None:
                        mat = bm.terms[tn].design_matrix
                        if isinstance(mat, pd.DataFrame):
                            cols.extend(list(mat.columns))
                        else:
                            cols.extend([f"{tn}_{i}" for i in range(mat.shape[1] if mat.ndim > 1 else 1)])
                self.column_names = cols if cols else [f"col_{i}" for i in range(dm.shape[1])]
                self.name = "BaselineModel"
        model = _BaselineWrapper(model)

    # Get design matrix and compute correlations.
    # Zero-variance columns yield undefined correlations; map those NaNs to 0
    # for stable plotting without runtime warning noise.
    X = model.design_matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Determine figure size
    n_regressors = X.shape[1]
    if figsize is None:
        size = max(8, min(15, n_regressors * 0.5))
        figsize = (size, size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle (optional)
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    if HAS_SEABORN:
        # Use seaborn if available
        sns.heatmap(
            corr_matrix,
            # mask=mask,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            annot=annot,
            fmt=fmt,
            xticklabels=model.column_names,
            yticklabels=model.column_names,
            ax=ax,
            **kwargs
        )
    else:
        # Fallback to matplotlib
        im = ax.imshow(
            corr_matrix,
            cmap=cmap,
            aspect='equal',
            vmin=-1,
            vmax=1,
            **kwargs
        )
        
        # Add grid lines
        for i in range(n_regressors + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5)
            ax.axvline(i - 0.5, color='white', linewidth=0.5)
        
        # Set ticks and labels
        ax.set_xticks(range(n_regressors))
        ax.set_yticks(range(n_regressors))
        ax.set_xticklabels(model.column_names)
        ax.set_yticklabels(model.column_names)
        
        # Add annotations if requested
        if annot:
            for i in range(n_regressors):
                for j in range(n_regressors):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:{fmt}}',
                                 ha='center', va='center',
                                 color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                                 fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', fontsize=10)
    
    # Rotate x labels if requested
    if rotate_x_text:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Set title
    ax.set_title(f'Regressor Correlation Matrix: {model.name}', fontsize=14, fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax


def plot_design_matrix(
    model: EventModel,
    term_name: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    separate_regressors: bool = True,
    **kwargs: object
) -> tuple[Figure, Axes]:
    """Plot time series of design matrix regressors.
    
    Creates line plots showing the time course of regressors in the
    design matrix.
    
    Parameters
    ----------
    model : EventModel
        Event model containing the design matrix
    term_name : str, optional
        Name of specific term to plot. If None, plots all terms
    figsize : tuple of float, optional
        Figure size as (width, height)
    separate_regressors : bool, default=True
        Whether to plot regressors in separate subplots when many exist
    **kwargs
        Additional arguments passed to plot functions
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : matplotlib.axes.Axes or array of Axes
        The axes object(s)
    
    Examples
    --------
    >>> # Plot all regressors
    >>> fig, axes = plot_design_matrix(model)
    >>> 
    >>> # Plot specific term only
    >>> fig, ax = plot_design_matrix(model, term_name='condition')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )
    
    # Get design matrix and time points
    X = model.design_matrix
    time_points = model.sampling_points
    
    # Select columns if term specified
    if term_name is not None:
        # Find columns matching term name
        import re
        pattern = rf"^{re.escape(term_name)}[_.\[]|^{re.escape(term_name)}$"
        matching_cols = [i for i, col in enumerate(model.column_names) 
                        if re.match(pattern, col)]
        
        if not matching_cols:
            raise ValueError(f"No columns found matching term name: {term_name}")
        
        X = X[:, matching_cols]
        col_names = [model.column_names[i] for i in matching_cols]
    else:
        col_names = model.column_names
    
    n_regressors = X.shape[1]
    
    # Determine layout
    if n_regressors > 6 and separate_regressors:
        # Use subplots
        n_cols = min(3, n_regressors)
        n_rows = (n_regressors + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (n_cols * 4, n_rows * 3)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Plot each regressor
        for i in range(n_regressors):
            ax = axes[i]
            ax.plot(time_points, X[:, i], linewidth=2, **kwargs)
            ax.set_title(col_names[i], fontsize=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_regressors, len(axes)):
            axes[i].set_visible(False)
        
    else:
        # Single plot with all regressors
        if figsize is None:
            figsize = (10, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all regressors
        for i in range(n_regressors):
            ax.plot(time_points, X[:, i], label=col_names[i], linewidth=2, **kwargs)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'Design Matrix Time Series: {model.name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legend
        if n_regressors <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes = ax
    
    plt.tight_layout()
    
    return fig, axes


# Convenience function to plot all visualizations
def plot_model_summary(
    model: EventModel,
    figsize: Optional[Tuple[float, float]] = None
) -> Dict[str, tuple[Figure, Axes]]:
    """Create summary visualization of event model.
    
    Generates three plots:
    1. Design matrix heatmap
    2. Correlation matrix
    3. Time series plot
    
    Parameters
    ----------
    model : EventModel
        Event model to visualize
    figsize : tuple of float, optional
        Base figure size (will be adapted for each plot)
    
    Returns
    -------
    dict
        Dictionary with keys 'design_map', 'correlation_map', 'time_series'
        containing (fig, ax) tuples for each plot
    
    Examples
    --------
    >>> plots = plot_model_summary(model)
    >>> # Access individual plots
    >>> design_fig, design_ax = plots['design_map']
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )
    
    results = {}
    
    # Design matrix heatmap
    dm_fig, dm_ax = design_map(model, figsize=figsize)
    results['design_map'] = (dm_fig, dm_ax)
    
    # Correlation matrix
    corr_fig, corr_ax = correlation_map(model)
    results['correlation_map'] = (corr_fig, corr_ax)
    
    # Time series
    ts_fig, ts_ax = plot_design_matrix(model)
    results['time_series'] = (ts_fig, ts_ax)

    return results


def plot_baseline_model(
    model,
    term_name: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: object
) -> tuple[Figure, Axes]:
    """Plot baseline model terms over time.

    Shows drift, block, and nuisance terms as line plots,
    with faceting by block/run when applicable.

    Parameters
    ----------
    model : BaselineModel
        Baseline model to visualize
    term_name : str, optional
        Specific term to plot ('drift', 'block', 'nuisance'). If None, plot all.
    title : str, optional
        Custom title
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments passed to plot functions

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of matplotlib.axes.Axes
        The axes objects

    Examples
    --------
    >>> # Plot all baseline terms
    >>> fig, axes = plot_baseline_model(baseline_model)
    >>>
    >>> # Plot only drift term
    >>> fig, axes = plot_baseline_model(baseline_model, term_name='drift')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for visualization")


    # Determine which terms to plot
    term_names = ['drift', 'block', 'nuisance'] if term_name is None else [term_name]
    active_terms = [(name, model.terms[name]) for name in term_names
                    if name in model.terms and model.terms[name] is not None]

    if not active_terms:
        raise ValueError("No terms to plot")

    n_terms = len(active_terms)

    if figsize is None:
        figsize = (10, 3 * n_terms)

    fig, axes = plt.subplots(n_terms, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (tname, term) in enumerate(active_terms):
        ax = axes[i]
        mat = term.design_matrix
        if isinstance(mat, pd.DataFrame):
            for col in mat.columns:
                ax.plot(mat[col].values, label=col, **kwargs)
        else:
            for j in range(mat.shape[1] if mat.ndim > 1 else 1):
                col = mat[:, j] if mat.ndim > 1 else mat
                ax.plot(col, label=f"{tname}_{j}", **kwargs)

        ax.set_title(tname.capitalize())
        ax.set_xlabel('Scan')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig, axes


def plot_sampling_frame(
    sframe,
    style: str = "timeline",
    show_ticks: bool = True,
    tick_every: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: object
) -> tuple[Figure, Axes]:
    """Plot sampling frame structure.

    Parameters
    ----------
    sframe : SamplingFrame or similar
        Sampling frame to visualize
    style : {'timeline', 'grid'}, default 'timeline'
        'timeline': horizontal bars per run
        'grid': scan tiles showing timing structure
    show_ticks : bool, default True
        Show tick marks
    tick_every : float, optional
        Tick spacing in seconds
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments passed to plot functions

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object

    Examples
    --------
    >>> # Timeline view
    >>> fig, ax = plot_sampling_frame(sframe, style='timeline')
    >>>
    >>> # Grid view
    >>> fig, ax = plot_sampling_frame(sframe, style='grid')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required")

    # Extract info from sampling frame
    if hasattr(sframe, 'blocklens'):
        blocklens = list(sframe.blocklens)
    elif hasattr(sframe, 'n_scans'):
        blocklens = [sframe.n_scans]
    else:
        raise ValueError("Cannot extract block info from sampling frame")

    if hasattr(sframe, "tr"):
        tr_source = sframe.tr
    elif hasattr(sframe, "TR"):
        tr_source = sframe.TR
    else:
        tr_source = 1.0

    n_blocks = len(blocklens)
    tr_arr = np.asarray(tr_source, dtype=float).reshape(-1)
    if len(tr_arr) == 1:
        tr_vals = np.repeat(float(tr_arr[0]), n_blocks)
    elif len(tr_arr) == n_blocks:
        tr_vals = tr_arr.astype(float)
    else:
        raise ValueError(
            f"Length of TR/tr ({len(tr_arr)}) must be 1 or match number of blocks ({n_blocks})"
        )

    if np.allclose(tr_vals, tr_vals[0]):
        tr_label = f"{float(tr_vals[0])}s"
    else:
        tr_label = f"{tr_vals.tolist()}s"

    if style == "timeline":
        if figsize is None:
            figsize = (12, max(2, n_blocks * 0.8))
        fig, ax = plt.subplots(figsize=figsize)

        current_time = 0
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_blocks, 3)))

        for i, blen in enumerate(blocklens):
            duration = float(blen) * float(tr_vals[i])
            ax.barh(i, duration, left=current_time, height=0.6,
                    color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
            ax.text(current_time + duration/2, i, f"Run {i+1}\n{blen} scans",
                    ha='center', va='center', fontsize=9)
            current_time += duration

        ax.set_yticks(range(n_blocks))
        ax.set_yticklabels([f"Run {i+1}" for i in range(n_blocks)])
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Sampling Frame (TR={tr_label}, {sum(blocklens)} total scans)')
        ax.invert_yaxis()

        if tick_every is not None:
            tick_every = float(tick_every)
            if tick_every <= 0:
                raise ValueError("tick_every must be positive")
            ax.set_xticks(np.arange(0.0, current_time + tick_every * 0.5, tick_every))

    elif style == "grid":
        if figsize is None:
            figsize = (12, max(2, n_blocks * 0.5))
        fig, ax = plt.subplots(figsize=figsize)

        max_len = max(blocklens)
        grid = np.zeros((n_blocks, max_len))
        for i, blen in enumerate(blocklens):
            grid[i, :blen] = 1

        ax.imshow(grid, aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_yticks(range(n_blocks))
        ax.set_yticklabels([f"Run {i+1}" for i in range(n_blocks)])
        ax.set_xlabel('Scan Index')
        ax.set_title(f'Scan Grid (TR={tr_label})')

        if tick_every is not None:
            tick_every = float(tick_every)
            if tick_every <= 0:
                raise ValueError("tick_every must be positive")
            # Grid x-axis is in scan indices; map seconds to scan steps.
            scan_step = max(1, int(round(tick_every / float(np.min(tr_vals)))))
            ax.set_xticks(np.arange(0, max_len + 1, scan_step))
    else:
        raise ValueError(f"style must be 'timeline' or 'grid', got '{style}'")

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return fig, ax
