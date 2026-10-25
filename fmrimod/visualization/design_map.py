"""Baseline-model and sampling-frame plots (former API, being ported)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

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



def plot_baseline_model(
    model: Any,
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
    sframe: Any,
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

        current_time: float = 0.0
        colors = cast(Any, plt.cm).Set2(np.linspace(0, 1, max(n_blocks, 3)))

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
