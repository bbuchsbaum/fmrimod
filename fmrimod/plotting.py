"""Plotting utilities for HRFs and regressors."""

from __future__ import annotations

from typing import Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd

from .hrf.core import HRF


def plot_hrf(
    hrf: HRF,
    time: Optional[ArrayLike] = None,
    normalize: bool = False,
    show_peak: bool = True,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs,
) -> "matplotlib.axes.Axes":
    """Plot a single HRF.

    For multi-basis HRFs each basis function is plotted as a separate line.

    Args:
        hrf: HRF object to plot.
        time: Time points (seconds).  Defaults to ``np.arange(0, span, 0.1)``.
        normalize: Normalize each basis to unit peak before plotting.
        show_peak: Annotate peak time and amplitude on single-basis plots.
        ax: Existing matplotlib Axes.  Created if ``None``.
        **kwargs: Forwarded to ``ax.plot()``.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if time is None:
        time = np.arange(0, hrf.span + 0.1, 0.1)
    time = np.asarray(time, dtype=np.float64)

    vals = hrf(time)
    if vals.ndim == 1:
        vals = vals[:, np.newaxis]

    if normalize:
        for i in range(vals.shape[1]):
            mx = np.max(np.abs(vals[:, i]))
            if mx > 0:
                vals[:, i] /= mx

    if ax is None:
        _, ax = plt.subplots()

    for i in range(vals.shape[1]):
        label = f"{hrf.name}" if vals.shape[1] == 1 else f"{hrf.name} [{i}]"
        ax.plot(time, vals[:, i], label=label, **kwargs)

    if show_peak and vals.shape[1] == 1:
        peak_idx = int(np.argmax(vals[:, 0]))
        peak_t = time[peak_idx]
        peak_v = vals[peak_idx, 0]
        ax.annotate(
            f"peak={peak_v:.2f} @ {peak_t:.1f}s",
            xy=(peak_t, peak_v),
            xytext=(peak_t + hrf.span * 0.1, peak_v * 0.9),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=8,
            color="grey",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(hrf.name)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    if vals.shape[1] > 1:
        ax.legend(fontsize=8)
    return ax


def plot_hrfs(
    *hrfs: HRF,
    time: Optional[ArrayLike] = None,
    normalize: bool = False,
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs,
) -> "matplotlib.axes.Axes":
    """Overlay multiple HRFs on the same axes for comparison.

    Only single-basis HRFs (or the first basis of multi-basis HRFs) are
    plotted to keep the comparison readable.

    Args:
        *hrfs: HRF objects to compare.
        time: Time points (seconds).  Defaults to the max span across HRFs.
        normalize: Normalize each HRF to unit peak.
        labels: Per-HRF labels for the legend.
        title: Plot title.
        ax: Existing matplotlib Axes.
        **kwargs: Forwarded to ``ax.plot()``.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if len(hrfs) == 0:
        raise ValueError("At least one HRF must be provided.")

    max_span = max(h.span for h in hrfs)
    if time is None:
        time = np.arange(0, max_span + 0.1, 0.1)
    time = np.asarray(time, dtype=np.float64)

    if labels is None:
        labels = [h.name for h in hrfs]

    if ax is None:
        _, ax = plt.subplots()

    for hrf, label in zip(hrfs, labels):
        vals = hrf(time)
        if vals.ndim == 2:
            vals = vals[:, 0]
        if normalize:
            mx = np.max(np.abs(vals))
            if mx > 0:
                vals = vals / mx
        ax.plot(time, vals, label=label, **kwargs)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "HRF comparison")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)
    return ax


def plot_regressor(
    reg,
    grid: Optional[ArrayLike] = None,
    precision: float = 0.33,
    show_onsets: bool = True,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs,
) -> "matplotlib.axes.Axes":
    """Plot an evaluated regressor time course.

    Args:
        reg: :class:`~fmrimod.regressor.core.Regressor` object.
        grid: Evaluation time grid.  Defaults to ``np.arange(0, max_onset + span, precision)``.
        precision: Temporal precision for evaluation.
        show_onsets: Draw vertical lines at event onsets.
        ax: Existing matplotlib Axes.
        **kwargs: Forwarded to ``ax.plot()``.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if grid is None:
        if len(reg.onsets) > 0:
            end = np.max(reg.onsets) + reg.span + 5
        else:
            end = reg.span + 5
        grid = np.arange(0, end, precision)
    grid = np.asarray(grid, dtype=np.float64)

    vals = reg.evaluate(grid, precision=precision)
    if vals.ndim == 1:
        vals = vals[:, np.newaxis]

    if ax is None:
        _, ax = plt.subplots()

    hrf_name = reg.hrf[0].name if reg.hrf_is_list else reg.hrf.name
    for i in range(vals.shape[1]):
        label = hrf_name if vals.shape[1] == 1 else f"{hrf_name} [{i}]"
        ax.plot(grid, vals[:, i], label=label, **kwargs)

    if show_onsets and len(reg.onsets) > 0:
        for onset in reg.onsets:
            ax.axvline(onset, color="red", alpha=0.3, linewidth=0.8, linestyle=":")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted response")
    ax.set_title(f"Regressor ({hrf_name})")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    if vals.shape[1] > 1:
        ax.legend(fontsize=8)
    return ax


def plot_regressors(
    *regs,
    grid: Optional[ArrayLike] = None,
    precision: float = 0.33,
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    **kwargs,
) -> "matplotlib.axes.Axes":
    """Overlay multiple regressors on the same axes.

    Only the first basis column of each regressor is plotted.

    Args:
        *regs: Regressor objects.
        grid: Evaluation time grid.
        precision: Temporal precision.
        labels: Per-regressor legend labels.
        title: Plot title.
        ax: Existing matplotlib Axes.
        **kwargs: Forwarded to ``ax.plot()``.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if len(regs) == 0:
        raise ValueError("At least one Regressor must be provided.")

    if grid is None:
        all_onsets = np.concatenate([r.onsets for r in regs if len(r.onsets) > 0])
        max_span = max(r.span for r in regs)
        end = (np.max(all_onsets) + max_span + 5) if len(all_onsets) > 0 else max_span + 5
        grid = np.arange(0, end, precision)
    grid = np.asarray(grid, dtype=np.float64)

    if labels is None:
        labels = []
        for r in regs:
            name = r.hrf[0].name if r.hrf_is_list else r.hrf.name
            labels.append(name)

    if ax is None:
        _, ax = plt.subplots()

    for reg, label in zip(regs, labels):
        vals = reg.evaluate(grid, precision=precision)
        if vals.ndim == 2:
            vals = vals[:, 0]
        ax.plot(grid, vals, label=label, **kwargs)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted response")
    ax.set_title(title or "Regressor comparison")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)
    return ax
