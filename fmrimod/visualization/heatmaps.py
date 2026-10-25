"""Matrix views of a design: ``design_map`` and ``correlation_map``.

Ports R fmridesign ``R/plot_heatmaps.R``. Both draw one cell per (row,
design column) with columns in design order and grouped by term under a
spanner (a bold label over a muted rule), and each carries a diagnostic:
column magnitudes under the design map, variance inflation on the
correlation map's diagonal. The numbers live in
:mod:`fmrimod.visualization.design_stats`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._design_common import (
    DIVERGING_MID,
    Group,
    boundaries,
    cbar_axes,
    colourbar,
    design_columns,
    draw_spanners,
    fd_cmap,
    finish_figure,
    fmt_tr,
    groups_of,
    is_baseline_model,
    model_design_matrix,
    model_sampling_frame,
    name_or_count,
    plural,
    range_label,
    resolve,
    timeline,
)
from ._helpers import DOT, mmss, new_figure, themed
from .style import BASE_SIZE, INK, WARN

# --------------------------------------------------------------------------
# Shared layout helpers
# --------------------------------------------------------------------------


def display_columns(model: Any) -> pd.DataFrame:
    """Column metadata in display order (R ``.fd_hm_columns``).

    Baseline columns are regrouped by term and then run, so each term's
    block-diagonal staircase reads as one unit; event columns keep design
    order and come first.
    """
    ci = design_columns(model)
    ev = ci[ci["kind"] == "event"]
    bl = ci[ci["kind"] == "baseline"]
    if len(bl):
        rank = {t: i for i, t in enumerate(dict.fromkeys(bl["term"]))}
        key = pd.DataFrame({
            "t": bl["term"].map(rank),
            "r": [0 if (r is None or (isinstance(r, float) and np.isnan(r))) else int(r) for r in bl["run"]],
            "c": bl["col"],
        }, index=bl.index)
        bl = bl.loc[key.sort_values(["t", "r", "c"]).index]
    return pd.concat([ev, bl]).reset_index(drop=True)


def _x_label_rotation(labels: list[str], panel_in: float, rotate: bool) -> tuple[float, str]:
    """Horizontal when labels fit their cells, else 40 degrees (<= 16
    columns) or vertical (R ``.fd_hm_x_text``)."""
    n = len(labels)
    if not rotate or n == 0:
        return 0.0, "center"
    widest = max(max(len(s) for s in lab.split("\n")) for lab in labels) * 0.075
    if widest <= panel_in / n * 0.9:
        return 0.0, "center"
    if n <= 16:
        return 40.0, "right"
    return 90.0, "center"


def _thin(n: int, max_labels: int) -> list[int]:
    if n <= max_labels:
        return list(range(n))
    k = int(np.ceil(n / max_labels))
    return list(range(0, n, k))


def _run_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(int(v))


# --------------------------------------------------------------------------
# design_map
# --------------------------------------------------------------------------


def design_map(
    model: Any,
    block_separators: bool = True,
    rotate_x_text: bool = True,
    fill_midpoint: float | None = None,
    fill_limits: tuple[float, float] | None = None,
    scale: str = "column",
    palette: str = "diverging",
    y_axis: str = "scan",
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    max_labels: int = 60,
    figsize: tuple[float, float] | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Design matrix as an image, in the style of SPM (R ``design_map``).

    One row per scan (time runs downward), one column per regressor.
    Columns are grouped by model term under a spanner, runs are separated
    by rules and labelled on the left, and the right axis marks the first
    scan of each run. By default each column is divided by its maximum
    absolute value (``scale="column"``) so regressors in different units
    share one diverging fill with zero at the neutral midpoint. Because
    that would make a near-flat regressor look healthy, a strip of bars
    under the matrix shows each column's peak relative to the largest in
    its term; columns below 5% of it are marked in the warning colour,
    outlined, and named in the subtitle. With 12 or fewer columns each
    label also gives the column's raw range.

    Baseline columns are grouped by term (drift, intercept, nuisance) and
    then run, with a "Run k" sub-spanner over each run's columns;
    nuisance columns are veiled so the block structure dominates.

    Parameters
    ----------
    model : EventModel, BaselineModel or FmriModel
    block_separators : bool
        Draw a rule at each run boundary.
    rotate_x_text : bool
        Angle column labels when they would overlap.
    fill_midpoint, fill_limits : optional
        Centre and limits of the fill scale (default symmetric about 0).
    scale : {"column", "none"}
        Divide each column by its max abs value, or plot raw values.
    palette : {"diverging", "grey"}
        The signed palette, or white-to-ink as in SPM.
    y_axis : {"scan", "time"}
        Label the right axis by scan number or onset time (m:ss).
    title, subtitle, caption : str, optional
        ``None`` uses the default; ``""`` removes it.
    max_labels : int
        Thin column labels to every k-th beyond this many.
    figsize : (float, float), optional
    ax : matplotlib Axes, optional
        Draw into this axes (the colourbar is placed under it).
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if scale not in ("column", "none"):
        raise ValueError(f"scale must be 'column' or 'none', got {scale!r}")
    if palette not in ("diverging", "grey"):
        raise ValueError(f"palette must be 'diverging' or 'grey', got {palette!r}")
    if y_axis not in ("scan", "time"):
        raise ValueError(f"y_axis must be 'scan' or 'time', got {y_axis!r}")

    X_all = model_design_matrix(model)
    ci = display_columns(model)
    X = X_all[:, ci["col"].to_numpy(dtype=int)]
    n, p = X.shape
    tl = timeline(model_sampling_frame(model), n)
    runs = tl.runs
    is_base = ci["kind"].to_numpy() == "baseline"

    mx = np.nanmax(np.abs(X), axis=0) if n else np.zeros(p)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    vals = X / np.where(mx < np.finfo(float).eps, 1.0, mx) if scale == "column" else X.copy()
    term_max = ci.groupby("term", sort=False)["term"].transform(lambda s: mx[s.index].max()).to_numpy()
    rel = np.where(term_max > 0, mx / np.where(term_max > 0, term_max, 1.0), 0.0)
    flat = (mx < np.finfo(float).eps) | (mx < 0.05 * term_max)
    quiet = is_base & (ci["term"].to_numpy() == "nuisance")

    groups = groups_of(ci["term"], ci["term_label"])
    bounds = boundaries(groups)
    sub_part = [
        _run_value(r) if b else (lab if (pd.notna(bt) and bt and int(bt) > 1) else "")
        for b, r, lab, bt in zip(is_base, ci["run"], ci["label"], ci["basis_total"], strict=False)
    ]
    sub_groups = groups_of([f"{t}|{s}" for t, s in zip(ci["term"], sub_part, strict=False)])
    sub_rules = [b for b in boundaries(sub_groups) if b not in bounds]

    col_labels = list(ci["full_label"])
    for i in np.flatnonzero(is_base):
        col_labels[i] = ci["label"].iloc[i]
        if ci["term"].iloc[i] == "block" and _run_value(ci["run"].iloc[i]):
            col_labels[i] = f"Run {_run_value(ci['run'].iloc[i])}"
    if p <= 12:
        lo = np.nanmin(X, axis=0)
        hi = np.nanmax(X, axis=0)
        col_labels = [f"{lab}\n{range_label(a, b)}" for lab, a, b in zip(col_labels, lo, hi, strict=False)]
    x_show = _thin(p, max_labels)

    gap = max(1.0, 0.012 * n)
    strip_h = max(2.0, 0.035 * n)
    s_top = n + 0.5 + gap
    s_bot = s_top + strip_h
    inset = 0.12 if p <= 60 else 0.0

    run_centres = ((runs["first_scan"] + runs["last_scan"]) / 2).to_numpy()
    y_ticks = list(dict.fromkeys([*runs["first_scan"].astype(int).tolist(), n]))
    if len(runs) == 1:
        from matplotlib.ticker import MaxNLocator

        inner = [v for v in MaxNLocator(5, integer=True).tick_values(1, n) if 0.08 * n < v < 0.92 * n]
        y_ticks = sorted({1, *[int(v) for v in inner], n})
    if y_axis == "time" and tl.unit == "s":
        onset = np.concatenate([[0.0], np.cumsum(np.repeat(runs["TR"].to_numpy(), runs["scans"].to_numpy()))])
        y_lab = [mmss(onset[t - 1]) for t in y_ticks]
        y_name = "Time (min:s)"
    else:
        y_lab = [str(t) for t in y_ticks]
        y_name = "Scan"

    fill_name = "Each column / its max abs value" if scale == "column" else "Value"
    from matplotlib.colors import Normalize

    if palette == "grey":
        lim = fill_limits or (float(np.nanmin(vals)), float(np.nanmax(vals)))
        cm, norm = fd_cmap("grey"), Normalize(*lim, clip=True)
        ticks = None
    elif fill_limits is None and fill_midpoint is None:
        lim_v = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
        lim_v = lim_v if lim_v > 0 else 1.0
        cm, norm = fd_cmap("diverging"), Normalize(-lim_v, lim_v, clip=True)
        ticks = [-1, -0.5, 0, 0.5, 1] if (scale == "column" and lim_v == 1.0) else None
    else:
        mid = 0.0 if fill_midpoint is None else float(fill_midpoint)
        half = float(np.nanmax(np.abs(vals - mid))) or 1.0
        lim = fill_limits or (mid - half, mid + half)
        cm, norm = fd_cmap("diverging"), Normalize(*lim, clip=True)
        ticks = None

    is_bl_model = is_baseline_model(model)
    title = resolve(title, "Baseline design matrix" if is_bl_model else "Design matrix")
    if subtitle is None:
        tr_txt = f" (TR {fmt_tr(runs['TR'])})" if tl.unit == "s" else ""
        subtitle = f" {DOT} ".join([plural(p, "regressor"), plural(len(runs), "run"), plural(n, "scan") + tr_txt])
        if flat.any():
            subtitle += f" {DOT} near-flat: " + name_or_count(list(ci["full_label"][flat]), "column")
    subtitle = resolve(subtitle, None)
    caption = resolve(caption, "Bars under the matrix: each column's peak abs value relative to the largest "
                               "in its term; orange marks columns below 5%.")

    own = ax is None
    if own:
        w = figsize[0] if figsize else float(np.clip(2.4 + 0.5 * p, 5.4, 11.5))
        h = figsize[1] if figsize else 6.6
        fig, ax = new_figure(figsize=(w, h))
    else:
        fig = ax.figure
    with themed():
        imkw = {"cmap": cm, "norm": norm, "interpolation": "nearest", "aspect": "auto"}
        imkw.update(kwargs)
        im = ax.imshow(vals, extent=(-0.5, p - 0.5, n + 0.5, 0.5), **imkw)
        ax.grid(False)
        from matplotlib.patches import Rectangle

        for qg in groups_of(quiet):
            if qg.label == "True":
                ax.add_patch(Rectangle((qg.start - 0.5, 0.5), qg.end - qg.start + 1, n, facecolor=DIVERGING_MID,
                                       alpha=0.45, edgecolor="none", zorder=2))
        for b in sub_rules:
            ax.plot([b, b], [0.5, n + 0.5], color=INK["rule"], linewidth=0.6, zorder=3)
        for b in bounds:
            ax.plot([b, b], [0.5, s_bot], color="white", linewidth=2.6, zorder=3.1, solid_capstyle="butt")
            ax.plot([b, b], [0.5, s_bot], color=INK["muted"], linewidth=0.8, zorder=3.2, solid_capstyle="butt")
        if block_separators and len(runs) > 1:
            for fs in runs["first_scan"].iloc[1:]:
                ax.plot([-0.5, p - 0.5], [fs - 0.5, fs - 0.5], color=INK["muted"], linewidth=0.8, zorder=3.3)
        for j in range(p):
            hgt = max(rel[j], 0.0) * strip_h
            ax.add_patch(Rectangle((j - 0.5 + inset, s_bot - hgt), 1 - 2 * inset, hgt, facecolor=INK["zero"],
                                   edgecolor="none", zorder=2))
            if flat[j]:
                ax.add_patch(Rectangle((j - 0.5, s_top), 1, strip_h, facecolor=WARN, edgecolor="none", zorder=2.1))
                ax.add_patch(Rectangle((j - 0.5, 0.5), 1, n, facecolor="none", edgecolor=WARN,
                                       linewidth=1.4 if p <= 40 else 0.9, zorder=4))
        ax.set_xlim(-0.5, p - 0.5)
        ax.set_ylim(s_bot, 0.5)

        # Spanners: terms over a rule; a run tier beneath for baseline columns.
        run_groups: list[Group] = []
        if is_base.any() and len(runs) > 1:
            for g in sub_groups:
                run_no = sub_part[g.start]
                if is_base[g.start] and run_no and g.end > g.start:
                    run_groups.append(Group(g.start, g.end, f"Run {run_no}", run_no))
        if run_groups:
            draw_spanners(ax, groups, text_pt=19, rule_pt=16)
            draw_spanners(ax, run_groups, text_pt=3, rule_pt=None, size=BASE_SIZE * 0.78, weight="normal",
                          colour=INK["text2"], overflow_pt=0)
        else:
            draw_spanners(ax, groups, text_pt=6, rule_pt=3)

        ax.set_yticks(run_centres)
        ax.set_yticklabels([f"Run {r}" for r in runs["run"]], rotation=90, va="center", color=INK["text"],
                           fontsize=BASE_SIZE * 0.85)
        ax.tick_params(axis="both", length=0)
        sec = ax.secondary_yaxis("right")
        sec.set_yticks(y_ticks)
        sec.set_yticklabels(y_lab)
        sec.tick_params(length=0, labelcolor=INK["muted"], labelsize=BASE_SIZE * 0.82, pad=3)
        sec.set_ylabel(y_name, color=INK["muted"], fontsize=BASE_SIZE * 0.78)
        for sp in sec.spines.values():
            sp.set_visible(False)

        panel_in = (fig.get_figwidth() - 1.3) if own else 5.0
        rot, ha = _x_label_rotation([col_labels[i] for i in x_show], panel_in, rotate_x_text)
        ax.set_xticks(x_show)
        ax.set_xticklabels([col_labels[i] for i in x_show], rotation=rot, ha=ha,
                           rotation_mode="anchor" if rot and rot < 90 else "default",
                           fontsize=BASE_SIZE * (0.78 if p > 24 else 0.82), color=INK["text2"],
                           linespacing=1.15)
        cax = cbar_axes(ax, "bottom", width_in=1.5, gap_pt=8)
        colourbar(fig, cax, im, fill_name, ticks=ticks)
        finish_figure(fig, title, subtitle, caption, own=own, ax=ax)
    return fig


def plot_model_summary(model: Any, figsize: tuple[float, float] | None = None) -> dict[str, Any]:
    """The design map, correlation map and regressor plot of a model.

    Returns a dict with ``"design_map"``, ``"correlation_map"`` and
    ``"time_series"`` figures.
    """
    from .design_map import correlation_map
    from .event_plot import plot_event_model

    return {
        "design_map": design_map(model, figsize=figsize),
        "correlation_map": correlation_map(model),
        "time_series": plot_event_model(model),
    }


__all__ = ["design_map", "display_columns", "plot_model_summary"]
