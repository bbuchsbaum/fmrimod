"""Matrix views of a design: ``design_map`` and ``correlation_map``.

Ports R fmridesign ``R/plot_heatmaps.R``. Both draw one cell per (row,
design column) with columns in design order and grouped by term under a
spanner (a bold label over a muted rule), and each carries a diagnostic:
column magnitudes under the design map, variance inflation on the
correlation map's diagonal. The numbers live in
:mod:`fmrimod.visualization.design_stats`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


import numpy as np
import pandas as pd

from ._design_common import (
    DIVERGING_MID,
    GE,
    PM,
    Group,
    boundaries,
    cbar_axes,
    colourbar,
    design_columns,
    draw_spanners,
    fd_cmap,
    fd_num,
    finish_figure,
    finish_to_panel,
    fmt_tr,
    groups_of,
    header_width,
    is_baseline_model,
    model_design_matrix,
    model_sampling_frame,
    name_or_count,
    plural,
    range_label,
    resolve,
    text_colour,
    timeline,
)
from ._helpers import DOT, MINUS, mmss, new_figure, themed
from .style import BASE_SIZE, INK, WARN

# --------------------------------------------------------------------------
# Shared layout helpers
# --------------------------------------------------------------------------


def display_columns(model: object) -> pd.DataFrame:
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


def _run_value(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(int(v))


# --------------------------------------------------------------------------
# design_map
# --------------------------------------------------------------------------


def design_map(
    model: object,
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
    ax: Axes | None = None,
    **kwargs: object,
) -> Figure:
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


# --------------------------------------------------------------------------
# correlation_map
# --------------------------------------------------------------------------


def _fmt_r(r: float, digits: int) -> str:
    """Correlation without the leading zero: ``.28``, ``−.3``; ``0`` for zero."""
    if not np.isfinite(r):
        return ""
    if round(abs(r), digits) >= 1:
        return (MINUS if r < 0 else "") + "1"
    s = f"{r:.{digits}f}"
    s = s.replace("0.", ".", 1) if s.lstrip("-").startswith("0.") else s
    if s.lstrip("-").strip(".0") == "":
        return "0"
    return s.replace("-", MINUS)


def correlation_map(
    model: object,
    rotate_x_text: bool = True,
    method: str = "pearson",
    half_matrix: bool = True,
    limits: str = "fixed",
    annotate: bool | None = None,
    flag_threshold: float = 0.5,
    vif_threshold: float = 5.0,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    within_run: bool | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    annot: bool | None = None,
) -> Figure:
    """Pairwise regressor correlations as a lower triangle (R ``correlation_map``).

    The fill is the diverging palette fixed at [-1, 1] with a neutral zero;
    up to 20 columns each cell also prints r (two decimals up to 10
    columns). Cells with ``|r| >= flag_threshold`` are outlined and counted
    in the subtitle, which also names the most correlated pair. Pairwise
    correlations miss multicollinearity, so the diagonal shows each
    column's variance inflation factor with per-run intercepts
    (:func:`~fmrimod.visualization.design_stats.design_vif`); VIFs at or
    above ``vif_threshold`` and aliased columns are flagged in the warning
    colour. Columns keep design order and are grouped by term, so the map
    lines up with :func:`design_map`.

    Parameters
    ----------
    model : EventModel, BaselineModel or FmriModel
    rotate_x_text : bool
        Angle column labels when they would overlap.
    method : {"pearson", "spearman"}
        Correlation method (VIFs are always linear).
    half_matrix : bool
        Lower triangle only (default) or every off-diagonal cell.
    limits : {"fixed", "data"}
        Fill over [-1, 1], or symmetric up to the largest ``|r|``.
    annotate : bool, optional
        Print r in the cells; default when there are at most 20 columns
        (12 for a baseline model). ``annot`` is an alias.
    flag_threshold, vif_threshold : float
        Outline cells at or above this ``|r|``; flag VIFs at or above this.
    within_run : bool, optional
        Correlate within runs with run intercepts removed (see
        :func:`~fmrimod.visualization.design_stats.design_correlations`).
        Defaults to ``True`` for a baseline model, ``False`` otherwise.
    title, subtitle, caption : str, optional
        ``None`` uses the default; ``""`` removes it.
    figsize : (float, float), optional
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    from .design_stats import correlation_matrix, vif, within_run_correlations

    if annot is not None:
        if annotate is not None:
            raise ValueError("Supply `annotate` or `annot`, not both.")
        annotate = annot
    if limits not in ("fixed", "data"):
        raise ValueError(f"limits must be 'fixed' or 'data', got {limits!r}")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    is_bl = is_baseline_model(model)
    if within_run is None:
        within_run = is_bl
    X_all = model_design_matrix(model)
    ci = display_columns(model)
    X = X_all[:, ci["col"].to_numpy(dtype=int)]
    n = X.shape[0]
    run = timeline(model_sampling_frame(model), n).run
    n_runs = len(np.unique(run))
    hidden = None
    if within_run:
        keep = (ci["term"] != "block").to_numpy()
        X, ci = X[:, keep], ci[keep].reset_index(drop=True)
        if X.shape[1] < 2:
            raise ValueError("Fewer than two columns remain to correlate after removing run intercepts.")
        R, hidden, _ = within_run_correlations(X, run, method)
    else:
        R = correlation_matrix(X, method)
    p = X.shape[1]
    labs = list(ci["full_label"])

    low = R.copy()
    low[np.triu_indices(p)] = np.nan
    max_r = float(np.nanmax(np.abs(low))) if np.isfinite(low).any() else np.nan
    limit = 1.0 if (limits == "fixed" or not np.isfinite(max_r) or max_r == 0) else max_r
    n_flag = int(np.nansum(np.abs(low) >= flag_threshold))
    vifs = vif(X, run) if p <= 150 else np.full(p, np.nan)
    aliased = np.isinf(vifs)
    high_vif = aliased | (np.isfinite(vifs) & (vifs >= vif_threshold))
    vif_lab = ["aliased" if a else ("" if not np.isfinite(v) else fd_num(v, 2)) for v, a in zip(vifs, aliased, strict=False)]
    if p <= 10:
        vif_lab = [f"VIF {s}" if np.isfinite(v) else s for s, v in zip(vif_lab, vifs, strict=False)]
    elif p > 20:
        vif_lab = [""] * p
    else:
        vif_lab = ["!" if a else s for s, a in zip(vif_lab, aliased, strict=False)]

    if subtitle is None:
        if p < 2:
            subtitle = "Only one regressor: nothing to correlate"
        elif hidden is not None and hidden[np.tril_indices(p, -1)].all() and np.isfinite(np.diag(R)).any():
            subtitle = "No two regressors share a run: nothing to correlate within runs"
        elif not np.isfinite(max_r):
            subtitle = "No regressor varies: correlations are undefined"
        else:
            i, j = np.argwhere(np.abs(low) == max_r)[0]
            subtitle = (f"Max abs r = {max_r:.2f}: {labs[j]} vs {labs[i]} {DOT} {n_flag} "
                        f"{'pair' if n_flag == 1 else 'pairs'} at abs r {GE} {flag_threshold:g}")
            if high_vif.any():
                subtitle += (f"\nVIF {GE} {vif_threshold:g} or aliased: "
                             + name_or_count([labs[k] for k in np.flatnonzero(high_vif)], "column"))
    subtitle = resolve(subtitle, None)
    if caption is None:
        what = "Spearman rank" if method == "spearman" else "Pearson"
        if within_run:
            body = (" r within runs (columns centred within each run, run intercepts removed; "
                    "run-specific columns correlated on their own run)")
        else:
            body = f" r across all {n:,} scans" + (" (runs concatenated)" if n_runs > 1 else "")
        caption = (f"{what}{body}.\nDiagonal: variance inflation factor of each column, with run means removed"
                   + (" (per-run intercepts)" if n_runs > 1 else "") + ".")
    caption = resolve(caption, None)
    title = resolve(title, "Baseline regressor correlations" if is_bl else "Regressor correlations")

    groups = groups_of(ci["term"], ci["term_label"])
    bounds = boundaries(groups)
    if annotate is None:
        annotate = p <= (12 if is_bl else 20)
    digits = 2 if p <= 10 else 1
    fs = BASE_SIZE * (0.92 if p <= 8 else 0.8 if p <= 14 else 0.66)

    cell = 0.6 if p * 0.6 <= 5 else 5.0 / p
    # A two- or three-column matrix would otherwise be smaller than its header.
    cell = max(cell, 2.0 / max(p, 1))
    panel = p * cell
    own = ax is None
    if own:
        lab_w = min(max(len(s) for s in labs) * 0.075, 2.6) if labs else 0.5
        rot_guess = max(len(s) for s in labs) * 0.075 > cell * 0.9
        bottom = (min(max(len(s) for s in labs) * 0.055, 1.9) if rot_guess else 0.25) + 0.25
        w = figsize[0] if figsize else max(panel + lab_w + 0.6, 5.4)
        h = figsize[1] if figsize else panel + bottom + 1.45 + (0.4 if p <= 10 else 0.0) + 0.45
        fig, ax = new_figure(figsize=(w, h))
    else:
        fig = ax.figure

    mask = np.tril(np.ones((p, p), dtype=bool), -1) if half_matrix else ~np.eye(p, dtype=bool)
    if hidden is not None:
        mask &= ~hidden
    Z = np.ma.masked_where(~mask | ~np.isfinite(R), R)
    with themed():
        cm = fd_cmap("diverging").with_extremes(bad=(0, 0, 0, 0))
        norm = Normalize(-limit, limit, clip=True)
        # Undefined (NaN) pairs that are drawn get the neutral band colour.
        undefined = mask & ~np.isfinite(R)
        mesh = ax.pcolormesh(np.arange(p + 1) - 0.5, np.arange(p + 1) - 0.5, Z, cmap=cm, norm=norm,
                             edgecolors="white", linewidth=0.8 if p <= 30 else 0.15)
        for i, j in np.argwhere(undefined):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=INK["band"], edgecolor="white",
                                   linewidth=0.8))
        # The diagonal holds each column's VIF; a neutral cell gives the label a home.
        for k in range(p):
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, facecolor=INK["band"], edgecolor="white",
                                   linewidth=0.8 if p <= 30 else 0.15, zorder=1))
        ax.set_xlim(-0.5, p - 0.5)
        ax.set_ylim(p - 0.5, -0.5)
        ax.grid(False)
        for b in bounds:
            if half_matrix:
                if b + 1 < p - 0.5:
                    ax.plot([b, b], [b + 1, p - 0.5], color=INK["muted"], linewidth=0.9, zorder=3)
                ax.plot([-0.5, b], [b, b], color=INK["muted"], linewidth=0.9, zorder=3)
            else:
                ax.plot([b, b], [-0.5, p - 0.5], color=INK["muted"], linewidth=0.9, zorder=3)
                ax.plot([-0.5, p - 0.5], [b, b], color=INK["muted"], linewidth=0.9, zorder=3)
        inset = 0.06
        lw = 1.6 if p <= 30 else 0.8
        for i, j in np.argwhere(mask & np.isfinite(R) & (np.abs(np.nan_to_num(R)) >= flag_threshold)):
            ax.add_patch(Rectangle((j - 0.5 + inset, i - 0.5 + inset), 1 - 2 * inset, 1 - 2 * inset,
                                   facecolor="none", edgecolor=INK["text"], linewidth=lw, zorder=4))
        for k in np.flatnonzero(high_vif):
            ax.add_patch(Rectangle((k - 0.5 + inset, k - 0.5 + inset), 1 - 2 * inset, 1 - 2 * inset,
                                   facecolor="none", edgecolor=WARN, linewidth=lw, zorder=4))
        if annotate:
            for i, j in np.argwhere(mask & np.isfinite(R)):
                v = R[i, j]
                ax.text(j, i, _fmt_r(v, digits), ha="center", va="center", fontsize=fs,
                        color=text_colour(v / limit), zorder=5)
        for k in range(p):
            if vif_lab[k]:
                ax.text(k, k, vif_lab[k], ha="center", va="center", fontsize=fs * 0.8,
                        color=WARN if high_vif[k] else INK["muted"], zorder=5)
        shown = _thin(p, 60)
        ax.set_yticks(shown)
        ax.set_yticklabels([labs[k] for k in shown], color=INK["text2"], fontsize=BASE_SIZE * 0.82)
        rot, ha = _x_label_rotation([labs[k] for k in shown], min(4.5, panel), rotate_x_text)
        ax.set_xticks(shown)
        ax.set_xticklabels([labs[k] for k in shown], rotation=rot, ha=ha, color=INK["muted"],
                           fontsize=BASE_SIZE * 0.82,
                           rotation_mode="anchor" if rot and rot < 90 else "default")
        ax.tick_params(length=0)
        # A single term spans every column and says nothing the axis labels do not.
        if sum(1 for g in groups if g.label) > 1:
            draw_spanners(ax, groups, text_pt=6, rule_pt=3)
        legend = "Spearman r" if method == "spearman" else "r"
        if limit != 1.0:
            legend = f"r ({PM}max abs r)"
        ticks = [-1, -0.5, 0, 0.5, 1] if limit == 1.0 else None
        if p <= 10 or not half_matrix:
            cax = cbar_axes(ax, "bottom", width_in=min(1.5, panel), gap_pt=8)
            colourbar(fig, cax, mesh, legend, ticks=ticks)
        else:
            cax = ax.inset_axes([0.6, 0.9, 0.38, 0.022])
            colourbar(fig, cax, mesh, legend, ticks=ticks, label_side="top")
        if own and figsize is None:
            finish_to_panel(fig, ax, title, subtitle, caption, width_in=panel, height_in=panel,
                            min_width=header_width(subtitle, title))
        else:
            if not own:
                ax.set_aspect("equal")
            finish_figure(fig, title, subtitle, caption, own=own, ax=ax)
    return fig


def plot_model_summary(model: object, figsize: tuple[float, float] | None = None) -> dict[str, Figure]:
    """The design map, correlation map and regressor plot of a model.

    Returns a dict with ``"design_map"``, ``"correlation_map"`` and
    ``"time_series"`` figures.
    """
    from .event_plot import plot_event_model

    return {
        "design_map": design_map(model, figsize=figsize),
        "correlation_map": correlation_map(model),
        "time_series": plot_event_model(model),
    }


__all__ = ["correlation_map", "design_map", "display_columns", "plot_model_summary"]
