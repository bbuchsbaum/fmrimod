"""Contrast weight heatmap (R fmridesign ``plot_contrasts``).

Every contrast is a row of weights over the design columns, in design
order and grouped by term, so it lines up column for column with
:func:`fmrimod.visualization.design_map`. Two margin columns check each
row: the sum of the weights (non-zero sums in the warning colour) and the
standard-error multiplier ``sqrt(c'(X'X)^+ c)`` with one intercept per run
(:func:`fmrimod.visualization.design_stats.contrast_se`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import singledispatch
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..types import Array, ModelProtocol

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass
class _Row:
    key: str  # contrast identity (term#name)
    name: str
    k: int  # component (1-based)
    K: int  # number of components
    f: bool  # F-contrast (several components)
    w: np.ndarray
    term: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@singledispatch
def plot_contrasts(x: object,
                   absolute_limits: bool = False,
                   rotate_x_text: bool = True,
                   scale_mode: str = "auto",
                   figsize: tuple[float, float] | None = None) -> Figure:
    """Plot contrast weights as a heatmap.

    Dispatches on ``x``: an ``EventModel`` (its contrasts, labelled from its
    columns, with standard errors from its design) or a dict mapping
    contrast names to weight vectors or ``p x k`` matrices (F-contrasts).

    Parameters
    ----------
    x : EventModel or dict
    absolute_limits : bool
        Fix the fill at [-1, 1] (or [0, 1] one-sided) instead of the
        largest absolute weight.
    rotate_x_text : bool
        Angle column labels when they would overlap.
    scale_mode : {"auto", "diverging", "one_sided"}
        ``"auto"`` and ``"diverging"`` are symmetric about 0;
        ``"one_sided"`` is a neutral-to-warm scale for non-negative weights.
    figsize : (float, float), optional

    The implementations also take ``annotate``, ``title``, ``subtitle``,
    ``caption`` and ``ax``; the dict one also ``labels`` (column labels)
    and ``design`` (a model or design matrix, enabling the SE column).

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError(f"plot_contrasts not implemented for {type(x)}")


def _dict_rows(contrast_dict: dict[str, object], f_always: bool = False) -> tuple[list[_Row], int]:
    if not contrast_dict:
        raise ValueError("No contrasts provided")
    first = np.asarray(next(iter(contrast_dict.values())), dtype=float)
    p = first.shape[0]
    rows: list[_Row] = []
    for name, weights in contrast_dict.items():
        W = np.asarray(weights, dtype=float)
        if W.ndim == 1:
            W = W[:, None]
        if W.shape[0] != p:
            raise ValueError(f"Contrast '{name}' has {W.shape[0]} rows, expected {p}")
        K = W.shape[1]
        for k in range(K):
            rows.append(_Row(str(name), str(name), k + 1, K, f_always or K > 1, W[:, k].copy()))
    return rows, p


def _design_and_runs(design: object, p: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if design is None:
        return None, None
    from ..visualization._design_common import (
        model_design_matrix,
        model_sampling_frame,
        timeline,
    )

    if isinstance(design, np.ndarray) or hasattr(design, "to_numpy") and not hasattr(design, "design_matrix"):
        X = np.asarray(design, dtype=float)
        run = np.ones(X.shape[0], dtype=int)
    else:
        model = design
        if getattr(model, "event_model", None) is not None:
            model = model.event_model
        X = model_design_matrix(model)
        run = timeline(model_sampling_frame(model), X.shape[0]).run
    if X.shape[1] != p:
        raise ValueError(f"design has {X.shape[1]} columns, the contrasts {p}")
    return X, run


@plot_contrasts.register(dict)
def _plot_contrasts_dict(contrast_dict: dict[str, Array],
                         absolute_limits: bool = False,
                         rotate_x_text: bool = True,
                         scale_mode: str = "auto",
                         figsize: tuple[float, float] | None = None,
                         labels: Sequence[str] | None = None,
                         design: Any = None,
                         **kwargs: object) -> Figure:
    """Plot contrasts from a dict of weight vectors or ``p x k`` matrices.

    ``labels`` names the design columns (default ``Reg1``, ...); a
    ``design`` (model or matrix) adds the SE column.
    """
    rows, p = _dict_rows(contrast_dict)
    col_labels = list(labels) if labels is not None else [f"Reg{i + 1}" for i in range(p)]
    if len(col_labels) != p:
        raise ValueError(f"labels has {len(col_labels)} entries, expected {p}")
    X, run = _design_and_runs(design, p)
    return render_contrasts(rows, col_labels, None, X, run, absolute_limits=absolute_limits,
                            rotate_x_text=rotate_x_text, scale_mode=scale_mode, figsize=figsize, **kwargs)


def _event_model_rows(model: object) -> list[_Row]:
    from . import contrast_weights

    cws = contrast_weights(model)
    rows: list[_Row] = []
    names = list(getattr(model, "column_names", []))
    p = len(names)
    for term_name, term_contrasts in cast(Any, cws or {}).items():
        for cname, cobj in cast(Any, term_contrasts or {}).items():
            if cobj is None:
                continue
            W = getattr(cobj, "offset_weights", None)
            if W is None:
                W = getattr(cobj, "weights", None)
            if W is None:
                continue
            row_names = list(getattr(W, "index", [])) if hasattr(W, "index") else []
            W = np.asarray(W, dtype=float)
            if W.ndim == 1:
                W = W[:, None]
            if W.shape[0] != p and row_names and all(r in names for r in row_names):
                full = np.zeros((p, W.shape[1]))
                full[[names.index(r) for r in row_names]] = W
                W = full
            if W.shape[0] != p:
                continue
            K = W.shape[1]
            is_f = K > 1 or "F" in type(cobj).__name__[:2]
            label = str(getattr(cobj, "name", None) or cname)
            for k in range(K):
                rows.append(_Row(f"{term_name}#{cname}", label, k + 1, K, is_f, W[:, k].copy(), term=str(term_name)))
    return rows


def plot_contrasts_event_model(model: ModelProtocol,
                               absolute_limits: bool = False,
                               rotate_x_text: bool = True,
                               scale_mode: str = "auto",
                               figsize: tuple[float, float] | None = None,
                               **kwargs: object) -> Any:
    """Plot the contrasts defined on an ``EventModel``'s terms.

    (Returns a matplotlib Figure; annotated ``Any`` so the hint resolves
    without importing matplotlib.)

    Columns are labelled and grouped as in ``design_map``; the SE column
    uses the model's design with one intercept per run.
    """
    from ..visualization._design_common import (
        design_columns,
        model_design_matrix,
        model_sampling_frame,
        timeline,
    )

    rows = _event_model_rows(model)
    if not rows:
        raise ValueError("No contrasts defined in this event model")
    ci = design_columns(model)
    ci = ci[ci["kind"] == "event"].reset_index(drop=True)
    X = model_design_matrix(model)
    run = timeline(model_sampling_frame(model), X.shape[0]).run
    return render_contrasts(rows, list(ci["full_label"]), ci, X, run, absolute_limits=absolute_limits,
                            rotate_x_text=rotate_x_text, scale_mode=scale_mode, figsize=figsize, **kwargs)


def render_contrasts(
    rows: list[_Row],
    col_labels: list[str],
    ci: pd.DataFrame | None,
    X: np.ndarray | None,
    run: np.ndarray | None,
    *,
    absolute_limits: bool = False,
    rotate_x_text: bool = True,
    scale_mode: str = "auto",
    figsize: tuple[float, float] | None = None,
    annotate: bool | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    ax: Axes | None = None,
    default_title: str = "Contrast weights",
) -> Figure:
    """Draw contrast rows as the R heatmap; shared by plot_Fcontrasts."""
    from matplotlib.colors import Normalize
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    from ..visualization._design_common import (
        PM,
        SIGMA,
        boundaries,
        cbar_axes,
        colourbar,
        draw_spanners,
        fd_cmap,
        fd_num,
        finish_figure,
        finish_to_panel,
        fmt_weight,
        groups_of,
        header_width,
        plural,
        resolve,
        text_colour,
    )
    from ..visualization._helpers import DOT, TIMES, new_figure, themed
    from ..visualization.design_stats import contrast_se
    from ..visualization.heatmaps import _thin, _x_label_rotation
    from ..visualization.style import BASE_SIZE, INK, WARN

    if scale_mode not in ("auto", "diverging", "one_sided"):
        raise ValueError(f"scale_mode must be 'auto', 'diverging' or 'one_sided', got {scale_mode!r}")
    p = len(col_labels)
    n_rows = len(rows)
    keys = [r.key for r in rows]
    short = [r.name for r in rows]
    first = [i == 0 or keys[i] != keys[i - 1] for i in range(n_rows)]
    uniq_first = [short[i] for i in range(n_rows) if first[i]]
    clash = {s for s in uniq_first if uniq_first.count(s) > 1}
    short = [f"{r.term}: {s}" if (s in clash and r.term) else s for r, s in zip(rows, short, strict=False)]
    is_f = np.array([r.f for r in rows])
    row_lab = [s if r.K == 1 else (f"{s}   1" if r.k == 1 else str(r.k)) for r, s in zip(rows, short, strict=False)]

    W = np.vstack([r.w for r in rows])
    W[np.abs(W) < 1e-12] = 0.0
    sums = W.sum(axis=1)
    empty = ~(W != 0).any(axis=1)
    balanced = (np.abs(sums) < 1e-8) & ~empty
    sum_lab = ["0" if abs(s) < 1e-8 else fd_num(s, 3) for s in sums]
    sum_lab = ["all 0" if e else s for s, e in zip(sum_lab, empty, strict=False)]
    sum_col = [INK["muted"] if b else WARN for b in balanced]
    se = contrast_se(X, run, W.T) if X is not None else None
    if se is not None:
        se_lab = ["" if e else ("not est." if not np.isfinite(v) else fd_num(v, 2)) for v, e in zip(se, empty, strict=False)]
        se_col = [WARN if not np.isfinite(v) else INK["text2"] for v in se]
    one_sided = scale_mode == "one_sided"
    wmax = float(np.abs(W).max()) or 1.0
    limit = 1.0 if absolute_limits else (max(float(W.max()), 0.0) if one_sided else wmax)
    limit = limit if limit > 0 else 1.0
    if annotate is None:
        annotate = p <= 40
    fs = BASE_SIZE * (0.86 if p <= 10 else 0.74 if p <= 20 else 0.6)

    groups = groups_of(ci["term"], ci["term_label"]) if ci is not None else []
    bounds = boundaries(groups)
    contrast_bounds = [i - 0.5 for i in range(1, n_rows) if keys[i] != keys[i - 1]]

    if one_sided:
        legend = "Weight (fixed 0 to 1)" if absolute_limits else "Weight (0 to max w)"
    else:
        legend = f"Weight (fixed {PM}1)" if absolute_limits else f"Weight ({PM}max abs w)"

    n_contrasts = len(dict.fromkeys(keys))
    if subtitle is None:
        subtitle = f"{plural(n_contrasts, 'contrast')} over {plural(p, 'regressor')}"
        f_keys = list(dict.fromkeys(k for k, f in zip(keys, is_f, strict=False) if f))
        if f_keys:
            subtitle += f" {DOT} {plural(len(f_keys), 'F-contrast')}"
        t_first = [i for i in range(n_rows) if first[i] and not is_f[i]]
        if t_first:
            subtitle += f" {DOT} {int(sum(balanced[i] for i in t_first))} of {len(t_first)} sum to zero"
        n_empty = len({keys[i] for i in range(n_rows) if empty[i]})
        if n_empty:
            subtitle += f" {DOT} {n_empty} {'has' if n_empty == 1 else 'have'} no non-zero weights"
        if se is not None:
            n_ne = len({keys[i] for i in range(n_rows) if not np.isfinite(se[i]) and not empty[i]})
            if n_ne:
                subtitle += f" {DOT} {n_ne} not estimable"
    subtitle = resolve(subtitle, None)
    se_head = f"SE {TIMES}{SIGMA}"
    if caption is None and se is not None:
        caption = (f"{se_head}: standard error of the contrast per unit noise SD, given this design "
                   "(with run intercepts). Lower is better.")
    caption = resolve(caption, None)
    title = resolve(title, default_title)

    cell_w = float(np.clip(5.2 / max(p, 1), 0.42, 1.5))
    row_h = 0.36
    panel_w, panel_h = p * cell_w, n_rows * row_h
    own = ax is None
    if own:
        fig, ax = new_figure(figsize=figsize or (panel_w + 4.0, panel_h + 2.6))
    else:
        fig = ax.figure
    with themed():
        cm = fd_cmap("magnitude" if one_sided else "diverging")
        norm = Normalize(0.0 if one_sided else -limit, limit, clip=True)
        mesh = ax.pcolormesh(np.arange(p + 1) - 0.5, np.arange(n_rows + 1) - 0.5, W, cmap=cm, norm=norm,
                             edgecolors="white", linewidth=1.4)
        ax.set_xlim(-0.5, p - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.grid(False)
        if annotate:
            for i in range(n_rows):
                for j in range(p):
                    w = W[i, j]
                    colour = INK["zero"] if w == 0 else text_colour(w / limit)
                    ax.text(j, i, fmt_weight(w), ha="center", va="center", fontsize=fs, color=colour, zorder=4)
        for b in bounds:
            ax.axvline(b, color="white", linewidth=4.0, zorder=3)
        if is_f.any():
            for b in contrast_bounds:
                ax.axhline(b, color=INK["muted"], linewidth=0.9, zorder=3.5)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_lab, color=INK["text"], fontsize=BASE_SIZE * 0.86)
        shown = _thin(p, 60)
        rot, ha = _x_label_rotation([col_labels[j] for j in shown], panel_w, rotate_x_text)
        ax.set_xticks(shown)
        ax.set_xticklabels([col_labels[j] for j in shown], rotation=rot, ha=ha, color=INK["muted"],
                           fontsize=BASE_SIZE * 0.82, rotation_mode="anchor" if rot and rot < 90 else "default")
        ax.tick_params(length=0)
        if groups:
            draw_spanners(ax, groups, text_pt=6, rule_pt=3)

        # Right margin: sum of weights, then the SE multiplier.
        char_pt = 6.2
        w_sum = char_pt * max(len(s) for s in [*sum_lab, "sum w"]) + 16
        base = blended_transform_factory(ax.transAxes, ax.transData)
        top = blended_transform_factory(ax.transAxes, ax.transAxes)
        cols = [(10.0, "sum w", sum_lab, sum_col)]
        if se is not None:
            cols.append((10.0 + w_sum, se_head, se_lab, se_col))
        for x_pt, head, labs, colours in cols:
            off = ScaledTranslation(x_pt / 72, 0, fig.dpi_scale_trans)
            ax.text(1.0, 1.0, head, transform=top + off + ScaledTranslation(0, 6 / 72, fig.dpi_scale_trans),
                    ha="left", va="bottom", fontsize=BASE_SIZE * 0.86, fontweight="bold", color=INK["text"],
                    clip_on=False)
            for i, (s, c) in enumerate(zip(labs, colours, strict=False)):
                ax.text(1.0, i, s, transform=base + off, ha="left", va="center", fontsize=BASE_SIZE * 0.82,
                        color=c, clip_on=False)
        cax = cbar_axes(ax, "bottom", width_in=1.5, gap_pt=8)
        colourbar(fig, cax, mesh, legend, ticks=[0, 0.5 * limit, limit] if one_sided else
                  [-limit, -0.5 * limit, 0, 0.5 * limit, limit])
        if own and figsize is None:
            finish_to_panel(fig, ax, title, subtitle, caption, width_in=panel_w, height_in=panel_h,
                            min_width=header_width(subtitle, title))
        else:
            finish_figure(fig, title, subtitle, caption, own=own, ax=ax)
    return fig


# Register the EventModel implementation after imports are resolved.
def _register_event_model() -> None:
    try:
        from ..design.event_model import EventModel

        cast(Any, plot_contrasts).register(EventModel)(plot_contrasts_event_model)
    except ImportError:  # pragma: no cover
        pass


_register_event_model()
