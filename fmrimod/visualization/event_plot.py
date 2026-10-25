"""Event-model plot: convolved regressors against time (R ``plot.event_model``).

Three layouts share one set of labels, colours and run furniture:

- ``"stacked"``: one row per condition (and basis function), area plus
  line, onset rug and duration bars in the row's colour, its own small
  right-hand amplitude axis;
- ``"overlay"``: every regressor on one amplitude axis;
- ``"heatmap"``: one raster row per regressor, for large (trialwise)
  designs, with each row's largest correlation in the right margin.

Runs are hairlines with "Run k" labels on top; no line crosses a run.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ._design_common import (
    Timeline,
    cbar_axes,
    colourbar,
    design_columns,
    fd_cmap,
    fd_num,
    fmt_tr,
    model_design_matrix,
    model_sampling_frame,
    plural,
    resolve,
    row_colours,
    signed_norm,
    timeline,
)
from ._helpers import (
    DOT,
    budget_breaks,
    finish,
    new_figure,
    row_label,
    run_axis,
    themed,
    time_axis,
)
from .style import BASE_SIZE, INK, WARN, palette

_STYLES = ("auto", "stacked", "overlay", "heatmap")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _row_keys(ci: pd.DataFrame) -> list[str]:
    """One display row per condition: basis functions share a row."""
    return [str(v) for v in ci["label"]]


def _abbreviate(labels: list[str], minlength: int) -> list[str]:
    """Shorten labels to about ``minlength`` characters (R ``abbreviate``):
    drop lower-case vowels from the right, then truncate; keep them unique."""
    out = []
    for s in labels:
        t = s
        while len(t) > minlength:
            m = re.search(r"(?<=\w)[aeiou](?=[^aeiou]*$)", t[1:])
            if not m:
                break
            i = m.start() + 1
            t = t[:i] + t[i + 1:]
        out.append(t[:minlength] if len(t) > minlength else t)
    seen: dict[str, int] = {}
    uniq = []
    for t in out:
        k = seen.get(t, 0)
        uniq.append(t if k == 0 else f"{t}_{k}")
        seen[t] = k + 1
    return uniq


def event_table(model: Any, ci: pd.DataFrame, row_key: list[str], tl: Timeline) -> pd.DataFrame | None:
    """Event onsets per display row, in plot time units.

    Columns: ``onset``, ``duration``, ``row``, ``event`` (one id per
    distinct event). ``None`` when the model carries no events or no
    sampling frame.
    """
    facts = getattr(model, "column_facts", None)
    events = getattr(model, "events", None)
    if facts is None or not isinstance(events, dict) or tl.unit != "s" or not len(ci):
        return None
    events = {**{str(getattr(v, "name", k)): v for k, v in events.items()}, **events}
    blockids = getattr(model, "blockids", None)
    cidx = getattr(model, "column_indices", {}) or {}
    rows: list[dict[str, Any]] = []
    run_ids = sorted(set(np.asarray(blockids).tolist())) if blockids is not None else [1]
    col_to_row = {int(c): row_key[i] for i, c in enumerate(ci["col"]) if ci["kind"].iloc[i] == "event"}
    for term, cols in cidx.items():
        cols = [int(c) for c in cols if int(c) in col_to_row]
        if not cols:
            continue
        names = [n for n in str(term).split(":") if n in events]
        if not names:
            continue
        ev0 = events[names[0]]
        onsets = np.asarray(getattr(ev0, "onsets", []), dtype=float)
        if onsets.size == 0:
            continue
        durs = np.broadcast_to(np.asarray(getattr(ev0, "durations", 0.0), dtype=float), onsets.shape)
        if blockids is not None and len(blockids) == len(onsets):
            run_ix = np.array([run_ids.index(b) for b in np.asarray(blockids).tolist()])
        else:
            run_ix = np.zeros(len(onsets), dtype=int)
        if run_ix.max() >= tl.n_runs:
            continue
        glob = onsets + tl.runs["start"].to_numpy()[run_ix]
        factors = [n for n in names if hasattr(events[n], "levels")]
        vals = {n: np.asarray(events[n].values).astype(str) for n in factors}
        single = vals[factors[0]] if len(factors) == 1 else None
        combo = (np.array(["_".join(f"{n}.{vals[n][i]}" for n in factors) for i in range(len(onsets))])
                 if factors else None)
        assigned: dict[str, np.ndarray] = {}
        for c in cols:
            f = facts[c]
            cands = {str(f.get("level")), str(f.get("condition"))}
            if not factors:
                hit = np.ones(len(onsets), dtype=bool)
            elif single is not None:
                hit = np.isin(single, list(cands))
            else:
                hit = np.isin(combo, list(cands))
            if not hit.any() and len(cidx[term]) == len(onsets):
                hit = np.zeros(len(onsets), dtype=bool)
                hit[list(cidx[term]).index(c)] = True
            r = col_to_row[c]
            assigned[r] = assigned[r] | hit if r in assigned else hit
        for r, hit in assigned.items():
            for i in np.flatnonzero(hit):
                rows.append({"onset": float(glob[i]),
                             "duration": float(durs[i]), "row": r,
                             "event": (int(run_ix[i]), round(float(onsets[i]), 6))})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    return out


def _peak_breaks(lo: float, hi: float, min_frac: float) -> list[float]:
    """Zero and round values near the row's extremes (R ``.fd_peak_breaks``),
    rounded towards zero; the trough is labelled only when it is far enough
    from zero (``min_frac`` of the span) not to collide with the 0 label."""

    def trunc2(v: float) -> float:
        if not np.isfinite(v) or v == 0:
            return 0.0
        d = 10.0 ** (np.floor(np.log10(abs(v))) - 1)
        return float(np.trunc(v / d) * d)

    r = hi - lo
    b = [0.0]
    if hi > 0.02 * r:
        b.append(trunc2(hi))
    if lo < -max(0.02, min_frac) * r:
        b.insert(0, trunc2(lo))
    return b


def _caption(ev: pd.DataFrame | None, y_scale: str, style: str) -> str:
    scale_txt = {
        "term": "Rows of the same term share a scale (right axis)",
        "row": "Each row has its own scale (right axis)",
        "shared": "All rows share one scale (right axis)",
    }[y_scale]
    if style == "overlay":
        return "Ticks mark event onsets." if ev is not None else ""
    if ev is None:
        return scale_txt + "."
    dur = bool((ev["duration"] > 0).any())
    return f"{scale_txt}. Ticks mark event onsets{'; bars show event durations' if dur else ''}."


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def plot_event_model(
    model: Any,
    term: str | None = None,
    style: str = "auto",
    show_events: bool = True,
    label_mode: str = "auto",
    max_labels: int = 30,
    block_x: str = "global",
    y_scale: str = "term",
    time_range: tuple[float, float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Any = None,
    abbrev_min: int = 10,
) -> Any:
    """Plot the convolved regressors of an event model against time.

    Parameters
    ----------
    model : EventModel
        The model to draw (an ``FmriModel`` or fit is reduced to its event
        model).
    term : str, optional
        Draw only this term (its name, e.g. ``"cond"`` or ``"cond:rt"``).
    style : {"auto", "stacked", "overlay", "heatmap"}
        ``"stacked"``: one row per condition (and basis function) with its
        own amplitude axis on the right; ``"overlay"``: every regressor on
        one axis; ``"heatmap"``: one raster row per regressor. ``"auto"``
        picks the heatmap above 12 rows or 3 basis functions, else stacked.
    show_events : bool
        Mark event onsets (rugs, or ticks in the heatmap) and durations.
    label_mode : {"auto", "compact", "none"}
        ``"compact"`` abbreviates row labels, ``"none"`` drops them and
        ``"auto"`` drops them above ``max_labels`` regressors.
    max_labels : int
        Label limit for ``label_mode="auto"``.
    block_x : {"global", "run"}
        ``"global"``: runs concatenated on one time axis; ``"run"``: one
        column of panels per run on run-relative time (stacked style).
    y_scale : {"term", "row", "shared"}
        Stacked rows share their amplitude axis within a term (and basis
        function), are scaled per row, or share one scale.
    time_range : (float, float), optional
        Zoom to this time window (seconds on the plotted axis).
    title, subtitle, caption : str, optional
        ``None`` uses the informative default; ``""`` removes it.
    figsize : (float, float), optional
        Defaults scale with the number of rows.
    ax : matplotlib Axes, optional
        Draw into this axes (overlay and heatmap styles only).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}, got {style!r}")
    if y_scale not in ("term", "row", "shared"):
        raise ValueError(f"y_scale must be 'term', 'row' or 'shared', got {y_scale!r}")
    if label_mode not in ("auto", "compact", "none"):
        raise ValueError(f"label_mode must be 'auto', 'compact' or 'none', got {label_mode!r}")
    if time_range is not None:
        tr_ = np.asarray(time_range, dtype=float)
        if tr_.shape != (2,) or not np.all(np.isfinite(tr_)) or tr_[0] >= tr_[1]:
            raise ValueError("time_range must be an increasing pair of numbers")
        time_range = (float(tr_[0]), float(tr_[1]))
    if hasattr(model, "betas") and hasattr(model, "model"):
        model = model.model
    if getattr(model, "event_model", None) is not None:
        model = model.event_model

    X = model_design_matrix(model)
    ci = design_columns(model)
    ci = ci[ci["kind"] == "event"].reset_index(drop=True)
    tl = timeline(model_sampling_frame(model), X.shape[0], block_x=block_x)
    has_frame = tl.unit == "s"

    if term is not None:
        names = [str(n) for n in getattr(model, "column_names", ci["name"])]
        pat = re.compile(rf"^{re.escape(term)}[_.\[]|^{re.escape(term)}$")
        keep = [i for i in range(len(ci)) if ci["term"].iloc[i] == term]
        if not keep:
            keep = [i for i in range(len(ci)) if pat.match(names[int(ci["col"].iloc[i])])]
        if not keep:
            raise ValueError(f"No columns found matching term name: {term}")
        ci = ci.iloc[keep].reset_index(drop=True)
    X = X[:, ci["col"].to_numpy(dtype=int)]

    row_key = _row_keys(ci)
    rows = list(dict.fromkeys(row_key))
    basis_ix = pd.to_numeric(ci["basis_total"], errors="coerce").fillna(1)
    n_basis = int(basis_ix.max()) if len(basis_ix) else 1
    if style == "auto":
        style = "heatmap" if (len(rows) > 12 or n_basis > 3) else "stacked"
    if ax is not None and style == "stacked":
        raise ValueError("the stacked style draws several panels; pass ax only with style='overlay' or 'heatmap'")
    suppress = label_mode == "none" or (label_mode == "auto" and X.shape[1] > max_labels and style != "heatmap")
    row_disp = dict(zip(rows, _abbreviate(rows, abbrev_min) if label_mode == "compact" else rows, strict=False))
    col_disp = _abbreviate(list(ci["full_label"]), abbrev_min) if label_mode == "compact" else list(ci["full_label"])

    colours = row_colours(ci, row_key)
    ev = event_table(model, ci, row_key, tl) if show_events else None
    n_ev = ev["event"].nunique() if ev is not None else 0
    parts = [plural(X.shape[1], "regressor")]
    if has_frame:
        parts += [plural(tl.n_runs, "run"), f"{X.shape[0]} scans (TR {fmt_tr(tl.TR)})"]
    if ev is not None:
        parts.append(plural(int(n_ev), "event"))
    title = resolve(title, "Event regressors")
    subtitle = resolve(subtitle, f" {DOT} ".join(parts))

    common: dict[str, Any] = {
        "model": model, "X": X, "ci": ci, "tl": tl, "row_key": row_key, "rows": rows, "colours": colours,
        "ev": ev, "suppress": suppress, "row_disp": row_disp, "col_disp": col_disp, "block_x": block_x,
        "time_range": time_range, "title": title, "subtitle": subtitle, "figsize": figsize,
    }
    if style == "heatmap":
        return _heatmap(**common, caption=caption, ax=ax)
    if style == "overlay":
        return _overlay(**common, caption=resolve(caption, _caption(ev, y_scale, "overlay")), ax=ax)
    return _stacked(**common, caption=resolve(caption, _caption(ev, y_scale, "stacked")), y_scale=y_scale)


def plot_design_matrix(
    model: Any,
    term_name: str | None = None,
    figsize: tuple[float, float] | None = None,
    separate_regressors: bool = True,
    **kwargs: Any,
) -> Any:
    """Alias of :func:`plot_event_model` kept for the former API.

    ``separate_regressors=False`` selects the overlay style.
    """
    style = kwargs.pop("style", "auto" if separate_regressors else "overlay")
    return plot_event_model(model, term=term_name, style=style, figsize=figsize, **kwargs)


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------


def _run_segments(tl: Timeline) -> list[np.ndarray]:
    return [np.flatnonzero(tl.run == r) for r in tl.runs["run"]]


def _x_frame(ax: Any, tl: Timeline, block_x: str, time_range: tuple[float, float] | None,
             label: bool = True, runs: list[int] | None = None) -> None:
    if tl.unit != "s":
        lo, hi = float(tl.runs["start"].iloc[0]), float(tl.runs["end"].iloc[-1])
        ax.set_xlim(*(time_range or (lo, hi)))
        if label:
            ax.set_xlabel("Scan")
        return
    sel = tl.runs if runs is None else tl.runs[tl.runs["run"].isin(runs)]
    lo, hi = float(sel["start"].min()), float(sel["end"].max())
    if time_range is not None:
        lo, hi = time_range
    time_axis(ax, lo, hi, label=("Time within run (min:s)" if block_x == "run" else "Time (min:s)") if label else None)


def _stacked(*, model: Any, X: np.ndarray, ci: pd.DataFrame, tl: Timeline, row_key: list[str], rows: list[str],
             colours: dict[str, str], ev: pd.DataFrame | None, suppress: bool, row_disp: dict[str, str],
             col_disp: list[str], block_x: str, time_range: Any, title: Any, subtitle: Any, caption: Any,
             figsize: Any, y_scale: str) -> Any:
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    # Display rows: condition x basis, in design order of first appearance.
    disp: list[tuple[str, int, str]] = []  # (row key, column index, label)
    for r in rows:
        js = [j for j, k in enumerate(row_key) if k == r]
        for b, j in enumerate(js):
            bl = ci["basis_label"].iloc[j]
            has_b = isinstance(bl, str) and len(js) > 1
            if not has_b:
                lab = row_disp[r]
            elif b == 0:
                lab = f"{row_disp[r]} {DOT} {bl}"
            else:
                lab = bl
            disp.append((r, j, "" if suppress else lab))
    n = len(disp)
    facet = block_x == "run" and tl.n_runs > 1 and tl.unit == "s"
    ncols = tl.n_runs if facet else 1
    row_h = 0.62 if n <= 8 else max(0.34, 5.6 / n)
    if figsize is None:
        figsize = (7.6, min(1.75 + n * row_h, 12.0))
    width_ratios = list(tl.runs["end"] - tl.runs["start"]) if facet else None
    fig, axes = new_figure(n, ncols, figsize=figsize, sharex="col", sharey="row", squeeze=False,
                           gridspec_kw={"width_ratios": width_ratios})

    # Shared y limits per scale group.
    def group_of(j: int) -> str:
        if y_scale == "shared":
            return "all"
        if y_scale == "row":
            return f"row{j}"
        return f"{ci['term'].iloc[j]}|{ci['basis_ix'].iloc[j]}"

    lims: dict[str, tuple[float, float]] = {}
    for _, j, _ in disp:
        g = group_of(j)
        v = X[:, j]
        if time_range is not None and tl.unit == "s":
            v = v[(tl.time >= time_range[0]) & (tl.time <= time_range[1])]
        lo, hi = min(0.0, float(np.nanmin(v)) if v.size else 0.0), max(0.0, float(np.nanmax(v)) if v.size else 0.0)
        a, b = lims.get(g, (lo, hi))
        lims[g] = (min(a, lo), max(b, hi))
    row_h_in = (figsize[1] - 1.75) / max(n, 1)
    min_frac = 11.0 / max(row_h_in * 72 / 1.30, 1.0)

    segs = _run_segments(tl)
    ev_by_row = dict(list(ev.groupby("row"))) if ev is not None else {}
    first_basis = {r: next(i for i, (k, _, _) in enumerate(disp) if k == r) for r in rows}
    with themed():
        for i, (r, j, lab) in enumerate(disp):
            lo, hi = lims[group_of(j)]
            if hi - lo <= 0:
                lo, hi = -1.0, 1.0
            rng = hi - lo
            ylim = (lo - 0.22 * rng, hi + 0.08 * rng)
            col = colours[r]
            for c in range(ncols):
                ax = axes[i, c]
                run_list = [int(tl.runs["run"].iloc[c])] if facet else list(tl.runs["run"])
                for k, idx in enumerate(segs):
                    if int(tl.runs["run"].iloc[k]) not in run_list:
                        continue
                    t, y = tl.time[idx], X[idx, j]
                    ax.fill_between(t, 0, y, color=col, alpha=0.12, linewidth=0)
                    ax.plot(t, y, color=col, linewidth=1.1, solid_joinstyle="round")
                ax.axhline(0, color=INK["rule"], linewidth=0.6, zorder=1.2)
                ax.set_ylim(*ylim)
                ax.grid(False)
                if tl.unit == "s" and not facet and tl.n_runs > 1:
                    run_axis(ax, tl.bounds, top=(i == 0))
                elif facet and i == 0:
                    ax.set_title(f"Run {run_list[0]}", loc="center", fontweight="normal",
                                 fontsize=BASE_SIZE * 0.82, color=INK["text2"], pad=3)
                if ev is not None and i == first_basis[r] and r in ev_by_row:
                    e = ev_by_row[r]
                    if facet:
                        start = float(tl.runs["start"].iloc[c])
                        e = e[e["event"].map(lambda v: v[0]) == c]
                        on = e["onset"].to_numpy() - start if start else e["onset"].to_numpy()
                    else:
                        on = e["onset"].to_numpy()
                    tr = ax.get_xaxis_transform()
                    for t0 in on:
                        ax.plot([t0, t0], [0.0, 0.16], transform=tr, color=col, linewidth=0.9, alpha=0.9,
                                solid_capstyle="butt", zorder=3)
                    d = e["duration"].to_numpy()
                    for t0, dd in zip(on, d, strict=False):
                        if dd > 0:
                            ax.plot([t0, t0 + dd], [0.012, 0.012], transform=tr, color=col, linewidth=3.2,
                                    solid_capstyle="butt", zorder=3)
                if c == ncols - 1:
                    ax.yaxis.tick_right()
                    ticks = _peak_breaks(lo, hi, min_frac)
                    ax.yaxis.set_major_locator(FixedLocator(ticks))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: fd_num(v, 2)))
                    ax.tick_params(axis="y", length=0, labelsize=BASE_SIZE * 0.72, labelcolor=INK["muted"], pad=2)
                else:
                    ax.yaxis.set_major_locator(NullLocator())
                    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
                if c == 0 and lab:
                    row_label(ax, lab, size=BASE_SIZE * 0.86)
                if i < n - 1:
                    ax.tick_params(axis="x", length=0, labelbottom=False)
            if i == n - 1:
                for c in range(ncols):
                    run_list = [int(tl.runs["run"].iloc[c])] if facet else None
                    _x_frame(axes[i, c], tl, block_x, time_range, label=(c == 0 or not facet), runs=run_list)
                    axes[i, c].tick_params(axis="x", length=2.5, color=INK["rule"])
                if facet and ncols > 1:
                    for c in range(1, ncols):
                        axes[i, c].set_xlabel("")
        finish(fig, title, subtitle, caption)
    return fig


def _overlay(*, model: Any, X: np.ndarray, ci: pd.DataFrame, tl: Timeline, row_key: list[str], rows: list[str],
             colours: dict[str, str], ev: pd.DataFrame | None, suppress: bool, row_disp: dict[str, str],
             col_disp: list[str], block_x: str, time_range: Any, title: Any, subtitle: Any, caption: Any,
             figsize: Any, ax: Any) -> Any:
    p = X.shape[1]
    reg_cols = palette(p)
    own = ax is None
    if own:
        fig, ax = new_figure(figsize=figsize or (7.6, 4.6))
    else:
        fig = ax.figure
    segs = _run_segments(tl)
    with themed():
        handles = []
        for j in range(p):
            h = None
            for idx in segs:
                (h,) = ax.plot(tl.time[idx], X[idx, j], color=reg_cols[j], linewidth=1.2)
            handles.append(h)
        ax.grid(axis="x", visible=False)
        if tl.unit == "s" and tl.n_runs > 1 and block_x == "global":
            run_axis(ax, tl.bounds)
        if ev is not None:
            tr = ax.get_xaxis_transform()
            for t0 in np.unique(ev["onset"].to_numpy()):
                ax.plot([t0, t0], [0, 0.03], transform=tr, color=INK["muted"], linewidth=0.8,
                        solid_capstyle="butt", zorder=3)
        lo, hi = float(np.nanmin(X)), float(np.nanmax(X))
        pad = 0.06 * (hi - lo or 1.0)
        ax.set_ylim(lo - pad * 1.6, hi + pad)
        from ._helpers import budget_yticks

        budget_yticks(ax)
        ax.set_ylabel("Regressor amplitude (a.u.)")
        _x_frame(ax, tl, block_x, time_range)
        if not suppress:
            labels = col_disp
            width = sum(len(s) + 5 for s in labels)
            ncol = p if width <= 80 else max(1, 80 // max(len(s) + 5 for s in labels))
            ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.07 if tl.n_runs > 1 else 1.0),
                      ncol=min(ncol, p), frameon=False, handlelength=1.4, columnspacing=1.4,
                      borderaxespad=0.0)
        if own:
            finish(fig, title, subtitle, caption)
        elif title:
            ax.set_title(title)
    return fig


def _heatmap(*, model: Any, X: np.ndarray, ci: pd.DataFrame, tl: Timeline, row_key: list[str], rows: list[str],
             colours: dict[str, str], ev: pd.DataFrame | None, suppress: bool, row_disp: dict[str, str],
             col_disp: list[str], block_x: str, time_range: Any, title: Any, subtitle: Any, caption: Any,
             figsize: Any, ax: Any) -> Any:
    p = X.shape[1]
    order = list(range(p))
    no_basis = pd.to_numeric(ci["basis_total"], errors="coerce").fillna(1).max() <= 1
    if ev is not None and no_basis:
        first = ev.groupby("row")["onset"].min().to_dict()
        order = sorted(range(p), key=lambda j: (first.get(row_key[j], np.inf), j))
    peak = float(np.nanmax(np.abs(X))) if X.size else 1.0
    peak = peak if peak > 0 else 1.0
    signed = bool(np.any(np.less(X, -0.01 * peak)))
    show_r = 1 < p <= 80 and block_x == "global"
    max_r = None
    if show_r:
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.corrcoef(X.T)
        R = np.where(np.isfinite(R), R, np.nan)
        np.fill_diagonal(R, np.nan)
        max_r = np.nanmax(np.abs(R), axis=0) if np.isfinite(R).any() else np.full(p, np.nan)

    own = ax is None
    if own:
        h = min(max(2.6, 1.9 + 0.36 * min(p, 40)), 11.0)
        fig, ax = new_figure(figsize=figsize or (7.6, h))
    else:
        fig = ax.figure
    with themed():
        cax = cbar_axes(ax, "top", gap_pt=30 if (tl.unit == "s" and tl.n_runs > 1) else 18)

    edges = np.empty(X.shape[0] + 1)
    t = tl.time
    if tl.unit == "s":
        trs = tl.TR[tl.run - 1]
        lo_e = t - trs / 2
        edges[:-1] = lo_e
        edges[-1] = t[-1] + trs[-1] / 2
    else:
        edges = np.arange(X.shape[0] + 1) + 0.5
    with themed():
        cm = fd_cmap("diverging" if signed else "magnitude")
        norm = signed_norm(peak, None if signed else 0.0)
        Z = X[:, order].T[::-1]  # top row = first in order
        mesh = ax.pcolormesh(edges, np.arange(p + 1) - 0.5, Z, cmap=cm, norm=norm, shading="flat",
                             rasterized=True)
        for k in range(1, p):
            ax.axhline(k - 0.5, color="white", linewidth=1.6 if p <= 40 else 0.4)
        ax.set_ylim(-0.5, p - 0.5)
        ax.grid(False)
        if tl.unit == "s" and tl.n_runs > 1:
            for b in tl.bounds[1:-1]:
                ax.axvline(b, color=INK["muted"], linewidth=0.7, zorder=3)
            run_axis(ax, tl.bounds)
            for line in ax.lines[-(tl.n_runs - 1):]:
                line.set_color(INK["muted"])
        pos = {j: p - 1 - k for k, j in enumerate(order)}
        if ev is not None:
            first_col = {}
            for j, k in enumerate(row_key):
                first_col.setdefault(k, j)
            xs, ys = [], []
            for k, g in ev.groupby("row"):
                if k in first_col:
                    xs.extend(g["onset"].tolist())
                    ys.extend([pos[first_col[k]]] * len(g))
            ax.scatter(xs, ys, marker="|", s=70 if p <= 20 else 30, color=INK["text"], linewidths=0.9, zorder=4)
        step = int(np.ceil(p / 25)) if p > 40 else 1
        shown = order[::step]
        if suppress:
            ax.set_yticks([])
        else:
            ax.set_yticks([pos[j] for j in shown])
            ax.set_yticklabels([col_disp[j] for j in shown], fontsize=BASE_SIZE * (0.82 if p <= 20 else 0.7),
                               color=INK["text2"])
        ax.tick_params(axis="y", length=0)
        _x_frame(ax, tl, block_x, time_range)
        if show_r and max_r is not None:
            tr = ax.get_yaxis_transform()
            fs = BASE_SIZE * (0.8 if p <= 20 else 0.65)
            for j in range(p):
                if (p > 40 and j not in shown) or not np.isfinite(max_r[j]):
                    continue
                v = max_r[j]
                s = f"{v:.2f}".lstrip("0") if v < 1 else "1"
                ax.text(1.012, pos[j], s, transform=tr, ha="left", va="center", fontsize=fs,
                        color=WARN if v >= 0.5 else INK["muted"], clip_on=False)
            ax.text(1.012, p - 0.5, "max r", transform=tr, ha="left", va="bottom", fontsize=BASE_SIZE * 0.8,
                    fontweight="bold", color=INK["text2"], clip_on=False)
        colourbar(fig, cax, mesh, "Amplitude (a.u.)",
                  ticks=budget_breaks(-peak if signed else 0.0, peak, stack=True))
        if caption is None:
            parts = []
            if ev is not None and no_basis:
                parts.append("Rows ordered by first onset; ticks mark event onsets.")
            elif ev is not None:
                parts.append("Ticks mark event onsets.")
            parts.append("Cells are sampled at the TR.")
            if show_r:
                parts.append("max r: largest correlation with any other regressor (orange when 0.5 or more).")
            caption = " ".join(parts)
        if own:
            finish(fig, title, subtitle, caption or None)
        elif title:
            ax.set_title(title)
    return fig


__all__ = ["event_table", "plot_design_matrix", "plot_event_model"]
