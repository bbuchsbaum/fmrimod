"""Baseline-model and sampling-frame plots (R fmridesign ``plot_baseline.R``).

``plot_baseline_model`` draws every time-varying baseline term on a shared
time axis, one lane per term (motion parameters split into translation and
rotation), with in-panel keys, run hairlines and a caption listing what is
constant and omitted. ``plot_sampling_frame`` draws the acquisition: runs
as bars on one time axis, as scan grids, or as a compact lane.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


import numpy as np
import pandas as pd

from ._design_common import (
    NDASH,
    design_columns,
    fd_num,
    finish_figure,
    fmt_tr,
    model_design_matrix,
    plural,
    resolve,
    timeline,
)
from ._helpers import DOT, MINUS, TIMES, mmss, new_figure, run_axis, themed, time_axis
from .style import BASE_SIZE, CATEGORICAL, INK, palette, tints

_BASIS_NAMES = {"bs": "B-spline", "ns": "natural spline", "poly": "polynomial", "constant": "constant",
                "cosine": "cosine"}
_MOTION_LANES = ("Translation", "Rotation", "Other nuisance")


# --------------------------------------------------------------------------
# Baseline model: data
# --------------------------------------------------------------------------


def motion_kind(label: str) -> str | None:
    """``"translation"``, ``"rotation"`` or ``None`` for a nuisance column name
    (``tx``/``rot_x``/``trans_y``/``x``/``pitch`` ...; R ``.fd_bl_motion_kind``)."""
    lab = label.lower()
    if (re.match(r"^(r|rot)[_.]?[xyz]($|[_.])", lab) or lab.startswith("rot")
            or re.search(r"(^|[_.])(pitch|roll|yaw)($|[_.])", lab)):
        return "rotation"
    if re.match(r"^(t|trans)[_.]?[xyz]($|[_.])", lab) or lab.startswith("trans") or re.match(r"^[xyz]($|[_.])", lab):
        return "translation"
    return None


def _term_columns(model: object) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    ci = design_columns(model)
    by_term: dict[str, list[int]] = {}
    for i, t in enumerate(ci["term"]):
        by_term.setdefault(str(t), []).append(i)
    return ci, by_term


def _constant_within_runs(X: np.ndarray, run: np.ndarray) -> bool:
    for r in np.unique(run):
        seg = X[run == r]
        if seg.size and np.any(np.ptp(seg, axis=0) > 1e-10 * max(1.0, float(np.abs(seg).max()))):
            return False
    return True


def _match_term(term: str, names: list[str]) -> str:
    if term in names:
        return term
    hits = [n for n in names if term.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"Specified term '{term}' not found. Available terms: {', '.join(names)}")
    raise ValueError(f"Specified term '{term}' matches multiple terms: {', '.join(hits)}. Please be more specific.")


def _lanes(term: str, labels: list[str]) -> list[str]:
    if term == "drift":
        return ["Drift"] * len(labels)
    if term == "block":
        return ["Intercept"] * len(labels)
    if term != "nuisance":
        return [term.capitalize()] * len(labels)
    kinds = [motion_kind(lab) for lab in labels]
    if "translation" in kinds and "rotation" in kinds:
        return ["Translation" if k == "translation" else "Rotation" if k == "rotation" else "Other nuisance"
                for k in kinds]
    return ["Nuisance"] * len(labels)


def _lane_order(lanes: list[str]) -> list[str]:
    uniq = list(dict.fromkeys(lanes))
    if not any(ln in _MOTION_LANES for ln in uniq):
        return uniq
    first = next(i for i, ln in enumerate(uniq) if ln in _MOTION_LANES)
    rest = [ln for ln in uniq if ln not in _MOTION_LANES]
    before = [ln for ln in rest if ln in uniq[:first]]
    return before + [m for m in _MOTION_LANES if m in uniq] + [ln for ln in rest if ln not in before]


def _span_text(v: np.ndarray, unit: str = "s") -> str:
    v = np.asarray(v, dtype=float)
    if v.max() - v.min() < 1:
        return f"{fd_num(v[0], 3)} {unit}"
    return f"{fd_num(v.min(), 3)}{NDASH}{fd_num(v.max(), 3)} {unit}"


def drift_cutoff_note(basis: str, k_per_run: np.ndarray, run_dur: np.ndarray) -> str | None:
    """Rough DCT-equivalent high-pass period ``2 T / k`` for spline/polynomial
    drift (R ``.fd_bl_cutoff_note``); ``None`` for other bases."""
    if basis not in ("bs", "ns", "poly"):
        return None
    ok = k_per_run > 0
    if not ok.any():
        return None
    k = k_per_run[ok]
    dur = run_dur[ok]
    period = 2 * dur / k
    k_txt = str(int(k[0])) if len(set(k.tolist())) == 1 else f"{int(k.min())}{NDASH}{int(k.max())}"
    what = {"bs": f"B-spline columns (degree-{k_txt} polynomial, no interior knots)",
            "ns": "natural-spline columns", "poly": "polynomial columns"}[basis]
    return (f"Drift: {k_txt} {what} per {_span_text(dur)} run.\nRoughly comparable to a DCT high-pass with "
            f"cutoff ~{_span_text(period)} (DCT rule: 2 {TIMES} run duration / columns).")


# --------------------------------------------------------------------------
# Baseline model: plot
# --------------------------------------------------------------------------


def plot_baseline_model(
    model: object,
    term: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    block_x: str = "global",
    drift_scale: str = "raw",
    figsize: tuple[float, float] | None = None,
    line_width: float = 1.2,
    term_name: str | None = None,
) -> Figure:
    """Plot the time-varying terms of a baseline model (R ``plot.baseline_model``).

    One lane per term on a shared time axis: drift basis functions keep
    their colour in every run, nuisance regressors keep the names you
    supplied, and when those look like rigid-body motion parameters
    translations and rotations get separate lanes (their units differ).
    Each lane has a key at its right edge in column order (a legend below
    the plot past seven series). Lines never cross runs. Terms constant
    within every run (run intercepts, a global intercept) are left out and
    listed in the caption; request one with ``term`` to draw it. For
    spline and polynomial drift the caption gives a rough DCT-equivalent
    high-pass period, ``2 x run duration / columns``.

    Parameters
    ----------
    model : BaselineModel
    term : str, optional
        One term to draw (``"drift"``, ``"block"`` or ``"nuisance"``; an
        exact or unique partial match). ``term_name`` is an alias.
    title, subtitle, caption : str, optional
        ``None`` uses the default; ``""`` removes it.
    block_x : {"global", "run"}
        Runs concatenated on one time axis, or one column per run.
    drift_scale : {"raw", "unit"}
        Plot drift as it enters the design, or divide each basis function
        by its peak within its run (the caption then gives the raw peaks).
    figsize : (float, float), optional
    line_width : float

    Returns
    -------
    matplotlib.figure.Figure
    """
    if block_x not in ("global", "run"):
        raise ValueError(f"block_x must be 'global' or 'run', got {block_x!r}")
    if drift_scale not in ("raw", "unit"):
        raise ValueError(f"drift_scale must be 'raw' or 'unit', got {drift_scale!r}")
    term = term if term is not None else term_name
    X_all = model_design_matrix(model)
    if X_all.shape[1] == 0:
        raise ValueError("Baseline model contains no terms.")
    sf = model.sampling_frame
    n = X_all.shape[0]
    tl = timeline(sf, n, block_x=block_x)
    ci, by_term = _term_columns(model)
    names = list(by_term)
    constant = {t: _constant_within_runs(X_all[:, cols], tl.run) for t, cols in by_term.items()}
    if term is None:
        plot_terms = [t for t in names if not constant[t]] or names
    else:
        plot_terms = [_match_term(term, names)]

    # Long data: one series per (column, run); run-specific columns only in their run.
    series: list[dict[str, Any]] = []
    for t in plot_terms:
        cols = by_term[t]
        labels = [str(ci["label"].iloc[c]) for c in cols]
        lanes = _lanes(t, labels)
        for c, lab, lane in zip(cols, labels, lanes, strict=False):
            r_col = ci["run"].iloc[c]
            runs = [int(r_col)] if (r_col is not None and not (isinstance(r_col, float) and np.isnan(r_col))) \
                else list(tl.runs["run"])
            for r in runs:
                idx = np.flatnonzero(tl.run == r)
                if idx.size:
                    series.append({"term": t, "col": c, "label": lab, "lane": lane, "run": r,
                                   "t": tl.time[idx], "y": X_all[idx, int(ci["col"].iloc[c])].copy()})
    if not series:
        raise ValueError("Nothing to plot: the selected baseline terms have no columns.")

    notes: list[str] = []
    basis = str(getattr(getattr(model, "drift_spec", None), "basis", "") or "")
    drift_cols = by_term.get("drift", [])
    k_per_run = np.array([sum(1 for c in drift_cols if ci["run"].iloc[c] == r) for r in tl.runs["run"]])
    if "drift" in plot_terms and tl.unit == "s" and drift_cols and all(
            ci["run"].iloc[c] is not None and not (isinstance(ci["run"].iloc[c], float)
                                                  and np.isnan(ci["run"].iloc[c])) for c in drift_cols):
        note = drift_cutoff_note(basis, k_per_run, (tl.runs["end"] - tl.runs["start"]).to_numpy())
        if note:
            notes.append(note)
    if drift_scale == "unit" and any(s["term"] == "drift" for s in series):
        peaks, ranges = [], []
        for s in series:
            if s["term"] == "drift":
                pk = float(np.abs(s["y"]).max())
                peaks.append(pk)
                ranges.append(float(np.ptp(s["y"])))
                if pk > 0:
                    s["y"] = s["y"] / pk
        note = (f"Drift scaled: each basis function divided by its peak |value| in its run "
                f"(raw peaks {fd_num(min(peaks))}{NDASH}{fd_num(max(peaks))}).")
        dr = [s for s in series if s["term"] == "drift"]
        flat = [f"{s['label']} (run {s['run']})" for s, rg in zip(dr, ranges, strict=False) if rg < 0.01 * max(ranges)]
        if flat:
            note += " Near-flat before scaling: " + ", ".join(dict.fromkeys(flat)) + "."
        notes.append(note)
    lane_order = _lane_order([s["lane"] for s in series])
    motion_split = any(ln in ("Translation", "Rotation") for ln in lane_order)
    if motion_split:
        notes.append("Translation and rotation are plotted in the units supplied.")
    if term is None:
        hidden = [t for t in names if constant[t] and t not in plot_terms]
        if hidden:
            desc = []
            for t in hidden:
                k = len(by_term[t])
                what = ("global intercept" if k == 1 else "run intercepts") if t == "block" else \
                    ("constant drift" if t == "drift" else t)
                desc.append(f"{what} ({k} column{'' if k == 1 else 's'})")
            notes.append("Not shown (constant within each run): " + ", ".join(desc) + ".")

    # Colours per lane in column order (shared across runs and motion lanes).
    colour: dict[tuple[str, str], str] = {}
    keys_by_lane: dict[str, list[str]] = {}
    for ln in lane_order:
        labs = list(dict.fromkeys(s["label"] for s in series if s["lane"] == ln))
        keys_by_lane[ln] = labs
        for lab, c in zip(labs, palette(len(labs)), strict=False):
            colour[(ln, lab)] = c
    big = [ln for ln in lane_order if len(keys_by_lane[ln]) > 7]

    if subtitle is None:
        parts = []
        if drift_cols:
            k = k_per_run[k_per_run > 0]
            per_run = len(k) > 0
            kk = (str(int(k[0])) if len(set(k.tolist())) == 1 else f"{int(k.min())}{NDASH}{int(k.max())}") \
                if per_run else str(len(drift_cols))
            parts.append(f"Drift: {_BASIS_NAMES.get(basis, basis)}, {kk}{' per run' if per_run else ''}")
        nz = by_term.get("nuisance", [])
        if nz:
            kn = np.array([sum(1 for c in nz if ci["run"].iloc[c] == r) for r in tl.runs["run"]])
            kn = kn[kn > 0]
            if len(kn):
                txt = str(int(kn[0])) if len(set(kn.tolist())) == 1 else f"{int(kn.min())}{NDASH}{int(kn.max())}"
                parts.append(f"{txt} nuisance per run")
            else:
                parts.append(f"{len(nz)} nuisance")
        parts.append(plural(tl.n_runs, "run"))
        if tl.unit == "s":
            parts.append(f"{n} scans (TR {fmt_tr(tl.TR)})")
        subtitle = f" {DOT} ".join(parts)
    subtitle = resolve(subtitle, None)
    caption = resolve(caption, "\n".join(notes) if notes else None)
    title = resolve(title, "Baseline model")

    n_lanes = len(lane_order)
    facet = block_x == "run" and tl.n_runs > 1
    ncols = tl.n_runs if facet else 1
    if figsize is None:
        max_key = max((len(k) for ln in lane_order if ln not in big for k in keys_by_lane[ln]), default=0)
        figsize = (7.6 + max_key * 0.06, min(1.7 + 2.5 * n_lanes + 0.18 * len(notes), 12.0))
    width_ratios = list(tl.runs["end"] - tl.runs["start"]) if facet else None
    fig, axes = new_figure(n_lanes, ncols, figsize=figsize, sharex="col", sharey="row", squeeze=False,
                           gridspec_kw={"width_ratios": width_ratios})
    from matplotlib.lines import Line2D

    with themed():
        for li, ln in enumerate(lane_order):
            lane_series = [s for s in series if s["lane"] == ln]
            for c in range(ncols):
                ax = axes[li, c]
                run_c = int(tl.runs["run"].iloc[c]) if facet else None
                for s in lane_series:
                    if facet and s["run"] != run_c:
                        continue
                    ax.plot(s["t"], s["y"], color=colour[(ln, s["label"])], linewidth=line_width)
                ax.grid(axis="x", visible=False)
                if tl.unit == "s" and not facet and tl.n_runs > 1:
                    run_axis(ax, tl.bounds, top=(li == 0))
                if facet and li == 0:
                    ax.set_title(f"Run {run_c}", loc="center", fontweight="normal", fontsize=BASE_SIZE * 0.82,
                                 color=INK["text2"], pad=3)
                if c == 0:
                    if facet:
                        ax.set_ylabel(ln, fontweight="bold", color=INK["text"])
                    else:
                        ax.set_title(ln, loc="left", fontsize=BASE_SIZE * 0.86, pad=4)
                if li < n_lanes - 1:
                    ax.tick_params(axis="x", length=0, labelbottom=False)
            # Pad the y range and add the in-panel key at the right edge.
            ax0 = axes[li, 0]
            ys = np.concatenate([s["y"] for s in lane_series])
            lo, hi = float(ys.min()), float(ys.max())
            pad = 0.06 * (hi - lo or 1.0)
            ax0.set_ylim(lo - pad, hi + pad)
            from ._helpers import budget_yticks

            budget_yticks(ax0)
            if ln not in big:
                labs = keys_by_lane[ln]
                k = len(labs)
                h = max(hi - lo, 1e-8)
                gap = max(0.13 * h, min(0.22 * h, h / (k - 1))) if k > 1 else 0.0
                mid = (hi + lo) / 2
                axr = axes[li, -1]
                tr = axr.get_yaxis_transform()
                for j, lab in enumerate(labs):
                    y = mid + gap * ((k - 1) / 2 - j)
                    axr.plot([1.012, 1.035], [y, y], transform=tr, color=colour[(ln, lab)], linewidth=3.2,
                             solid_capstyle="butt", clip_on=False)
                    axr.text(1.045, y, lab, transform=tr, ha="left", va="center", fontsize=BASE_SIZE * 0.82,
                             color=INK["text"], clip_on=False)
        for c in range(ncols):
            axb = axes[-1, c]
            if tl.unit == "s":
                sel = tl.runs if not facet else tl.runs.iloc[[c]]
                time_axis(axb, float(sel["start"].min()), float(sel["end"].max()), target=7,
                          label=("Time within run (min:s)" if facet else "Time (min:s)") if c == 0 else None)
            else:
                axb.set_xlabel("Scan" if c == 0 else "")
        if not motion_split and not facet:
            axes[(n_lanes - 1) // 2, 0].set_ylabel("Regressor value")
        if big:
            handles, labels = [], []
            seen: dict[str, list[str]] = {}
            for ln in big:
                for lab in keys_by_lane[ln]:
                    seen.setdefault(colour[(ln, lab)], []).append(lab)
            for col, labs in seen.items():
                handles.append(Line2D([], [], color=col, linewidth=2.4))
                labels.append(" / ".join(dict.fromkeys(labs)))
            from matplotlib.transforms import ScaledTranslation

            axl = axes[-1, 0]
            below = axl.transAxes + ScaledTranslation(0, -0.46, fig.dpi_scale_trans)
            axl.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 0.0), bbox_transform=below,
                       ncol=min(len(labels), 6), frameon=False, borderaxespad=0.0)
        finish_figure(fig, title, subtitle, caption)
    return fig


# --------------------------------------------------------------------------
# Sampling frame
# --------------------------------------------------------------------------


def _frame_events(events: object, runs: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Run-relative onsets and run numbers for the coverage overlay, and a
    caption note (R ``.fd_sf_events``)."""
    if hasattr(events, "events") and hasattr(events, "blockids") and isinstance(events.events, dict):
        ev0 = next(iter(events.events.values()))
        ev = pd.DataFrame({"onset": np.asarray(ev0.onsets, dtype=float),
                           "run": np.asarray(events.blockids, dtype=int)})
        ids = sorted(ev["run"].unique())
        ev["run"] = [ids.index(r) + 1 for r in ev["run"]]
    elif isinstance(events, pd.DataFrame) and "onset" in events.columns and (
            "run" in events.columns or "block" in events.columns):
        rc = "run" if "run" in events.columns else "block"
        r = events[rc]
        r = r if pd.api.types.is_numeric_dtype(r) else pd.Categorical(r).codes + 1
        ev = pd.DataFrame({"onset": events["onset"].astype(float).to_numpy(), "run": np.asarray(r, dtype=int)})
    else:
        raise ValueError("`events` must be an EventModel or a DataFrame with `onset` and `run` (or `block`) columns.")
    ev = ev[np.isfinite(ev["onset"])]
    dur = dict(zip(runs["run"], runs["end"] - runs["start"], strict=False))
    inside = ev.apply(lambda r: r["run"] in dur and 0 <= r["onset"] < dur[int(r["run"])], axis=1) \
        if len(ev) else pd.Series(dtype=bool)
    kept = ev[inside.to_numpy(dtype=bool)] if len(ev) else ev
    n_out = len(ev) - len(kept)
    empty = [int(r) for r in runs["run"] if r not in set(kept["run"])]
    note = [f"Ticks below each run: {len(kept)} event onset{'' if len(kept) == 1 else 's'}."]
    if n_out:
        note.append(f"{n_out} onset{'' if n_out == 1 else 's'} outside the sampled runs not shown.")
    if empty:
        note.append(f"No events in run{'' if len(empty) == 1 else 's'} {', '.join(map(str, empty))}.")
    return kept, " ".join(note)


def _frame_breaks(runs: pd.DataFrame, total: float) -> list[float]:
    """Run boundaries, thinned so labels do not collide (R ``.fd_sf_breaks``)."""
    if len(runs) == 1:
        if total < 150:
            step = 30 if total > 60 else 10
        else:
            step = next((s for s in (10, 15, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600) if total / s <= 9),
                        3600)
        return list(np.arange(0, total + 1e-9, step))
    b = sorted(set(runs["start"]).union(runs["end"]))
    keep = [b[0]]
    for v in b[1:]:
        if v - keep[-1] >= total * 0.06:
            keep.append(v)
    if keep[-1] != b[-1]:
        keep[-1] = b[-1]
    return keep


def _stack(left: np.ndarray, right: np.ndarray, gap: float) -> np.ndarray:
    row_end: list[float] = []
    out = np.zeros(len(left), dtype=int)
    for i in np.argsort(left):
        free = [r for r, e in enumerate(row_end) if e + gap <= left[i]]
        r = free[0] if free else len(row_end)
        if r == len(row_end):
            row_end.append(right[i])
        else:
            row_end[r] = right[i]
        out[i] = r
    return out


def plot_sampling_frame(
    sframe: object,
    style: str = "timeline",
    show_ticks: bool = False,
    tick_every: int | None = None,
    events: object = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the runs of a sampling frame (R ``plot.sampling_frame``).

    Parameters
    ----------
    sframe : SamplingFrame
    style : {"timeline", "grid", "lane"}
        ``"timeline"``: one bar per run on a shared time axis; ``"grid"``:
        each run's scans as cells on a within-run scan axis; ``"lane"``: one
        row of adjacent run segments annotated above.
    show_ticks : bool
        Timeline: white ticks every ``tick_every`` scans inside each bar.
    tick_every : int, optional
        Scans between ticks (timeline) or between shading changes (grid;
        default 5, or 10 for runs over 100 scans).
    events : EventModel or DataFrame, optional
        Onsets (``onset`` run-relative seconds and ``run``/``block``) drawn
        as ticks under each run, as a coverage check; runs without events
        and onsets outside their run are noted in the caption.
    title, subtitle, caption : str, optional
        ``None`` uses the default; ``""`` removes it.
    figsize : (float, float), optional
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.patches import Rectangle

    if style not in ("timeline", "grid", "lane"):
        raise ValueError(f"style must be 'timeline', 'grid' or 'lane', got {style!r}")
    bl = np.asarray(sframe.blocklens, dtype=int)
    runs = timeline(sframe, int(bl.sum())).runs.copy()
    runs["dur"] = runs["end"] - runs["start"]
    n_runs = len(runs)
    if tick_every is None:
        tick_every = 10 if (style == "grid" and bl.max() > 100) else 5
    if not tick_every > 0:
        raise ValueError("tick_every must be positive")
    tick_every = max(1, int(round(tick_every)))
    trs = list(dict.fromkeys(float(v) for v in runs["TR"]))
    per_run_tr = len(trs) > 1
    total = float(runs["dur"].sum())
    tr_text = f"{trs[0]:g} s" if not per_run_tr else f"{min(trs):g}{NDASH}{max(trs):g} s (varies by run)"
    sub_default = f" {DOT} ".join([plural(n_runs, "run"), f"{int(bl.sum())} scans", f"TR {tr_text}",
                                   f"total {mmss(total)}"])
    counts = runs["scans"].value_counts()
    modal = int(counts.idxmax())
    has_mode = n_runs > 2 and counts.max() > 1 and counts.max() >= n_runs / 2
    runs["odd"] = has_mode & (runs["scans"] != modal)
    labs = []
    for r in runs.itertuples():
        delta = f" ({'+' if r.scans > modal else MINUS}{abs(r.scans - modal)})" if r.odd else ""
        tr_part = f"TR {r.TR:g} s {DOT} " if per_run_tr else ""
        labs.append(f"{r.scans} scans{delta} {DOT} {tr_part}{mmss(r.dur)}")
    runs["lab"] = labs
    odd_note = f"Bold: run length differs from the most common length ({modal} scans)." if runs["odd"].any() else None
    ev, ev_note = (_frame_events(events, runs) if events is not None else (None, None))
    has_ev = ev is not None and len(ev) > 0
    runs["y"] = n_runs - runs["run"] + 1
    half = 0.36
    concat_note = "Runs are concatenated on one time axis." if n_runs > 1 else None

    own = ax is None
    if own:
        h = 1.9 + 0.32 * n_runs + (0.5 if style == "lane" else 0.0) + (0.2 if has_ev else 0.0)
        fig, ax = new_figure(figsize=figsize or (7.6, max(h, 2.4)))
    else:
        fig = ax.figure
    blue = CATEGORICAL[0]
    blue2 = tints(blue, 4)[1]
    with themed():
        ax.grid(axis="y", visible=False)
        if style == "timeline":
            char_w = total / 115
            pad = total * 0.008
            need = runs["lab"].str.len() * char_w
            inside = bool((need < runs["dur"] * 0.85).all())
            for r in runs.itertuples():
                ax.add_patch(Rectangle((r.start, r.y - half), r.dur, 2 * half, facecolor=blue, edgecolor="none"))
                if show_ticks and r.scans > tick_every:
                    for k in range(tick_every, int(r.scans), tick_every):
                        x = r.start + k * r.TR
                        ax.plot([x, x], [r.y - half, r.y - half * 0.4], color="white", linewidth=0.8, alpha=0.85)
                ax.text(r.start + pad if inside else r.end + pad, r.y, r.lab, ha="left", va="center",
                        fontsize=BASE_SIZE * 0.86, color="white" if inside else INK["text2"],
                        fontweight="bold" if r.odd else "normal")
            right = 0.015 * total if inside else float((runs["end"] + pad + need).max()) - total + 0.01 * total
            ax.set_xlim(-0.01 * total, total + right)
            if has_ev:
                for e in ev.itertuples():
                    rr = runs[runs["run"] == e.run].iloc[0]
                    x = rr["start"] + e.onset
                    ax.plot([x, x], [rr["y"] - half - 0.04, rr["y"] - half - 0.16], color=INK["text"], linewidth=0.8)
            from matplotlib.ticker import FixedLocator, FuncFormatter

            ax.xaxis.set_major_locator(FixedLocator(_frame_breaks(runs, total)))
            ax.xaxis.set_major_formatter(FuncFormatter(mmss))
            ax.set_xlabel("Time (min:s)")
            ax.grid(axis="x", visible=False)
            notes = [concat_note, f"Ticks every {tick_every} scans." if show_ticks else None]
            default_title = "Acquisition timeline"
        elif style == "lane":
            runs["mid"] = (runs["start"] + runs["end"]) / 2
            cw = total / 125
            w = np.maximum(runs["lab"].str.len(), runs["run"].astype(str).str.len() + 4) * cw
            two_line = bool((w <= runs["dur"] * 0.95).all())
            if two_line:
                runs["lx"], runs["ly"] = runs["mid"], 1.5
            else:
                runs["lab1"] = [f"Run {r}: {lab}" for r, lab in zip(runs["run"], runs["lab"], strict=False)]
                w = runs["lab1"].str.len() * cw
                runs["lx"] = np.minimum(np.maximum(runs["mid"], w / 2), total - w / 2)
                row = _stack((runs["lx"] - w / 2).to_numpy(), (runs["lx"] + w / 2).to_numpy(), total * 0.01)
                runs["ly"] = 1.55 + row * 0.55
            for r in runs.itertuples():
                ax.add_patch(Rectangle((r.start, 0.6), r.dur, 0.8, facecolor=blue, edgecolor="white", linewidth=1.6))
                if two_line:
                    ax.text(r.lx, r.ly + 0.42, f"Run {r.run}", ha="center", va="bottom", fontsize=BASE_SIZE * 0.9,
                            fontweight="bold", color=INK["text"])
                    ax.text(r.lx, r.ly, r.lab, ha="center", va="bottom", fontsize=BASE_SIZE * 0.82,
                            color=INK["text2"], fontweight="bold" if r.odd else "normal")
                else:
                    ax.plot([r.mid, r.lx], [1.42, r.ly - 0.05], color=INK["rule"], linewidth=0.8)
                    ax.text(r.lx, r.ly, r.lab1, ha="center", va="bottom", fontsize=BASE_SIZE * 0.82,
                            color=INK["text2"], fontweight="bold" if r.odd else "normal")
            if has_ev:
                for e in ev.itertuples():
                    x = float(runs.loc[runs["run"] == e.run, "start"].iloc[0]) + e.onset
                    ax.plot([x, x], [0.55, 0.35], color=INK["text"], linewidth=0.8)
            ax.set_xlim(-0.01 * total, total * 1.01)
            ax.set_ylim(0.3 if has_ev else 0.5, float(runs["ly"].max()) + (0.95 if two_line else 0.5))
            ax.set_yticks([])
            from matplotlib.ticker import FixedLocator, FuncFormatter

            ax.xaxis.set_major_locator(FixedLocator(_frame_breaks(runs, total)))
            ax.xaxis.set_major_formatter(FuncFormatter(mmss))
            ax.set_xlabel("Time (min:s)")
            ax.grid(axis="x", visible=False)
            notes = [concat_note]
            default_title = "Acquisition timeline"
        else:
            n_max = int(bl.max())
            fine = n_max <= 60
            for r in runs.itertuples():
                for s in range(1, int(r.scans) + 1):
                    shade = blue if ((s - 1) // tick_every) % 2 == 0 else blue2
                    ax.add_patch(Rectangle((s - 0.5, r.y - half), 1, 2 * half, facecolor=shade,
                                           edgecolor="white" if fine else "none", linewidth=0.8 if fine else 0))
                ax.text(r.scans + 0.5 + n_max * 0.01, r.y, r.lab, ha="left", va="center",
                        fontsize=BASE_SIZE * 0.86, color=INK["text2"], fontweight="bold" if r.odd else "normal")
            if has_ev:
                for e in ev.itertuples():
                    rr = runs[runs["run"] == e.run].iloc[0]
                    x = e.onset / rr["TR"] + 0.5
                    ax.plot([x, x], [rr["y"] - half - 0.04, rr["y"] - half - 0.16], color=INK["text"], linewidth=0.8)
            from matplotlib.ticker import MaxNLocator

            ax.set_xlim(0.5 - 0.005 * n_max, n_max + 0.5 + 0.2 * n_max)
            ticks = [v for v in MaxNLocator(5, integer=True).tick_values(1, n_max) if 1 <= v <= n_max]
            ax.set_xticks(ticks)
            ax.set_xlabel("Scan (within run)")
            ax.grid(axis="x", visible=True, color=INK["grid"])
            notes = [f"Shading alternates every {tick_every} scans."]
            default_title = "Scans per run"
        if style != "lane":
            extra = max(0, 4 - n_runs) * 0.3
            ax.set_ylim(0.5 - extra - (0.2 if has_ev else 0), n_runs + 0.5 + extra)
            ax.set_yticks(runs["y"].tolist())
            ax.set_yticklabels([f"Run {r}" for r in runs["run"]], fontweight="bold", color=INK["text"],
                               fontsize=BASE_SIZE * 0.85)
            ax.tick_params(axis="y", length=0)
        parts = [p for p in [*notes, ev_note, odd_note] if p]
        caption = resolve(caption, " ".join(parts) if parts else None)
        finish_figure(fig, resolve(title, default_title), resolve(subtitle, sub_default), caption, own=own, ax=ax)
    return fig


__all__ = ["drift_cutoff_note", "motion_kind", "plot_baseline_model", "plot_sampling_frame"]
