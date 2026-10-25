"""Shared building blocks for fmrimod figures.

These carry the idioms of the R plot overhaul (fmrihrf ``R/plot-style.R``,
fmridesign ``R/plot_theme.R``, fmrireg ``R/plot-helpers.R``) so every plot
family uses them the same way:

- a left-aligned figure header (bold title, muted subtitle) and a muted
  explanatory caption (:func:`finish`);
- runs as hairline boundaries plus "Run k" labels on a top axis, never a
  line drawn across a run boundary (:func:`run_axis`);
- time on an m:ss axis at round steps (:func:`time_axis`);
- a small, readable tick budget (:func:`budget_breaks`);
- event onsets as rug ticks or duration bars (:func:`onset_rug`,
  :func:`event_bars`);
- direct labels with a white halo instead of legends (:func:`halo_text`,
  :func:`place_labels`);
- one column-metadata table for labels, groups and runs
  (:func:`column_info`).
"""

from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .style import BASE_SIZE, INK, theme_rc

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.text import Text
    from matplotlib.ticker import Formatter
    from matplotlib.transforms import Bbox

TIMES = "×"
DOT = "·"
MINUS = "−"
NDASH = "–"
GE = "≥"
PM = "±"


def require_matplotlib() -> object:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - matplotlib is a hard dep
        raise ImportError("fmrimod plotting needs matplotlib") from exc
    return plt


def num(x: float, digits: int = 2) -> str:
    """``x`` to ``digits`` significant figures with a true minus sign."""
    if x is None or not np.isfinite(x):
        return "NA"
    if x == 0:
        return "0"
    s = f"{float(x):.{digits}g}"
    if "e" in s:
        s = f"{float(x):.{max(digits - 1, 0)}e}"
    return s.replace("-", MINUS)


def corr_text(r: float) -> str:
    """Correlation without the leading zero (``.28``, ``−.02``)."""
    if not np.isfinite(r):
        return ""
    s = f"{abs(r):.2f}".lstrip("0") if abs(r) < 1 else "1"
    if s in (".00", ""):
        return "0"
    return (MINUS if r < 0 else "") + s


def wrap(text: str | None, width: int) -> str | None:
    if not text:
        return text
    return "\n".join(textwrap.fill(p, width) for p in str(text).split("\n"))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] | None = None,
    base_size: float = BASE_SIZE,
    **subplot_kw: object,
) -> tuple[Figure, Axes | np.ndarray]:
    """``plt.subplots`` under the fmrimod theme.

    The rc context is entered here and stays in force for drawing because
    matplotlib reads most rc values when artists are created; callers draw
    inside :func:`themed` for anything created later.
    """
    plt = require_matplotlib()
    import matplotlib as mpl

    with mpl.rc_context(theme_rc(base_size)):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (7.0, 4.2), **subplot_kw)
    return fig, axes


def themed(base_size: float = BASE_SIZE) -> AbstractContextManager[None]:
    """``with themed(): ...`` draws under the fmrimod theme."""
    import matplotlib as mpl

    return mpl.rc_context(theme_rc(base_size))


def freeze_fonts(fig: Figure) -> Figure:
    """Pin the theme's font family on every text artist of ``fig``.

    A text whose family is the generic ``"sans-serif"`` resolves it from the
    rcParams in force when it is drawn, and tick labels are created at draw
    time, so a figure laid out under the theme but saved outside it would be
    measured in one font and drawn in another (clipping margin text).
    """
    from matplotlib.text import Text

    fam = theme_rc()["font.sans-serif"]
    for a in fig.axes:
        a.tick_params(axis="both", which="both", labelfontfamily=fam)
    for t in fig.findobj(Text):
        t.set_fontfamily(fam)
    return fig


def finish(
    fig: Figure,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    *,
    base_size: float = BASE_SIZE,
    layout: bool = True,
    title_width: int | None = None,
) -> Figure:
    """Lay out ``fig`` and add the header and caption.

    The header is aligned to the figure's left edge (R
    ``plot.title.position = "plot"``), so it lines up with axis labels
    rather than the panel. ``layout=False`` keeps the caller's axes
    placement and only reserves room by shrinking it.
    """
    w_in, h_in = fig.get_size_inches()
    pad = 0.10
    chars = max(int(w_in * 72 / (base_size * 0.52)), 20)
    t_chars = max(int(w_in * 72 / (base_size * 1.25 * 0.6)), 16)
    title = wrap(title, title_width or t_chars)
    subtitle = wrap(subtitle, chars)
    caption = wrap(caption, int(chars * 1.18))

    def lines(t: str | None) -> int:
        return 0 if not t else t.count("\n") + 1

    t_size, s_size, c_size = base_size * 1.25, base_size * 0.95, base_size * 0.8
    top = pad
    top += lines(title) * t_size * 1.25 / 72
    top += lines(subtitle) * s_size * 1.3 / 72
    if title or subtitle:
        top += 0.10
    bottom = (lines(caption) * c_size * 1.35 / 72 + 0.12) if caption else 0.0

    rect = (0.0, bottom / h_in, 1.0, 1.0 - top / h_in)
    freeze_fonts(fig)
    if layout:
        with themed(base_size):
            fig.tight_layout(rect=rect, pad=0.5)
    else:
        for ax in fig.axes:
            box = ax.get_position()
            scale = (rect[3] - rect[1]) / 1.0
            ax.set_position([box.x0, rect[1] + box.y0 * scale, box.width, box.height * scale])

    x = pad / w_in
    y = 1.0 - pad / h_in
    if title:
        fig.text(
            x, y, title, ha="left", va="top", fontsize=t_size, fontweight="bold",
            color=INK["text"], family=theme_rc()["font.sans-serif"],
        )
        y -= lines(title) * t_size * 1.25 / 72 / h_in
    if subtitle:
        fig.text(
            x, y, subtitle, ha="left", va="top", fontsize=s_size, color=INK["text2"],
            family=theme_rc()["font.sans-serif"], linespacing=1.25,
        )
    if caption:
        fig.text(
            x, pad / h_in * 0.6, caption, ha="left", va="bottom", fontsize=c_size,
            color=INK["muted"], family=theme_rc()["font.sans-serif"], linespacing=1.3,
        )
    return fig


# --------------------------------------------------------------------------
# Axes idioms
# --------------------------------------------------------------------------


def budget_breaks(
    lo: float, hi: float, *, stack: bool = False, floor: float = 0.0
) -> np.ndarray:
    """Evenly spaced round y breaks under a label budget (R ``.budget_breaks``).

    The first grid, fine to coarse, with at most 4 labels (5 when only that
    labels both signs; 3 or 2 on a stacked panel), at least 12% (22%
    stacked) of the range apart, and labels on both sides of zero when the
    data cross it. ``floor`` is the bottom fraction of the axis kept free
    for event marks.
    """
    lower = lo + floor * (hi - lo)
    upper = hi
    rng = upper - lower
    if not np.isfinite(rng) or rng <= 0:
        return np.array([0.0 if lo <= 0 <= hi else lo])
    crosses = lower < -0.1 * rng and upper > 0
    budget = (3 if crosses else 2) if stack else 4
    gap = 0.22 if stack else 0.12
    k = math.floor(math.log10(rng))
    steps = sorted(m * 10.0**e for m in (1, 2, 2.5, 5) for e in range(k - 2, k + 2))

    def grid(step: float) -> np.ndarray:
        b = np.arange(math.ceil(lower / step - 1e-9), math.floor(upper / step + 1e-9) + 1) * step
        b[np.abs(b) < step * 1e-9] = 0.0
        return np.round(b, 12)

    fallback = None
    for step in steps:
        b = grid(step)
        if not b.size or b.size > budget or (b.size > 1 and step < gap * rng):
            continue
        if not crosses or ((b < 0).any() and (b > 0).any()):
            return b
        if fallback is None:
            fallback = b
    if crosses and not stack:
        for step in steps:
            b = grid(step)
            if b.size <= budget + 1 and step >= gap * rng and (b < 0).any() and (b > 0).any():
                return b
    if fallback is not None:
        return fallback
    return grid(10.0 ** math.floor(math.log10(rng)))


def axis_number_formatter(ticks: Sequence[float]) -> Formatter:
    """One decimal precision for the whole axis, true minus, no ``0.0``."""
    from matplotlib.ticker import FuncFormatter

    ticks = np.asarray(ticks, dtype=float)
    decimals = 0
    for d in range(7):
        if np.allclose(np.round(ticks, d), ticks, atol=1e-12):
            decimals = d
            break
    else:
        decimals = 6

    def fmt(v: float, _pos: object = None) -> str:
        if abs(v) < 10 ** (-decimals - 1):
            return "0"
        return f"{v:.{decimals}f}".replace("-", MINUS)

    return FuncFormatter(fmt)


def budget_yticks(ax: Axes, *, stack: bool = False, floor: float = 0.0) -> np.ndarray:
    """Apply :func:`budget_breaks` and its formatter to ``ax``'s y axis."""
    lo, hi = ax.get_ylim()
    ticks = budget_breaks(lo, hi, stack=stack, floor=floor)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(axis_number_formatter(ticks))
    ax.set_ylim(lo, hi)
    return ticks


_TIME_STEPS = (5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600)


def mmss(seconds: float, _pos: object = None) -> str:
    s = int(round(float(seconds)))
    sign = MINUS if s < 0 else ""
    s = abs(s)
    return f"{sign}{s // 60}:{s % 60:02d}"


def time_axis(ax: Axes, t_min: float, t_max: float, *, target: int = 6, label: str | None = "Time (min:s)") -> None:
    """m:ss labels at round steps, about ``target`` per axis."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    span = max(t_max - t_min, 1e-9)
    step = next((s for s in _TIME_STEPS if span / s <= target), _TIME_STEPS[-1])
    ticks = np.arange(math.ceil(t_min / step) * step, t_max + 1e-9, step)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(mmss))
    ax.set_xlim(t_min, t_max)
    if label is not None:
        ax.set_xlabel(label)


def run_axis(
    ax: Axes,
    bounds: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    top: bool = True,
    show_labels: bool | None = None,
) -> None:
    """Hairlines at interior run boundaries and "Run k" labels on top.

    ``bounds`` are the run edges in data coordinates (``n_runs + 1``
    values). Labels appear only for two or more runs, as in R.
    """
    bounds = list(bounds)
    n = len(bounds) - 1
    for b in bounds[1:-1]:
        ax.axvline(b, color=INK["zero"], linewidth=0.6, zorder=1.5)
    if show_labels is None:
        show_labels = n >= 2
    if top and show_labels and n >= 1:
        mids = [(bounds[i] + bounds[i + 1]) / 2 for i in range(n)]
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(mids)
        sec.set_xticklabels(list(labels) if labels is not None else [f"Run {i + 1}" for i in range(n)])
        sec.tick_params(length=0, labelcolor=INK["text2"], labelsize=BASE_SIZE * 0.82, pad=2)
        for spine in sec.spines.values():
            spine.set_visible(False)


def zero_line(ax: Axes, *, colour: str | None = None, lw: float = 0.6) -> None:
    ax.axhline(0.0, color=colour or INK["zero"], linewidth=lw, zorder=1.4)


def onset_rug(
    ax: Axes,
    onsets: Sequence[float],
    *,
    colour: str | None = None,
    length: float = 0.06,
    lw: float = 0.9,
    alpha: float = 0.9,
    y0: float = 0.0,
) -> None:
    """Vertical ticks at ``onsets`` along the bottom ``length`` of the axes."""
    trans = ax.get_xaxis_transform()
    for t in onsets:
        ax.plot([t, t], [y0, y0 + length], transform=trans, color=colour or INK["muted"],
                linewidth=lw, alpha=alpha, solid_capstyle="butt", clip_on=False, zorder=3)


def event_bars(
    ax: Axes,
    onsets: Sequence[float],
    durations: Sequence[float] | float,
    *,
    y0: float,
    y1: float,
    colour: str | None = None,
    alpha: float = 0.8,
    min_width: float | None = None,
) -> None:
    """Event marks as bars from onset to onset + duration (R fmrihrf).

    Point events get ``min_width`` (default 0.4% of the x range) so they stay
    visible.
    """
    from matplotlib.patches import Rectangle

    onsets = np.asarray(onsets, dtype=float)
    durs = np.broadcast_to(np.asarray(durations, dtype=float), onsets.shape)
    if min_width is None:
        lo, hi = ax.get_xlim()
        min_width = 0.004 * (hi - lo)
    for t, d in zip(onsets, durs, strict=False):
        ax.add_patch(Rectangle((t, y0), max(d, min_width), y1 - y0, facecolor=colour or INK["text2"],
                               edgecolor="none", alpha=alpha, zorder=2.5))


def halo_text(ax: Axes, x: float, y: float, text: str, *, colour: str, size: float = 8.8,
              weight: str = "bold", ha: str = "center", va: str = "bottom", **kw: object) -> Text:
    """Text with a white halo so it reads over lines and grids."""
    import matplotlib.patheffects as pe

    return ax.text(x, y, text, color=colour, fontsize=size, fontweight=weight, ha=ha, va=va,
                   path_effects=[pe.withStroke(linewidth=2.6, foreground="white")], zorder=6, **kw)


def place_labels(
    ax: Axes,
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    texts: Sequence[str],
    colours: Sequence[str],
    *,
    size: float = 8.8,
) -> None:
    """Label each curve near its peak, nudging labels apart.

    A greedy version of fmrireg's ``.place_curve_labels``: start at the
    peak (centre of a plateau), try small lifts and left/right shifts, and
    keep the first spot that clears earlier labels.
    """
    fig = ax.figure
    fig.canvas.draw_idle()
    placed: list[Bbox] = []
    y_lo, y_hi = ax.get_ylim()
    x_lo, x_hi = ax.get_xlim()
    dy = 0.045 * (y_hi - y_lo)
    dx = 0.05 * (x_hi - x_lo)
    renderer = fig.canvas.get_renderer()
    for (x, y), text, colour in zip(curves, texts, colours, strict=False):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).any():
            continue
        peak = np.nanmax(np.abs(y))
        idx = np.flatnonzero(np.abs(y) >= peak - 1e-12 * max(peak, 1e-300))
        i = int(idx[len(idx) // 2])
        up = y[i] >= 0
        base_x, base_y = float(x[i]), float(y[i])
        chosen = None
        for push in (0.0, 1.0, -1.0, 2.0, -2.0):
            for lift in (0.0, 1.0, 2.0, 3.0):
                tx = base_x + push * dx
                ty = base_y + (lift * dy if up else -lift * dy)
                art = halo_text(ax, tx, ty, text, colour=colour, size=size,
                                va="bottom" if up else "top")
                bb = art.get_window_extent(renderer).expanded(1.05, 1.1)
                if all(not bb.overlaps(p) for p in placed):
                    chosen = bb
                    break
                art.remove()
            if chosen is not None:
                break
        if chosen is None:
            art = halo_text(ax, base_x, base_y, text, colour=colour, size=size,
                            va="bottom" if up else "top")
            chosen = art.get_window_extent(renderer)
        placed.append(chosen)


def row_label(ax: Axes, text: str, *, size: float | None = None, colour: str | None = None) -> None:
    """Horizontal, right-aligned row label left of a small-multiple panel."""
    ax.set_ylabel(text, rotation=0, ha="right", va="center", labelpad=6,
                  fontsize=size or BASE_SIZE * 0.85, color=colour or INK["text"])


def legend_bottom(ax: Axes, handles: Sequence[object] | None = None, labels: Sequence[str] | None = None,
                  ncol: int | None = None) -> Legend | None:
    h, lab = ax.get_legend_handles_labels() if handles is None else (handles, labels)
    if not h:
        return None
    if ncol is None:
        width = sum(len(str(t)) + 4 for t in lab)
        ncol = len(lab) if width <= 60 else max(1, 60 // max(len(str(t)) + 4 for t in lab))
    return ax.legend(h, lab, loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=ncol,
                     frameon=False, handlelength=1.4, columnspacing=1.2)


# --------------------------------------------------------------------------
# Column metadata
# --------------------------------------------------------------------------


_BASIS_NAMES = {
    ("spmg2", 1): "canonical",
    ("spmg2", 2): "temporal deriv.",
    ("spmg3", 1): "canonical",
    ("spmg3", 2): "temporal deriv.",
    ("spmg3", 3): "dispersion deriv.",
}


def level_label(level: str | None, term: str | None = None) -> str | None:
    """``cond.face_load.high`` to ``face × high``; ``cond.face`` to ``face``."""
    if level is None:
        return None
    factors = (term or "").split(":") if term else []
    if factors and all(f for f in factors):
        parts = []
        rest = level
        for i, f in enumerate(factors):
            prefix = f + "."
            if not rest.startswith(prefix):
                parts = []
                break
            rest = rest[len(prefix):]
            if i + 1 < len(factors):
                nxt = "_" + factors[i + 1] + "."
                j = rest.find(nxt)
                if j < 0:
                    parts = []
                    break
                parts.append(rest[:j])
                rest = rest[j + 1:]
            else:
                parts.append(rest)
        if parts:
            return f" {TIMES} ".join(parts)
    m = re.match(r"^[^.]+\.(.+)$", level)
    return m.group(1) if m else level


def basis_label(basis_name: str | None, ix: int | None, total: int | None) -> str | None:
    if ix is None or total is None or pd.isna(ix) or pd.isna(total) or total <= 1:
        return None
    key = (str(basis_name or "").lower(), int(ix))
    if key in _BASIS_NAMES:
        return _BASIS_NAMES[key]
    if str(basis_name or "").lower() in ("fir", "tent"):
        return f"lag {int(ix) - 1}"
    return f"b{int(ix)}"


def _baseline_label(name: str, run: object) -> tuple[str, str]:
    stem = re.sub(r"^base_", "", name)
    stem = re.sub(r"_block_\d+$", "", stem)
    stem = re.sub(r"^constant_?\d*$", "intercept", stem)
    group = re.sub(r"\d+$", "", stem) or stem
    label = stem if run is None or (isinstance(run, float) and np.isnan(run)) else f"{stem} {DOT} run {int(run)}"
    return group, label


def column_info(model: object) -> pd.DataFrame:
    """One row per design column with labels and groupings for plotting.

    Accepts an ``EventModel``, ``BaselineModel``, ``FmriModel`` or a fit
    (anything with ``.model``). Columns: ``col`` (0-based), ``name``,
    ``kind`` (``event``/``baseline``), ``role``, ``term``, ``term_label``,
    ``condition`` (pretty level, ``face × high``), ``basis_ix``,
    ``basis_total``, ``basis_label``, ``run``, ``label`` (short, unique)
    and ``full_label`` (label plus basis/run qualifier).
    """
    from ..design_colmap import design_colmap

    parts: list[pd.DataFrame] = []
    if hasattr(model, "betas") and hasattr(model, "model"):
        model = model.model
    ev = getattr(model, "event_model", None)
    bl = getattr(model, "baseline_model", None)
    if ev is None and bl is None:
        if getattr(model, "sampling_frame", None) is not None and hasattr(model, "terms") and "Baseline" in type(model).__name__:
            bl = model
        else:
            ev = model
    typed: list[object] = []
    if ev is not None:
        from ..design.columns import _event_columns

        parts.append(design_colmap(ev).assign(kind="event"))
        typed = list(_event_columns(ev))
    if bl is not None:
        parts.append(design_colmap(bl).assign(kind="baseline"))
    cm = pd.concat(parts, ignore_index=True)
    rows = []
    for i, r in enumerate(cm.to_dict("records")):
        name = str(r["name"])
        if r["kind"] == "event":
            dc = typed[i] if i < len(typed) else None
            term = (dc.term if dc is not None else None) or r.get("term_tag") or None
            cond = level_label(dc.level if dc is not None else r.get("condition"), term)
            bl_lab = basis_label(r.get("basis_name"), r.get("basis_ix"), r.get("basis_total"))
            label = cond or name
            full = label + (f" {DOT} {bl_lab}" if bl_lab else "")
            term_label = (term or "").replace(":", f" {TIMES} ") or None
            run = None
        else:
            run = r.get("run")
            term, label = _baseline_label(name, run)
            term_label, cond, bl_lab, full = term, None, None, label
        rows.append(
            {
                "col": i,
                "name": name,
                "kind": r["kind"],
                "role": r.get("role"),
                "term": term,
                "term_label": term_label,
                "condition": cond,
                "basis_ix": r.get("basis_ix"),
                "basis_total": r.get("basis_total"),
                "basis_label": bl_lab,
                "run": run,
                "label": label,
                "full_label": full,
            }
        )
    out = pd.DataFrame(rows)
    # ``label`` names a condition (shared by its basis columns); it is
    # prefixed with the term only when two different (term, condition)
    # pairs would read the same. ``full_label`` names one column.
    owners = out.groupby("label")[["term", "condition"]].apply(
        lambda g: len({(t, c) for t, c in zip(g["term"], g["condition"], strict=False)})
    )
    clash = out["label"].map(owners) > 1
    out.loc[clash, "label"] = [
        f"{t}: {lab}" if t else lab for t, lab in zip(out.loc[clash, "term_label"], out.loc[clash, "label"], strict=False)
    ]
    out.loc[clash, "full_label"] = [
        f"{t}: {lab}" if t else lab
        for t, lab in zip(out.loc[clash, "term_label"], out.loc[clash, "full_label"], strict=False)
    ]
    seen: dict[str, int] = {}
    vals = []
    for v in out["full_label"]:
        k = seen.get(v, 0)
        vals.append(v if k == 0 else f"{v} #{k + 1}")
        seen[v] = k + 1
    out["full_label"] = vals
    return out


__all__ = [
    "DOT",
    "MINUS",
    "TIMES",
    "axis_number_formatter",
    "budget_breaks",
    "budget_yticks",
    "column_info",
    "corr_text",
    "event_bars",
    "finish",
    "freeze_fonts",
    "halo_text",
    "level_label",
    "mmss",
    "new_figure",
    "num",
    "onset_rug",
    "place_labels",
    "row_label",
    "run_axis",
    "themed",
    "time_axis",
    "zero_line",
]
