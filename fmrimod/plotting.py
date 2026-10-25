"""HRF and regressor plots (the R fmrihrf plotting family).

One engine draws every figure here, ported from fmrihrf's ``.gg_series``
(``R/plot-style.R``): curves on a shared time axis, either overlaid in one
panel or stacked one panel per curve, with a zero rule, event marks as bars
in a band under the curves, sample points, a dashed grey reference curve,
direct labels at the peaks of a single basis set, and a small budget of
round y ticks so short panels stay legible.

Public functions:

- :func:`plot_hrfs` and :func:`plot_regressors` compare several HRFs or
  regressors (R ``plot_hrfs`` / ``plot_regressors``);
- :func:`plot_hrf` and :func:`plot_regressor` draw one object (R
  ``plot.HRF`` / ``plot.Reg``), also reachable as :meth:`HRF.plot
  <fmrimod.hrf.core.HRF.plot>` and :meth:`Regressor.plot
  <fmrimod.regressor.core.Regressor.plot>`;
- :func:`hrf_curves` and :func:`regressor_curves` return the long data the
  plots draw (``time``, ``series``, ``response``), the frame R returns
  alongside its plot.

Every plot function returns a matplotlib :class:`~matplotlib.figure.Figure`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .hrf.core import HRF
from .regressor.core import Regressor, RegressorSet

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text
else:
    Axes = Any
    Figure = Any

__all__ = [
    "hrf_curves",
    "plot_hrf",
    "plot_hrfs",
    "plot_regressor",
    "plot_regressors",
    "regressor_curves",
]

BasisMode = Literal["all", "first"]
Layout = Literal["overlay", "stack"]
PaletteKind = Literal["auto", "categorical", "ordered"]
Scales = Literal["free_y", "fixed"]
OnsetMode = bool | Literal["first"] | None
HrfInput = HRF | Sequence[HRF] | Mapping[str, HRF]
RegInput = Regressor | RegressorSet | Sequence[Regressor] | Mapping[str, Regressor]
_T = TypeVar("_T")

# Event bars drawn in a band below the curves, as fractions of the data range
# below its lowest point (R .gg_series): overlay 0.16-0.06, stacked 0.22-0.06.
_MARK_TOP = 0.06
_MARK_BOTTOM = {"overlay": 0.16, "stack": 0.22}
# Fraction of the axis the event band takes, where no tick label may go.
_MARK_FLOOR = {"overlay": 0.13, "stack": 0.19}
# ggplot2 scale_linetype_discrete() dash patterns, for many-curve overlays.
_DASHES: tuple[str | tuple[int, tuple[int, ...]], ...] = (
    "solid",
    (0, (2, 2)),
    (0, (4, 2)),
    (0, (4, 4)),
    (0, (1, 3)),
    (0, (1, 3, 4, 3)),
    (0, (7, 3)),
    (0, (2, 2, 6, 2)),
)
_LINE_WIDTH = 1.7


# --------------------------------------------------------------------------
# Data: one long frame per figure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Curves:
    """Evaluated curves plus what the renderer needs to know about them."""

    time: NDArray[np.float64]
    resp: NDArray[np.float64]  # (len(time), n_series)
    labels: list[str]
    owner: list[int]  # index of the HRF/regressor each curve comes from
    nb: list[int]  # curves contributed by each object
    explicit_labels: bool

    @property
    def single_set(self) -> bool:
        return len(self.nb) == 1 and self.nb[0] > 1

    def frame(self) -> pd.DataFrame:
        return _long_frame(self.time, self.resp, self.labels)


def _long_frame(time: NDArray[np.float64], resp: NDArray[np.float64], labels: Sequence[str]) -> pd.DataFrame:
    n = len(labels)
    df: pd.DataFrame = pd.DataFrame(
        {
            "time": np.tile(time, n),
            "series": pd.Categorical(np.repeat(list(labels), len(time)), categories=list(labels)),
            "response": resp.T.reshape(-1) if n else np.array([], dtype=float),
        }
    )
    return df


def _collect(objs: Sequence[object], cls: type[_T], noun: str) -> tuple[list[_T], list[str] | None]:
    """Objects given as separate arguments, one list, or one name->object mapping."""
    names: list[str] | None = None
    if len(objs) == 1 and isinstance(objs[0], Mapping):
        names = [str(k) for k in objs[0]]
        items = list(objs[0].values())
    elif len(objs) == 1 and isinstance(objs[0], (list, tuple)):
        items = list(objs[0])
    else:
        items = list(objs)
    if not items:
        raise ValueError(f"At least one {noun} must be provided.")
    for i, obj in enumerate(items):
        if not isinstance(obj, cls):
            raise TypeError(
                f"All arguments must be {noun} objects; argument {i + 1} is a {type(obj).__name__}."
            )
    return items, names


def _make_unique(labels: Sequence[str]) -> list[str]:
    """R ``make.unique(x, sep = " ")``: later duplicates get `` 1``, `` 2``."""
    out: list[str] = []
    seen = set(labels)
    counts: dict[str, int] = {}
    first: set[str] = set()
    for lab in labels:
        if lab not in first:
            first.add(lab)
            out.append(lab)
            continue
        k = counts.get(lab, 0)
        while True:
            k += 1
            cand = f"{lab} {k}"
            if cand not in seen:
                break
        counts[lab] = k
        seen.add(cand)
        out.append(cand)
    return out


def _basis_names(basis_name: str, nb: int) -> list[str]:
    """Names of the columns of one basis set.

    SPMG2/SPMG3 columns read canonical / temporal deriv. / dispersion deriv.
    (the naming used across fmrimod figures); other sets ``B1``, ``B2``...
    as in R.
    """
    from .visualization._helpers import _BASIS_NAMES

    key = basis_name.lower()
    names = [_BASIS_NAMES.get((key, k)) for k in range(1, nb + 1)]
    if all(names):
        return [str(n) for n in names]
    return [f"B{k}" for k in range(1, nb + 1)]


def _series_labels(
    nb: Sequence[int],
    labels: Sequence[str] | None,
    defaults: Sequence[str],
    basis_names: Sequence[str],
    what: str,
) -> list[str]:
    """One label per curve (R ``.series_labels``).

    ``labels`` may name every curve or every object. The basis columns of an
    object are labelled ``<object> B<k>``, or just ``B<k>`` when a single
    basis set is shown.
    """
    if labels is not None and isinstance(labels, str):
        labels = [labels]
    if labels is not None and len(labels) == sum(nb):
        out = [str(x) for x in labels]
    else:
        if labels is not None and len(labels) != len(nb):
            raise ValueError(f"Length of 'labels' must match the number of {what} or of curves.")
        base = [str(x) for x in (labels if labels is not None else defaults)]
        single = len(nb) == 1 and nb[0] > 1
        out = []
        for i, k in enumerate(nb):
            if k == 1:
                out.append(base[i])
                continue
            cols = _basis_names(basis_names[i], k)
            if single:
                out.extend(cols)
            else:
                sep = " " if cols[0] == "B1" else " · "
                out.extend(f"{base[i]}{sep}{c}" for c in cols)
    return _make_unique(out) if len(set(out)) < len(out) else out


def _normalise(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each column divided by its largest absolute value (R ``.normalise_result``)."""
    peak = np.nanmax(np.abs(y), axis=0) if y.size else np.ones(y.shape[1])
    peak = np.where(np.isfinite(peak) & (peak > 0), peak, 1.0)
    return y / peak


def _as_matrix(y: object, n: int) -> NDArray[np.float64]:
    arr = np.asarray(y.toarray() if hasattr(y, "toarray") else y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    return arr.reshape(n, -1)


def _eval_hrf(h: HRF, time: NDArray[np.float64], normalize: bool, basis: BasisMode) -> NDArray[np.float64]:
    y = _as_matrix(h(time), len(time))
    if basis == "first":
        y = y[:, :1]
    return _normalise(y) if normalize else y


def _hrf_curve_set(
    hrfs: Sequence[HrfInput],
    *,
    time: ArrayLike | None,
    normalize: bool,
    labels: Sequence[str] | None,
    basis: BasisMode,
) -> _Curves:
    _check_choice("basis", basis, ("all", "first"))
    items, names = _collect(hrfs, HRF, "HRF")  # type: ignore[type-abstract]
    if time is None:
        span = max(float(getattr(h, "span", 24.0) or 24.0) for h in items)
        time = np.round(np.arange(0.0, span + 1e-9, 0.1), 10)
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    if t.size == 0:
        raise ValueError("time must contain at least one point")
    mats = [_eval_hrf(h, t, normalize, basis) for h in items]
    nb = [m.shape[1] for m in mats]
    hrf_names = [str(getattr(h, "name", None) or "HRF") for h in items]
    defaults = names or (
        [f"{n}_{i + 1}" for i, n in enumerate(hrf_names)] if len(set(hrf_names)) < len(hrf_names) else hrf_names
    )
    series = _series_labels(nb, labels, defaults, hrf_names, "HRFs")
    return _Curves(
        time=t,
        resp=np.hstack(mats),
        labels=series,
        owner=[i for i, k in enumerate(nb) for _ in range(k)],
        nb=nb,
        explicit_labels=labels is not None,
    )


def hrf_curves(
    *hrfs: HrfInput,
    time: ArrayLike | None = None,
    normalize: bool = False,
    labels: Sequence[str] | None = None,
    basis: BasisMode = "all",
) -> pd.DataFrame:
    """Long-form curves drawn by :func:`plot_hrfs`.

    Args:
        *hrfs: HRF objects, a list of them, or one ``{label: hrf}`` mapping.
        time: Time points in seconds; defaults to 0 to the longest span in
            0.1 s steps.
        normalize: Divide each curve by its largest absolute value.
        labels: One label per HRF or per curve (after basis expansion).
        basis: ``"all"`` expands a multi-basis HRF to one curve per basis
            function; ``"first"`` keeps only its first column.

    Returns:
        DataFrame with columns ``time``, ``series`` (categorical, in plot
        order) and ``response``.
    """
    return _hrf_curve_set(hrfs, time=time, normalize=normalize, labels=labels, basis=basis).frame()


def _regs_from(regs: Sequence[RegInput]) -> tuple[list[Regressor], list[str] | None]:
    if len(regs) == 1 and isinstance(regs[0], RegressorSet):
        rs = regs[0]
        return list(rs.regressors), [str(lv) for lv in rs.levels]
    return _collect(regs, Regressor, "Regressor")


def _reg_grid(regs: Sequence[Regressor]) -> NDArray[np.float64]:
    """0.25 s steps from 0 to the last onset plus span (R ``.reg_plot_grid``)."""
    ends = [
        float(np.max(r.onsets)) + float(r.span or 0.0) if len(r.onsets) else float(r.span or 0.0)
        for r in regs
    ]
    return np.round(np.arange(0.0, max(ends) + 1e-9, 0.25), 10)


def _plot_precision(precision: float | None, grid: NDArray[np.float64]) -> float:
    """Grid spacing capped at 0.33 s and floored at 0.01 s (R ``.plot_precision``)."""
    if precision is not None:
        return float(precision)
    u = np.unique(grid)
    if u.size < 2:
        return 0.33
    step = float(f"{float(np.min(np.diff(u))):.8g}")
    return min(0.33, max(0.01, step))


def _eval_reg(r: Regressor, grid: NDArray[np.float64], precision: float, basis: BasisMode) -> NDArray[np.float64]:
    y = _as_matrix(r.evaluate(grid, precision=precision), len(grid))
    return y[:, :1] if basis == "first" else y


def _hrf_name(r: Regressor) -> str:
    if getattr(r, "hrf_is_list", False):
        return "trial-varying HRF"
    return str(getattr(r.hrf, "name", None) or "custom HRF")


def _reg_curve_set(
    regs: Sequence[RegInput],
    *,
    grid: ArrayLike | None,
    precision: float | None,
    labels: Sequence[str] | None,
    basis: BasisMode,
) -> tuple[_Curves, list[Regressor], float]:
    _check_choice("basis", basis, ("all", "first"))
    items, names = _regs_from(regs)
    if not items:
        raise ValueError("At least one Regressor must be provided.")
    g = _reg_grid(items) if grid is None else np.asarray(grid, dtype=np.float64).reshape(-1)
    if g.size == 0:
        raise ValueError("grid must contain at least one point")
    prec = _plot_precision(precision, g)
    mats = [_eval_reg(r, g, prec, basis) for r in items]
    nb = [m.shape[1] for m in mats]
    defaults = names or [f"Regressor {i + 1}" for i in range(len(items))]
    basis_names = [_hrf_name(r) for r in items]
    series = _series_labels(nb, labels, defaults, basis_names, "regressors")
    curves = _Curves(
        time=g,
        resp=np.hstack(mats),
        labels=series,
        owner=[i for i, k in enumerate(nb) for _ in range(k)],
        nb=nb,
        explicit_labels=labels is not None,
    )
    return curves, items, prec


def regressor_curves(
    *regs: RegInput,
    grid: ArrayLike | None = None,
    precision: float | None = None,
    labels: Sequence[str] | None = None,
    basis: BasisMode = "all",
) -> pd.DataFrame:
    """Long-form curves drawn by :func:`plot_regressors`.

    Args:
        *regs: Regressor objects, a list of them, a ``{label: regressor}``
            mapping, or one :class:`~fmrimod.regressor.core.RegressorSet`
            (one curve per condition).
        grid: Evaluation times; defaults to 0 to the last onset plus span in
            0.25 s steps.
        precision: HRF sampling precision; defaults to the grid spacing,
            capped at 0.33 s and floored at 0.01 s.
        labels: One label per regressor or per curve.
        basis: ``"all"`` or ``"first"`` basis column of multi-basis
            regressors.

    Returns:
        DataFrame with columns ``time``, ``series`` and ``response``.
    """
    curves, _, _ = _reg_curve_set(regs, grid=grid, precision=precision, labels=labels, basis=basis)
    return curves.frame()


def _event_marks(
    curves: _Curves, items: Sequence[Regressor], show_onsets: OnsetMode, layout: Layout
) -> tuple[pd.DataFrame | None, bool]:
    """Event rows ``(onset, duration, series)`` and whether they colour by series.

    Every curve gets its own regressor's events, or with ``"first"`` only the
    curves of the first regressor (R ``plot_regressors``).
    """
    if show_onsets is None:
        show_onsets = True if layout == "stack" else "first"
    if show_onsets is False:
        return None, False
    first_only = isinstance(show_onsets, str) and show_onsets == "first"
    rows = []
    for s, owner in enumerate(curves.owner):
        if first_only and owner != 0:
            continue
        r = items[owner]
        if not len(r.onsets):
            continue
        dur = np.resize(np.asarray(r.duration, dtype=float).reshape(-1), len(r.onsets))
        if not np.size(r.duration):
            dur = np.zeros(len(r.onsets))
        rows.append(pd.DataFrame({"onset": np.asarray(r.onsets, dtype=float), "duration": dur,
                                  "series": curves.labels[s]}))
    if not rows:
        return None, not first_only
    marks = pd.concat(rows, ignore_index=True)
    lo, hi = float(np.min(curves.time)), float(np.max(curves.time))
    marks = marks[(marks["onset"] >= lo) & (marks["onset"] <= hi)].reset_index(drop=True)
    return (marks if len(marks) else None), not first_only


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _check_choice(name: str, value: object, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, choices))}; got {value!r}")


def _series_palette(n: int, palette: PaletteKind, ordered_hint: bool) -> tuple[str, list[str]]:
    """Palette kind and colours (R ``.series_palette``): ordered for a long
    basis set or more than six curves, categorical otherwise."""
    from .visualization.style import palette as make_palette

    _check_choice("palette", palette, ("auto", "categorical", "ordered"))
    kind = palette
    if kind == "auto":
        kind = "ordered" if (ordered_hint or n > 6) else "categorical"
    return kind, make_palette(n, kind)


def _budget_breaks(lo: float, hi: float, *, stack: bool = False, floor: float = 0.0) -> NDArray[np.float64]:
    from .visualization._helpers import budget_breaks

    return budget_breaks(lo, hi, stack=stack, floor=floor)


def _set_budget_yticks(ax: Axes, *, stack: bool, floor: float) -> None:
    from .visualization._helpers import axis_number_formatter

    lo, hi = ax.get_ylim()
    ticks = _budget_breaks(lo, hi, stack=stack, floor=floor)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(axis_number_formatter(ticks.tolist()))
    ax.set_ylim(lo, hi)


def _legend_ncol(labels: Sequence[str], width_in: float) -> int:
    """Legend columns that fit one row of the figure (R ``.legend_ncol``).

    R budgets 36 characters for a phone-width figure; here the budget is
    the characters of legend text a row of ``width_in`` inches holds.
    """
    budget = max(int(width_in * 72 / (8.2 * 0.62) * 0.7), 20)
    width = [len(s) + 4 for s in labels]
    if sum(width) <= budget:
        return len(labels)
    fit = max(1, min(len(labels), budget // max(width)))
    # Balance the rows, so a wrapped legend has no orphaned last entry.
    return math.ceil(len(labels) / math.ceil(len(labels) / fit))


def _text_colour(colour: str) -> str:
    """A series colour dark enough to read as text on white: light ramp
    stops (the start of the ordered palette) are shaded towards the ink."""
    from matplotlib.colors import to_rgb

    from .visualization.style import shade

    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(colour)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    amount = min(max((lum - 0.27) / 0.4, 0.0), 0.45)
    return shade(colour, amount) if amount > 0 else colour


def _byrow(handles: list[_T], labels: list[str], ncol: int) -> tuple[list[_T], list[str]]:
    """Reorder legend entries so matplotlib's column-major fill reads by row
    (ggplot ``guide_legend(byrow = TRUE)``). Matplotlib gives the first
    ``n % ncol`` columns the extra entry, which is where a by-row fill puts
    the last, partial row."""
    n = len(labels)
    nrow = math.ceil(n / max(ncol, 1))
    order = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < n]
    return [handles[k] for k in order], [labels[k] for k in order]


def _peak_labels(ax: Axes, curves: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
                 texts: Sequence[str], colours: Sequence[str], *, size: float = 8.6) -> list[Text]:
    """Bold labels at curve peaks, kept off other labels and curves.

    R puts each label just above the centre of its curve's (positive) peak.
    Close curves make those labels collide, so each label scores spots on a
    grid around its peak - curve points under the label cost far more than
    distance from the peak - and takes the cheapest spot that stays inside
    the panel and clear of earlier labels.
    """
    from matplotlib.transforms import Bbox

    from .visualization._helpers import halo_text

    fig = cast("Figure", ax.figure)
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    to_px = ax.transData.transform
    pts = np.vstack([to_px(np.column_stack([x, y])[np.isfinite(y)]) for x, y in curves])
    axbox = ax.get_window_extent(renderer)
    px_per_pt = fig.dpi / 72.0
    placed: list[Bbox] = []
    artists = []
    for (x, y), text, colour in zip(curves, texts, colours, strict=False):
        if not np.isfinite(y).any():
            continue
        y_lo, y_hi = ax.get_ylim()
        tol = 1e-6 * (y_hi - y_lo)
        # Anchors: the centre of the peak (plateaus such as FIR bins get a
        # centred label), and the trough of a curve that dips well below 0.
        anchors = []
        top = np.flatnonzero(y >= np.nanmax(y) - tol)
        anchors.append((int(top[(len(top) - 1) // 2]), 1.0, 0.0))
        if np.nanmin(y) < -0.25 * max(np.nanmax(y), 0.0):
            bot = np.flatnonzero(y <= np.nanmin(y) + tol)
            anchors.append((int(bot[(len(bot) - 1) // 2]), -1.0, 30.0))
        probe = halo_text(ax, float(x[0]), float(y[0]), text, colour=colour, size=size,
                          ha="center", va="bottom")
        ext = probe.get_window_extent(renderer)
        probe.remove()
        w, h = ext.width * 1.06, ext.height * 1.1
        gap = 3.0 * px_per_pt
        best = None
        for i, side, cost in anchors:
            peak = to_px((float(x[i]), float(y[i])))
            for dy in np.arange(0.0, 6.01, 0.5) * h * 0.5:
                for dx in np.arange(-3.0, 3.01, 0.25) * w * 0.5:
                    cx = peak[0] + dx
                    by = peak[1] + gap + dy if side > 0 else peak[1] - gap - dy - h
                    bb = Bbox.from_bounds(cx - w / 2, by, w, h)
                    if bb.x0 < axbox.x0 or bb.x1 > axbox.x1 or bb.y1 > axbox.y1 or bb.y0 < axbox.y0:
                        continue
                    if any(bb.overlaps(p) for p in placed):
                        continue
                    inside = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1)
                              & (pts[:, 1] >= bb.y0 - 1) & (pts[:, 1] <= bb.y1 + 1))
                    score = cost + 40.0 * int(inside.sum()) + math.hypot(dx, 1.6 * dy)
                    if best is None or score < best[0]:
                        best = (score, cx, by, bb)
        if best is None:
            i = anchors[0][0]
            peak = to_px((float(x[i]), float(y[i])))
            best = (0.0, peak[0], peak[1] + gap, Bbox.from_bounds(peak[0] - w / 2, peak[1] + gap, w, h))
        bx, by = ax.transData.inverted().transform((best[1], best[2] + (h - ext.height) / 2))
        art = halo_text(ax, float(bx), float(by), text, colour=_text_colour(colour), size=size, ha="center",
                        va="bottom")
        artists.append(art)
        placed.append(best[3])
    return artists


@dataclass
class _Spec:
    """Everything the series engine draws (R ``.gg_series`` arguments)."""

    curves: _Curves
    colours: list[str]
    kind: str
    title: str | None
    subtitle: str | None
    caption: str | None
    ylab: str
    xlab: str = "Time (s)"
    layout: Layout = "overlay"
    scales: Scales = "free_y"
    marks: pd.DataFrame | None = None
    marks_by_series: bool = False
    mark_alpha: float = 0.8
    samples: _Curves | None = None
    reference: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    direct_labels: bool = False
    peak: bool = False
    figsize: tuple[float, float] | None = None


def _yrange(values: Sequence[NDArray[np.float64] | None]) -> tuple[float, float]:
    finite = [v[np.isfinite(v)] for v in values if v is not None and np.size(v)]
    allv = np.concatenate(finite + [np.array([0.0])])
    return float(np.min(allv)), float(np.max(allv))


def _draw_marks(ax: Axes, marks: pd.DataFrame, x_range: float, y0: float, y1: float,
                colour: str, alpha: float) -> None:
    from matplotlib.patches import Rectangle

    min_w = 0.004 * x_range
    for onset, dur in zip(marks["onset"], marks["duration"], strict=False):
        ax.add_patch(Rectangle((onset, y0), max(float(dur), min_w), y1 - y0, facecolor=colour,
                               edgecolor="none", alpha=alpha, zorder=2.5, linewidth=0))


def _render(spec: _Spec, ax: Axes | None = None) -> Figure:
    from .visualization._helpers import finish, new_figure, themed, zero_line
    from .visualization.style import INK

    c = spec.curves
    n = len(c.labels)
    stack = spec.layout == "stack"
    if stack and ax is not None:
        raise ValueError("ax= draws a single panel; use layout='overlay' or omit ax")
    colour_of = dict(zip(c.labels, spec.colours, strict=False))
    linestyles = spec.kind == "categorical" and n > 4 and not stack
    x_lo, x_hi = float(np.nanmin(c.time)), float(np.nanmax(c.time))
    x_range = (x_hi - x_lo) or 1.0
    ref_y = spec.reference[1] if spec.reference is not None else None
    has_marks = spec.marks is not None and len(spec.marks) > 0

    with themed():
        if ax is not None:
            fig, axes = cast("Figure", ax.figure), [ax]
        else:
            if stack:
                per = min(1.25, max(0.9, 7.0 / max(n, 1)))
                h = 1.35 + per * n + (0.25 if spec.caption else 0.0)
                size = spec.figsize or (7.0, max(3.0, h))
            else:
                h = 4.2 + (0.0 if (spec.direct_labels or n == 1) else 0.25 * math.ceil(
                    n / _legend_ncol(c.labels, 7.0)))
                size = spec.figsize or (7.0, h)
            fig, axs = new_figure(n if stack else 1, 1, figsize=size, sharex=True,
                                  sharey=stack and spec.scales == "fixed")
            axes = list(np.atleast_1d(axs))

        panels = [list(range(n))] if not stack else [[j] for j in range(n)]
        shared = _yrange([c.resp.reshape(-1), ref_y])
        for ax_i, idx in zip(axes, panels, strict=False):
            if stack and spec.scales != "fixed":
                lo, hi = _yrange([c.resp[:, idx].reshape(-1), ref_y])
            else:
                lo, hi = shared
            span_y = (hi - lo) if np.isfinite(hi - lo) and hi > lo else 1.0
            zero_line(ax_i, colour=INK["zero"], lw=0.6)
            bottom = lo
            if has_marks:
                assert spec.marks is not None
                y0 = lo - _MARK_BOTTOM[spec.layout] * span_y
                y1 = lo - _MARK_TOP * span_y
                bottom = y0
                names = [c.labels[j] for j in idx]
                sub = spec.marks[spec.marks["series"].isin(names)]
                if not stack and spec.marks_by_series:
                    for name in names:
                        _draw_marks(ax_i, sub[sub["series"] == name], x_range, y0, y1,
                                    colour_of[name], spec.mark_alpha)
                elif len(sub):
                    dedup = sub.drop_duplicates(["onset", "duration"])
                    _draw_marks(ax_i, dedup, x_range, y0, y1, INK["text2"], spec.mark_alpha)
            if spec.reference is not None:
                ax_i.plot(spec.reference[0], spec.reference[1], color=INK["muted"], linewidth=1.2,
                          linestyle=(0, (2.5, 2.0)), zorder=2.8, label="_reference")
            for k, j in enumerate(idx):
                ls = _DASHES[k % len(_DASHES)] if linestyles else "solid"
                ax_i.plot(c.time, c.resp[:, j], color=spec.colours[j], linewidth=_LINE_WIDTH,
                          linestyle=ls, label=c.labels[j], zorder=3 + 0.01 * k,
                          solid_joinstyle="round")
                if spec.samples is not None:
                    s = spec.samples
                    ax_i.plot(s.time, s.resp[:, j], linestyle="none", marker="o", markersize=4.2,
                              markeredgewidth=0, color=spec.colours[j], zorder=4, label="_samples")
            # Limits: ggplot expansion (5% overlay, 14% above for direct
            # labels, 20% headroom for the peak note as R adds 18%, 8% stacked).
            full = hi - bottom if hi > bottom else 1.0
            if stack:
                pad_lo = pad_hi = 0.08
            else:
                pad_lo = 0.05
                if spec.direct_labels and lo < -0.25 * max(hi, 0.0):
                    pad_lo = 0.12
                pad_hi = 0.14 if spec.direct_labels else (0.20 if spec.peak else 0.05)
            if not (stack and spec.scales == "fixed" and ax_i is not axes[0]):
                ax_i.set_ylim(bottom - pad_lo * full, hi + pad_hi * full)
            ax_i.set_xlim(x_lo - 0.02 * x_range, x_hi + 0.02 * x_range)
            _set_budget_yticks(ax_i, stack=stack,
                               floor=_MARK_FLOOR[spec.layout] if has_marks else 0.0)
            ax_i.grid(visible=True, color=INK["grid"], linewidth=0.6)
            if stack:
                name = c.labels[idx[0]]
                ax_i.set_title(name, loc="left", fontsize=8.6, fontweight="bold",
                               color=_text_colour(colour_of[name]), pad=2.0)

        if stack:
            for a in axes[:-1]:
                a.tick_params(axis="x", length=0)
            axes[-1].set_xlabel(spec.xlab)
            if ax is None:
                fig.supylabel(spec.ylab, fontsize=9.0, color=INK["text2"], x=0.012)
        else:
            a = axes[0]
            a.set_xlabel(spec.xlab)
            a.set_ylabel(spec.ylab)

        if not stack and spec.peak and n == 1:
            y = c.resp[:, 0]
            if np.isfinite(y).any():
                i = int(np.nanargmax(y))
                a = axes[0]
                a.plot([c.time[i]], [y[i]], marker="o", markersize=5.5, markeredgewidth=0,
                       color=spec.colours[0], zorder=4, linestyle="none", label="_peak")
                a.annotate(f"Peak: {c.time[i]:.1f} s", (c.time[i], y[i]), xytext=(0, 7),
                           textcoords="offset points", ha="center", va="bottom", fontsize=8.6,
                           color=INK["text2"], zorder=6)

        title, subtitle, caption = spec.title, spec.subtitle, spec.caption
        if ax is not None:
            a = axes[0]
            if title:
                a.set_title(title, loc="left")
            if not stack and n > 1 and not spec.direct_labels:
                a.legend(frameon=False, ncol=_legend_ncol(c.labels, 4.5), loc="best")
            elif spec.direct_labels:
                fig.canvas.draw()
                _peak_labels(a, [(c.time, c.resp[:, j]) for j in range(n)], c.labels, spec.colours)
            return fig

        if not stack and n > 1 and not spec.direct_labels:
            ncol = _legend_ncol(c.labels, fig.get_size_inches()[0])
            from matplotlib.transforms import ScaledTranslation

            off = ScaledTranslation(0, -0.46, fig.dpi_scale_trans)
            handles, texts = _byrow(*axes[0].get_legend_handles_labels(), ncol)
            axes[0].legend(handles, texts, loc="upper left", bbox_to_anchor=(0.0, 0.0),
                           bbox_transform=axes[0].transAxes + off, ncol=ncol, frameon=False,
                           handlelength=2.2 if linestyles else 1.6, columnspacing=1.4,
                           borderaxespad=0.0, handletextpad=0.5)
        w_in = fig.get_size_inches()[0]
        finish(fig, title=title or None, subtitle=subtitle or None, caption=caption or None,
               title_width=max(int(w_in * 72 / (10.0 * 1.25 * 0.6)) - 2, 20))
        if stack and n > 1:
            # Air between a panel and the next panel's label (R panel.spacing).
            h_ax = axes[0].get_position().height * fig.get_size_inches()[1]
            fig.subplots_adjust(hspace=fig.subplotpars.hspace + 0.12 / max(h_ax, 0.1))
        if spec.direct_labels and not stack:
            fig.canvas.draw()
            _peak_labels(axes[0], [(c.time, c.resp[:, j]) for j in range(n)], c.labels, spec.colours)
    return fig


def _caption(user: str | None, default: str | None) -> str | None:
    if user is None:
        return default
    return user or None


# --------------------------------------------------------------------------
# Public plots
# --------------------------------------------------------------------------


def plot_hrfs(
    *hrfs: HrfInput,
    time: ArrayLike | None = None,
    normalize: bool = False,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    basis: BasisMode = "all",
    layout: Layout = "overlay",
    palette: PaletteKind = "auto",
    reference: HRF | None = None,
    reference_label: str | None = None,
    scales: Scales = "free_y",
    caption: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Compare HRFs on shared axes (R ``plot_hrfs``).

    Multi-basis HRFs expand to one curve per basis function, so
    ``plot_hrfs(HRF_SPMG3)`` shows the canonical response and both
    derivatives, labelled at their peaks.

    Args:
        *hrfs: HRF objects, a list of them, or one ``{label: hrf}`` mapping.
        time: Time points (s); defaults to 0 to the longest span, 0.1 s steps.
        normalize: Scale each curve to peak at 1 (compare shapes).
        labels: One label per HRF or per curve. Defaults to the HRF names;
            the columns of a single basis set are named after their role
            (``canonical``, ``temporal deriv.``...) or ``B1``, ``B2``...
        title: Defaults to the HRF name for one HRF, else "HRF comparison";
            ``""`` suppresses it.
        subtitle: Optional subtitle.
        basis: ``"all"`` basis functions or only the ``"first"``.
        layout: ``"overlay"`` (one panel) or ``"stack"`` (one panel per curve
            on a shared time axis).
        palette: ``"auto"`` uses the ordered ramp for a single basis set of
            more than three functions or more than six curves, else the
            categorical palette; ``"ordered"`` suits HRFs that differ along
            one parameter (lag, width).
        reference: Optional HRF drawn as a dashed grey curve outside the
            legend, evaluated and normalized like the others.
        reference_label: Names the reference in the caption; defaults to its
            name.
        scales: For ``layout="stack"``: ``"free_y"`` or ``"fixed"``.
        caption: Overrides the default caption ("Dashed grey: <reference>").
        ax: Draw a single-panel overlay into this axes.
        figsize: Figure size in inches; scaled to the panel count by default.

    Returns:
        The matplotlib Figure. :func:`hrf_curves` returns the plotted data.
    """
    _check_choice("layout", layout, ("overlay", "stack"))
    _check_choice("scales", scales, ("free_y", "fixed"))
    if reference is not None and not isinstance(reference, HRF):
        raise TypeError("'reference' must be an HRF object")
    curves = _hrf_curve_set(hrfs, time=time, normalize=normalize, labels=labels, basis=basis)
    items, _ = _collect(hrfs, HRF, "HRF")  # type: ignore[type-abstract]
    n_series = len(curves.labels)
    kind, colours = _series_palette(n_series, palette, curves.single_set and curves.nb[0] > 3)
    if title is None:
        title = str(getattr(items[0], "name", None) or "HRF") if len(items) == 1 else "HRF comparison"
    ref = None
    default_caption = None
    if reference is not None:
        ref_y = _eval_hrf(reference, curves.time, normalize, "first")[:, 0]
        ref = (curves.time, ref_y)
        name = reference_label or str(getattr(reference, "name", None) or "reference")
        default_caption = f"Dashed grey: {name}"
    spec = _Spec(
        curves=curves,
        colours=colours,
        kind=kind,
        title=title,
        subtitle=subtitle,
        caption=_caption(caption, default_caption),
        ylab="Response / peak" if normalize else "Response",
        layout=layout,
        scales=scales,
        reference=ref,
        direct_labels=curves.single_set and not curves.explicit_labels and layout == "overlay",
        figsize=figsize,
    )
    return _render(spec, ax)


def plot_hrf(
    hrf: HRF,
    time: ArrayLike | None = None,
    normalize: bool = False,
    show_peak: bool = True,
    ax: Axes | None = None,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot one HRF (R ``plot.HRF``).

    A single-basis HRF is drawn with its peak marked and annotated
    ("Peak: 5.0 s"); a multi-basis HRF draws every basis function in one
    panel, coloured and labelled by basis.

    Args:
        hrf: The HRF.
        time: Time points (s); defaults to 0 to ``hrf.span`` in 0.1 s steps.
        normalize: Scale each curve to peak at 1.
        show_peak: Mark the peak of a single-basis HRF.
        ax: Draw into this axes.
        title: Defaults to the HRF name; ``""`` suppresses it.
        subtitle: Optional subtitle.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    if not isinstance(hrf, HRF):
        raise TypeError(f"plot_hrf needs an HRF, got {type(hrf).__name__}")
    curves = _hrf_curve_set([hrf], time=time, normalize=normalize, labels=None, basis="all")
    n = len(curves.labels)
    if n > 1:
        return plot_hrfs(hrf, time=time, normalize=normalize, title=title, subtitle=subtitle,
                         ax=ax, figsize=figsize)
    kind, colours = _series_palette(1, "categorical", False)
    spec = _Spec(
        curves=curves,
        colours=colours,
        kind=kind,
        title=str(hrf.name or "HRF") if title is None else title,
        subtitle=subtitle,
        caption=None,
        ylab="Response / peak" if normalize else "Response",
        peak=show_peak,
        figsize=figsize or (6.0, 3.8),
    )
    return _render(spec, ax)


def plot_regressors(
    *regs: RegInput,
    grid: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    show_onsets: OnsetMode = None,
    onset_alpha: float = 0.8,
    precision: float | None = None,
    basis: BasisMode = "all",
    layout: Layout = "overlay",
    samples: ArrayLike | None = None,
    palette: PaletteKind = "auto",
    scales: Scales = "free_y",
    caption: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Compare regressors on a shared time axis (R ``plot_regressors``).

    Basis-set regressors expand to one curve per basis function and a
    :class:`~fmrimod.regressor.core.RegressorSet` to one curve per
    condition. Events are drawn as bars from onset to offset in a band under
    the curves; point events get a minimum width of 0.4% of the time range.

    Args:
        *regs: Regressors, a list of them, a ``{label: regressor}`` mapping,
            or one ``RegressorSet``.
        grid: Evaluation times; defaults to 0 to the last onset plus span in
            0.25 s steps. Use a fine grid with ``samples`` to show scan times.
        labels: One label per regressor or per curve; defaults to the
            mapping keys, condition levels, or "Regressor 1", "Regressor 2"...
        title: Defaults to the curve label for one curve, else "Regressor
            comparison"; ``""`` suppresses it.
        subtitle: Optional subtitle.
        show_onsets: ``True`` marks every curve's events in its colour;
            ``"first"`` only the first regressor's, in grey; ``False`` none.
            ``None`` uses ``"first"`` for overlays and ``True`` for stacks
            (each panel marks its own events in grey).
        onset_alpha: Opacity of the event bars.
        precision: HRF sampling precision; defaults to the grid spacing,
            capped at 0.33 s and floored at 0.01 s.
        basis: ``"all"`` basis columns or only the ``"first"``.
        layout: ``"overlay"`` or ``"stack"``.
        samples: Sample times (e.g. scan acquisitions) at which the
            regressors are evaluated and drawn as points.
        palette: ``"auto"``, ``"categorical"`` or ``"ordered"``.
        scales: For ``layout="stack"``: ``"free_y"`` or ``"fixed"`` (one
            shared range, to compare amplitudes).
        caption: Overrides the default caption describing the event bars.
        ax: Draw a single-panel overlay into this axes.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure. :func:`regressor_curves` returns the data.
    """
    _check_choice("layout", layout, ("overlay", "stack"))
    _check_choice("scales", scales, ("free_y", "fixed"))
    if show_onsets not in (None, True, False, "first"):
        raise ValueError("show_onsets must be True, False, 'first' or None")
    curves, items, prec = _reg_curve_set(regs, grid=grid, precision=precision, labels=labels,
                                         basis=basis)
    marks, by_series = _event_marks(curves, items, show_onsets, layout)
    sample_curves = None
    if samples is not None:
        st = np.asarray(samples, dtype=np.float64).reshape(-1)
        mats = [_eval_reg(r, st, prec, basis) for r in items]
        sample_curves = _Curves(time=st, resp=np.hstack(mats), labels=curves.labels,
                                owner=curves.owner, nb=curves.nb, explicit_labels=True)
    n_series = len(curves.labels)
    kind, colours = _series_palette(n_series, palette, curves.single_set and curves.nb[0] > 3)
    if title is None:
        title = curves.labels[0] if n_series == 1 else "Regressor comparison"
    default_caption = None
    if marks is not None:
        default_caption = "Bars: events, onset to offset"
    spec = _Spec(
        curves=curves,
        colours=colours,
        kind=kind,
        title=title,
        subtitle=subtitle,
        caption=_caption(caption, default_caption),
        ylab="Response",
        layout=layout,
        scales=scales,
        marks=marks,
        marks_by_series=by_series,
        mark_alpha=onset_alpha,
        samples=sample_curves,
        figsize=figsize,
    )
    return _render(spec, ax)


def plot_regressor(
    reg: Regressor,
    grid: ArrayLike | None = None,
    precision: float | None = None,
    show_onsets: bool = True,
    ax: Axes | None = None,
    *,
    layout: Layout = "stack",
    onset_alpha: float = 0.5,
    title: str | None = None,
    subtitle: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot one regressor's predicted response (R ``plot.Reg``).

    Titled "<HRF> regressor, N events", with each event drawn as a grey bar
    under the curve. A basis-set regressor gets one panel per basis
    function (``layout="overlay"`` puts them in one panel).

    Args:
        reg: The :class:`~fmrimod.regressor.core.Regressor`.
        grid: Evaluation times; defaults to 0 to the last onset plus span in
            0.25 s steps.
        precision: HRF sampling precision; defaults to the grid spacing,
            capped at 0.33 s.
        show_onsets: Draw the event bars.
        ax: Draw a single panel into this axes (overlay only).
        layout: ``"stack"`` or ``"overlay"`` for multi-basis regressors.
        onset_alpha: Opacity of the event bars.
        title: Overrides the default title; ``""`` suppresses it.
        subtitle: Optional subtitle.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    if not isinstance(reg, Regressor):
        raise TypeError(f"plot_regressor needs a Regressor, got {type(reg).__name__}")
    n_events = len(reg.onsets)
    if title is None:
        title = f"{_hrf_name(reg)} regressor, {n_events} event{'' if n_events == 1 else 's'}"
    nb = reg.nbasis
    labels = None if nb > 1 else ["Response"]
    if nb > 1 and ax is not None:
        layout = "overlay"
    return plot_regressors(
        reg,
        grid=grid,
        labels=labels,
        title=title,
        subtitle=subtitle,
        show_onsets="first" if show_onsets else False,
        onset_alpha=onset_alpha,
        precision=precision,
        layout=layout if nb > 1 else "overlay",
        ax=ax,
        figsize=figsize or ((6.5, 3.8) if nb == 1 else None),
    )
