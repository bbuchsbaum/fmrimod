"""Shared pieces of the design-plot family (R fmridesign ``plot_theme.R``).

Column metadata with baseline labels, the per-scan timeline, column groups
and their spanners, structured factorial colours, colourbars and the small
text formatters the event, heatmap, contrast and baseline plots share.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._helpers import DOT, MINUS, TIMES, column_info
from .style import BASE_SIZE, CATEGORICAL, INK, colour_map, ramp, shade, tints

NDASH = "–"
SIGMA = "σ"
GE = "≥"
PM = "±"


# --------------------------------------------------------------------------
# Model access
# --------------------------------------------------------------------------


def model_design_matrix(model: Any) -> np.ndarray:
    """The full design matrix as a float array (``FmriModel`` builds it on call)."""
    dm = model.design_matrix
    if callable(dm):
        dm = dm()
    if isinstance(dm, pd.DataFrame):
        dm = dm.to_numpy()
    out = np.asarray(dm, dtype=float)
    return out.reshape(-1, 1) if out.ndim == 1 else out


def model_sampling_frame(model: Any) -> Any:
    for attr in ("sampling_frame", "sampling_info"):
        try:
            sf = getattr(model, attr, None)
        except Exception:  # pragma: no cover - a dataset that cannot answer
            sf = None
        if sf is not None and hasattr(sf, "blocklens"):
            return sf
    return None


def is_baseline_model(model: Any) -> bool:
    from ..baseline.baseline_model import BaselineModel

    return isinstance(model, BaselineModel)


_DRIFT_RE = re.compile(r"^base_[A-Za-z]+?(\d+)_block_\d+$")


def design_columns(model: Any) -> pd.DataFrame:
    """:func:`column_info` with baseline rows labelled by role.

    Baseline columns are grouped as ``drift`` (labelled ``drift k``),
    ``block`` (``intercept``) and ``nuisance`` (the user's column names),
    as in R ``.fd_colinfo``; their ``full_label`` adds the run.
    """
    ci = column_info(model).copy()
    trial = (ci["kind"] == "event") & (ci["term"] == "trial") & ci["condition"].astype(str).str.fullmatch(r"\d+")
    for i in np.flatnonzero(trial.to_numpy()):
        lab = f"trial {int(ci.at[i, 'condition'])}"
        ci.at[i, "condition"] = lab
        ci.at[i, "full_label"] = str(ci.at[i, "full_label"]).replace(str(ci.at[i, "label"]), lab, 1)
        ci.at[i, "label"] = lab
    base = ci["kind"] == "baseline"
    if not base.any():
        return ci
    for i in np.flatnonzero(base.to_numpy()):
        name = str(ci.at[i, "name"])
        role = ci.at[i, "role"]
        run = ci.at[i, "run"]
        has_run = run is not None and not (isinstance(run, float) and np.isnan(run))
        if role == "drift":
            m = _DRIFT_RE.match(name)
            label = f"drift {m.group(1)}" if m else re.sub(r"^base_", "", name)
            term, term_label = "drift", "drift"
        elif role == "intercept":
            label, term, term_label = "intercept", "block", "intercept"
        else:
            label = re.sub(r"^nuis_run\d+_", "", name)
            label = re.sub(r"_block_\d+$", "", label)
            term, term_label = "nuisance", "nuisance"
        ci.at[i, "term"] = term
        ci.at[i, "term_label"] = term_label
        ci.at[i, "label"] = label
        ci.at[i, "full_label"] = f"{label} {DOT} run {int(run)}" if has_run else label
        ci.at[i, "run"] = int(run) if has_run else None
    seen: dict[str, int] = {}
    vals = []
    for v in ci["full_label"]:
        k = seen.get(v, 0)
        vals.append(v if k == 0 else f"{v} #{k + 1}")
        seen[v] = k + 1
    ci["full_label"] = vals
    return ci


def short_labels(ci: pd.DataFrame) -> list[str]:
    """Column labels for views whose term spanners already name the term.

    The bare condition (plus basis) for event columns; ``full_label`` when
    that would not be unique within its term.
    """
    out = []
    for r in ci.itertuples():
        if r.kind == "event" and isinstance(r.condition, str) and r.condition:
            out.append(r.condition + (f" {DOT} {r.basis_label}" if isinstance(r.basis_label, str) else ""))
        else:
            out.append(r.full_label)
    key = list(zip(ci["term"], out, strict=False))
    if len(set(key)) != len(key):
        return list(ci["full_label"])
    return out


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------


@dataclass
class Timeline:
    run: np.ndarray  # 1-based run of each scan
    time: np.ndarray  # scan time (s) or scan number
    runs: pd.DataFrame  # run, start, end, scans, TR, first_scan, last_scan (1-based)
    unit: str  # "s" or "scan"
    TR: np.ndarray

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def bounds(self) -> list[float]:
        """Run edges for :func:`run_axis` (global time only)."""
        return [float(self.runs["start"].iloc[0])] + [float(v) for v in self.runs["end"]]


def timeline(sframe: Any, n_rows: int, block_x: str = "global") -> Timeline:
    """Per-scan runs and times plus a run table (R ``.fd_timeline``).

    A frame whose scans do not match ``n_rows`` falls back to scan units.
    """
    if block_x not in ("global", "run"):
        raise ValueError(f"block_x must be 'global' or 'run', got {block_x!r}")
    bl = None if sframe is None else np.asarray(sframe.blocklens, dtype=int)
    if bl is None or int(bl.sum()) != n_rows:
        runs = pd.DataFrame({"run": [1], "start": [0.5], "end": [n_rows + 0.5], "scans": [n_rows],
                             "TR": [1.0], "first_scan": [1], "last_scan": [n_rows]})
        return Timeline(np.ones(n_rows, dtype=int), np.arange(1, n_rows + 1, dtype=float), runs,
                        "scan", np.array([1.0]))
    tr = np.broadcast_to(np.asarray(sframe.tr, dtype=float), bl.shape).copy()
    run = np.repeat(np.arange(1, len(bl) + 1), bl)
    time = np.asarray(sframe.sample_times(global_time=(block_x == "global")), dtype=float)
    dur = bl * tr
    if block_x == "global":
        end = np.cumsum(dur)
        start = np.concatenate([[0.0], end[:-1]])
    else:
        start = np.zeros(len(bl))
        end = dur
    last = np.cumsum(bl)
    runs = pd.DataFrame({"run": np.arange(1, len(bl) + 1), "start": start, "end": end, "scans": bl,
                         "TR": tr, "first_scan": last - bl + 1, "last_scan": last})
    return Timeline(run, time, runs, "s", tr)


def fmt_tr(tr: Sequence[float]) -> str:
    """``2 s`` or ``2/1.5 s``."""
    vals = list(dict.fromkeys(float(v) for v in np.atleast_1d(tr)))
    return "/".join(f"{v:g}" for v in vals) + " s"


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def resolve(value: str | None, default: str | None) -> str | None:
    """``None`` means the default, ``""`` suppresses."""
    if value is None:
        return default
    return value or None


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def name_or_count(items: Sequence[str], word: str, max_named: int = 3) -> str:
    items = list(items)
    return ", ".join(items) if len(items) <= max_named else plural(len(items), word)


def fd_num(x: float, digits: int = 2) -> str:
    """R ``.fd_num``: ``digits`` significant figures, plain notation, true minus."""
    if x is None or not np.isfinite(x):
        return ""
    if x == 0:
        return "0"
    mag = int(np.floor(np.log10(abs(x))))
    dec = max(digits - 1 - mag, 0)
    s = f"{round(float(x), dec):.{dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace("-", MINUS)


def fmt_weight(w: float) -> str:
    """Contrast weights as ``+1``, ``−0.5``, ``+0.333``; zero as ``0``."""
    if not np.isfinite(w):
        return ""
    if w == 0:
        return "0"
    s = f"{abs(w):.3g}"
    return (MINUS if w < 0 else "+") + s


def range_label(lo: float, hi: float) -> str:
    """A column's range as ``0 to 1.9``; a small opposite-sign excursion
    (at most 5% of the peak, e.g. an HRF undershoot) reads as 0."""
    peak = max(abs(lo), abs(hi))
    lo_d = 0.0 if (lo < 0 < hi and abs(lo) <= 0.05 * peak) else lo
    hi_d = 0.0 if (hi > 0 > lo and abs(hi) <= 0.05 * peak) else hi
    return f"{fd_num(lo_d)} to {fd_num(hi_d)}"


def text_colour(v: float, threshold: float = 0.6) -> str:
    """Readable text on a diverging fill of relative strength ``v``."""
    return "#FFFFFF" if np.isfinite(v) and abs(v) >= threshold else INK["text"]


def wrap_items(items: Sequence[str], width: int, sep: str = "  ") -> str:
    """Join ``items`` with ``sep``, breaking lines at about ``width`` chars."""
    lines: list[str] = []
    cur = ""
    for it in items:
        if cur and len(cur) + len(sep) + len(it) > width:
            lines.append(cur)
            cur = it
        else:
            cur = f"{cur}{sep}{it}" if cur else it
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Groups and spanners
# --------------------------------------------------------------------------


@dataclass
class Group:
    start: int  # 0-based, inclusive
    end: int  # 0-based, inclusive
    label: str
    short: str | None = None

    @property
    def centre(self) -> float:
        return (self.start + self.end) / 2


def groups_of(keys: Sequence[Any], labels: Sequence[Any] | None = None) -> list[Group]:
    """Runs of consecutive equal keys (R ``.fd_hm_groups``)."""
    keys = [str(k) for k in keys]
    labels = list(labels) if labels is not None else keys
    out: list[Group] = []
    for i, k in enumerate(keys):
        if out and keys[out[-1].start] == k and out[-1].end == i - 1:
            out[-1].end = i
        else:
            out.append(Group(i, i, str(labels[i])))
    return out


def boundaries(groups: Sequence[Group]) -> list[float]:
    """Positions (0-based cell units) of the rules between groups."""
    return [g.end + 0.5 for g in groups[:-1]]


def draw_spanners(
    ax: Any,
    groups: Sequence[Group],
    *,
    text_pt: float = 6.0,
    rule_pt: float | None = 3.0,
    size: float = BASE_SIZE * 0.86,
    weight: str = "bold",
    colour: str | None = None,
    overflow_pt: float = 24.0,
) -> list[Any]:
    """Term spanners above ``ax``: a label over a thin rule per group.

    x is in data units (columns at 0..p-1), y in points above the axes.
    Labels that do not fit their group are shortened by
    :func:`fit_spanners` after layout.
    """
    import matplotlib.transforms as mtransforms

    fig = ax.figure
    base = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    out = []
    for g in groups:
        if not g.label:
            continue
        t_text = base + mtransforms.ScaledTranslation(0, text_pt / 72, fig.dpi_scale_trans)
        txt = ax.text(g.centre, 1.0, g.label, transform=t_text, ha="center", va="bottom",
                      fontsize=size, fontweight=weight, color=colour or INK["text"], clip_on=False)
        short = g.short if g.short is not None else (g.label[:3] + "." if len(g.label) > 5 else g.label)
        txt._fd_spanner = (g.start - 0.5, g.end + 0.5, g.label, short, overflow_pt)  # type: ignore[attr-defined]
        out.append(txt)
        if rule_pt is not None:
            t_rule = base + mtransforms.ScaledTranslation(0, rule_pt / 72, fig.dpi_scale_trans)
            ln = ax.plot([g.start - 0.5, g.end + 0.5], [1.0, 1.0], transform=t_rule, color=INK["muted"],
                         linewidth=0.8, solid_capstyle="butt", clip_on=False)[0]
            ln._fd_rule = (g.start - 0.5, g.end + 0.5)  # type: ignore[attr-defined]
            out.append(ln)
    return out


def fit_spanners(fig: Any) -> None:
    """Shorten spanner labels that overflow their group; inset the rules."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    for ax in fig.axes:
        for art in list(ax.texts) + list(ax.lines):
            spec = getattr(art, "_fd_spanner", None)
            if spec is not None:
                x0, x1, label, short, over = spec
                width = abs(ax.transData.transform((x1, 0))[0] - ax.transData.transform((x0, 0))[0])
                for cand in (label, short, ""):
                    art.set_text(cand)
                    if art.get_window_extent(renderer).width <= width - 4 * pt + over * pt:
                        break
            rule = getattr(art, "_fd_rule", None)
            if rule is not None:
                x0, x1 = rule
                d0, d1 = ax.transData.transform((x0, 0))[0], ax.transData.transform((x1, 0))[0]
                pad = max(0.0, min(3 * pt, (d1 - d0) / 4))
                inv = ax.transData.inverted()
                art.set_xdata([inv.transform((d0 + pad, 0))[0], inv.transform((d1 - pad, 0))[0]])


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


def structured_colours(ci: pd.DataFrame, row_key: Sequence[str]) -> dict[str, str] | None:
    """Row colours that expose factorial structure (R ``.fd_structured_colours``).

    For a term crossing two or more factors, hue follows the first factor's
    level and lightness the remaining factors' combination (darker = later
    level). Other terms take the next free hues. ``None`` when that needs
    more hues than the categorical palette has.
    """
    rows = list(dict.fromkeys(row_key))
    keys = list(row_key)
    first = ci.iloc[[keys.index(r) for r in rows]]
    hue_key: list[str] = [""] * len(rows)
    shade_ix = [0] * len(rows)
    shade_n = [1] * len(rows)
    sep = f" {TIMES} "
    terms = list(first["term"].astype(str))
    conds = [c if isinstance(c, str) and c else lab for c, lab in zip(first["condition"], first["label"], strict=False)]
    for tt in dict.fromkeys(terms):
        ii = [i for i, t in enumerate(terms) if t == tt]
        parts = [conds[i].split(sep) for i in ii]
        nf = {len(p) for p in parts}
        if len(nf) == 1 and nf.pop() >= 2:
            rest = [sep.join(p[1:]) for p in parts]
            levels = list(dict.fromkeys(rest))
            for j, i in enumerate(ii):
                hue_key[i] = f"{tt} {parts[j][0]}"
                shade_ix[i] = levels.index(rest[j])
                shade_n[i] = len(levels)
        else:
            for i in ii:
                hue_key[i] = f"{tt} {conds[i]}"
    hues = list(dict.fromkeys(hue_key))
    if len(hues) > len(CATEGORICAL):
        return None
    base = dict(zip(hues, CATEGORICAL, strict=False))
    out = {}
    for i, r in enumerate(rows):
        b = base[hue_key[i]]
        out[r] = b if shade_n[i] <= 1 else ramp([tints(b, 3)[1], b, shade(b)], shade_n[i])[shade_ix[i]]
    return out


def row_colours(ci: pd.DataFrame, row_key: Sequence[str]) -> dict[str, str]:
    """Structured colours, or one hue per term when there are too many rows."""
    struct = structured_colours(ci, row_key)
    if struct is not None:
        return struct
    keys = list(row_key)
    rows = list(dict.fromkeys(keys))
    term_of = {r: str(ci["term"].iloc[keys.index(r)]) for r in rows}
    cmap = colour_map([term_of[r] for r in rows])
    return {r: cmap[term_of[r]] for r in rows}


# --------------------------------------------------------------------------
# Colourbar
# --------------------------------------------------------------------------


def cbar_axes(
    ax: Any,
    where: str = "bottom",
    *,
    width_in: float = 1.5,
    height_pt: float = 6.0,
    gap_pt: float = 10.0,
) -> Any:
    """A slim colourbar axes anchored to ``ax``'s right edge.

    ``"bottom"`` sits ``gap_pt`` below the x tick labels and axis title;
    ``"top"`` sits ``gap_pt`` above the axes. The position is computed at
    draw time (an axes locator), so it follows layout changes, and the axes
    is a child of ``ax`` so ``tight_layout`` reserves room for it.
    """
    from matplotlib.transforms import Bbox

    cax = ax.inset_axes([0, 0, 1, 1])

    def locate(_a: Any, renderer: Any) -> Any:
        fig = ax.figure
        pos = ax.get_position()
        fh, fw = fig.get_figheight(), fig.get_figwidth()
        h = height_pt / 72 / fh
        w = min(width_in / fw, pos.width)
        if where == "top":
            y0 = pos.y1 + gap_pt / 72 / fh
        else:
            xb = ax.xaxis.get_tightbbox(renderer)
            low = fig.transFigure.inverted().transform((0, xb.y0))[1] if xb is not None else pos.y0
            y0 = min(low, pos.y0) - gap_pt / 72 / fh - h
        return Bbox.from_bounds(pos.x1 - w, y0, w, h)

    cax.set_axes_locator(locate)
    return cax


def colourbar(
    fig: Any,
    cax: Any,
    mappable: Any,
    label: str | None,
    ticks: Sequence[float] | None = None,
    *,
    label_side: str = "left",
) -> Any:
    """A slim horizontal colourbar in ``cax`` with its title beside it."""
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    if ticks is not None:
        cb.set_ticks(list(ticks))
    cb.ax.xaxis.set_major_formatter(_bar_formatter())
    cb.ax.tick_params(length=2, width=0.4, color=INK["text2"], labelsize=BASE_SIZE * 0.82,
                      labelcolor=INK["text"], pad=1.5)
    cb.ax.grid(False)
    if label:
        if label_side == "left":
            cb.ax.text(-0.04, 0.5, label, transform=cb.ax.transAxes, ha="right", va="center",
                       fontsize=BASE_SIZE * 0.85, color=INK["text2"])
        else:
            cb.ax.set_title(label, loc="left", fontsize=BASE_SIZE * 0.85, color=INK["text2"],
                            fontweight="normal", pad=3)
    return cb


def _bar_formatter() -> Any:
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _p=None: fd_num(v, 2) if abs(v) > 1e-12 else "0")


def fd_cmap(kind: str = "diverging") -> Any:
    """:func:`style.cmap` without the ``set_bad`` call that matplotlib 3.11
    flags as pending deprecation (uses ``with_extremes``)."""
    from matplotlib.colors import LinearSegmentedColormap

    from .style import DIVERGING, MAGNITUDE

    stops = {"diverging": DIVERGING, "magnitude": MAGNITUDE, "grey": ("#FFFFFF", INK["text"])}.get(kind)
    if stops is None:
        raise ValueError(f"unknown colormap kind {kind!r}")
    cm = LinearSegmentedColormap.from_list(f"fmrimod_{kind}", ramp(stops, 256))
    return cm.with_extremes(bad=INK["band"])


def signed_norm(limit: float, lower: float | None = None) -> Any:
    from matplotlib.colors import Normalize

    limit = limit if np.isfinite(limit) and limit > 0 else 1.0
    return Normalize(vmin=-limit if lower is None else lower, vmax=limit, clip=True)


__all__ = [
    "GE",
    "NDASH",
    "PM",
    "SIGMA",
    "Group",
    "Timeline",
    "boundaries",
    "cbar_axes",
    "colourbar",
    "design_columns",
    "draw_spanners",
    "fd_cmap",
    "fd_num",
    "fit_spanners",
    "fmt_tr",
    "fmt_weight",
    "groups_of",
    "is_baseline_model",
    "model_design_matrix",
    "model_sampling_frame",
    "name_or_count",
    "plural",
    "range_label",
    "resolve",
    "row_colours",
    "short_labels",
    "signed_norm",
    "structured_colours",
    "text_colour",
    "timeline",
    "wrap_items",
]
