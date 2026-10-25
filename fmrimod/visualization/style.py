"""fmrimod's visual design system: colour tokens, colormaps and the theme.

One system for every fmrimod figure, so a condition keeps its colour from the
design plot to the fit plot. The tokens follow R fmridesign's
``R/plot_theme.R`` (the most complete of the three R systems); fmrihrf's and
fmrireg's idioms (direct labels, run axis, recessive grids) are carried by
:mod:`fmrimod.visualization._helpers`.

Colour roles:

- :data:`INK` - text, secondary text, muted labels, rules, grid, bands and
  the zero line.
- :data:`CATEGORICAL` - up to seven series, assigned by design position and
  never cycled. ``CATEGORICAL[1]`` doubles as the warning colour
  (:data:`WARN`).
- :data:`DIVERGING` - signed values about zero (design maps, correlations,
  weights), neutral warm-grey midpoint.
- :data:`MAGNITUDE` - non-negative values, white to the warm pole.

The theme is a matplotlib rc dict applied inside every fmrimod plot
function, so figures look the same whatever the user's global style.
:func:`set_theme` overrides it (like R's ``options(fmrireg.plot_theme)``).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

INK: dict[str, str] = {
    "text": "#1C232E",
    "text2": "#4A5465",
    "muted": "#7D8697",
    "rule": "#DADDE3",
    "grid": "#ECEEF1",
    "band": "#F4F5F7",
    "zero": "#B8BEC8",
}

CATEGORICAL: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#C28A00",
    "#7B5CB8",
    "#4FA3D9",
    "#B04A7A",
)

WARN: str = CATEGORICAL[1]

DIVERGING: tuple[str, ...] = (
    "#1D4F7A",
    "#5D8FBF",
    "#B9D0E6",
    "#F7F6F4",
    "#EDBBA6",
    "#C8664A",
    "#8A2A1C",
)

MAGNITUDE: tuple[str, ...] = ("#FFFFFF", "#EDBBA6", "#C8664A", "#8A2A1C", "#4A140B")

# Ordered ramp for more series than CATEGORICAL holds (e.g. FIR bases,
# trialwise columns): light blue to the categorical blue to navy.
ORDERED: tuple[str, ...] = ("#9DB7D5", "#0072B2", "#12324F")

FONT_FAMILY: tuple[str, ...] = (
    "Inter",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
)

BASE_SIZE: float = 10.0


# --------------------------------------------------------------------------
# Colour arithmetic (CIE Lab interpolation, as R's colorRampPalette(space="Lab"))
# --------------------------------------------------------------------------

_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def _hex_to_rgb(colour: str) -> np.ndarray:
    c = colour.lstrip("#")
    return np.array([int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _rgb_to_hex(rgb: np.ndarray) -> str:
    v = np.clip(np.round(np.asarray(rgb) * 255.0), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*v)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (_M_RGB2XYZ @ lin) / _WHITE_D65
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[0] + 16) / 116
    f = np.array([fy + lab[1] / 500, fy, fy - lab[2] / 200])
    xyz = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE_D65
    lin = np.linalg.solve(_M_RGB2XYZ, xyz)
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def ramp(anchors: Sequence[str], n: int) -> list[str]:
    """``n`` colours evenly spaced through ``anchors`` in CIE Lab."""
    if n <= 0:
        return []
    if n == 1:
        return [anchors[0]]
    labs = np.array([_srgb_to_lab(_hex_to_rgb(a)) for a in anchors])
    pos = np.linspace(0.0, len(anchors) - 1, n)
    out = []
    for p in pos:
        i = min(int(np.floor(p)), len(anchors) - 2)
        t = p - i
        out.append(_rgb_to_hex(_lab_to_srgb((1 - t) * labs[i] + t * labs[i + 1])))
    return out


def tints(colour: str, n: int) -> list[str]:
    """``n`` steps from ``colour`` towards white (R ``.fd_tints``)."""
    return ramp([colour, "#FFFFFF"], n + 2)[:n]


def shade(colour: str, amount: float = 0.45) -> str:
    """Mix ``colour`` towards the text ink by ``amount`` (R ``.fd_shade``)."""
    a, b = _hex_to_rgb(colour), _hex_to_rgb(INK["text"])
    return _rgb_to_hex((1 - amount) * a + amount * b)


def palette(n: int, kind: str = "categorical") -> list[str]:
    """Series colours.

    ``"categorical"`` gives the first ``n`` of :data:`CATEGORICAL` and falls
    back to the :data:`ORDERED` ramp above seven; ``"ordered"`` always ramps;
    ``"diverging"`` ramps :data:`DIVERGING`.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if kind == "categorical":
        return list(CATEGORICAL[:n]) if n <= len(CATEGORICAL) else ramp(ORDERED, n)
    if kind == "ordered":
        return ramp(ORDERED, n)
    if kind == "diverging":
        return ramp(DIVERGING, n)
    raise ValueError(f"kind must be 'categorical', 'ordered' or 'diverging', got {kind!r}")


def colour_map(keys: Sequence[Hashable]) -> dict[Hashable, str]:
    """Stable colour per key in first-appearance order (R ``.fd_colour_map``)."""
    uniq = list(dict.fromkeys(keys))
    return dict(zip(uniq, palette(len(uniq)), strict=False))


def cmap(kind: str = "diverging") -> Colormap:
    """Matplotlib colormap for ``"diverging"``, ``"magnitude"`` or ``"grey"``."""
    from matplotlib.colors import LinearSegmentedColormap

    stops = {
        "diverging": DIVERGING,
        "magnitude": MAGNITUDE,
        "grey": ("#FFFFFF", INK["text"]),
    }.get(kind)
    if stops is None:
        raise ValueError(f"unknown colormap kind {kind!r}")
    cm = LinearSegmentedColormap.from_list(f"fmrimod_{kind}", ramp(stops, 256))
    cm.set_bad(INK["band"])
    return cm


# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------


_FONTS_CACHE: list[str] | None = None


def _available_fonts() -> list[str]:
    """The installed members of :data:`FONT_FAMILY` (missing ones make
    matplotlib log a warning per lookup)."""
    global _FONTS_CACHE
    if _FONTS_CACHE is None:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
        _FONTS_CACHE = [f for f in FONT_FAMILY if f in installed] or ["DejaVu Sans"]
    return _FONTS_CACHE


def _default_rc(base_size: float = BASE_SIZE) -> dict[str, object]:
    from cycler import cycler

    return {
        "font.family": _available_fonts(),
        "font.sans-serif": _available_fonts(),
        "font.size": base_size,
        "text.color": INK["text"],
        "axes.labelcolor": INK["text2"],
        "axes.labelsize": base_size * 0.9,
        "axes.titlesize": base_size * 0.9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK["text"],
        "axes.titlepad": 4.0,
        "axes.edgecolor": INK["rule"],
        "axes.linewidth": 0.0,
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
        "axes.unicode_minus": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": INK["rule"],
        "ytick.color": INK["rule"],
        "xtick.labelcolor": INK["muted"],
        "ytick.labelcolor": INK["muted"],
        "xtick.labelsize": base_size * 0.82,
        "ytick.labelsize": base_size * 0.82,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "legend.frameon": False,
        "legend.fontsize": base_size * 0.82,
        "legend.title_fontsize": base_size * 0.85,
        "legend.labelcolor": INK["text"],
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.2,
        "figure.facecolor": "white",
        "figure.dpi": 110,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
        "image.interpolation": "nearest",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }


_THEME_OVERRIDE: dict[str, object] | None = None


def theme_rc(base_size: float = BASE_SIZE) -> dict[str, object]:
    """The rc dict fmrimod plots draw under (defaults plus :func:`set_theme`)."""
    rc = _default_rc(base_size)
    if _THEME_OVERRIDE:
        rc.update(_THEME_OVERRIDE)
    return rc


def set_theme(rc: Mapping[str, object] | None = None) -> None:
    """Override rc keys for every fmrimod figure; ``None`` restores the default.

    >>> set_theme({"font.size": 12, "font.sans-serif": ["Arial"]})
    """
    global _THEME_OVERRIDE
    _THEME_OVERRIDE = dict(rc) if rc else None


@contextmanager
def theme(base_size: float = BASE_SIZE) -> Iterator[None]:
    """Context manager that draws under the fmrimod theme.

    Use it to style your own figures consistently with fmrimod's::

        with fmrimod.visualization.theme():
            fig, ax = plt.subplots()
    """
    import matplotlib as mpl

    with mpl.rc_context(theme_rc(base_size)):
        yield


__all__ = [
    "BASE_SIZE",
    "CATEGORICAL",
    "DIVERGING",
    "INK",
    "MAGNITUDE",
    "ORDERED",
    "WARN",
    "cmap",
    "colour_map",
    "palette",
    "ramp",
    "set_theme",
    "shade",
    "theme",
    "theme_rc",
    "tints",
]
