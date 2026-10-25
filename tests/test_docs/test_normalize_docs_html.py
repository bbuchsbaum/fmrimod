"""Keep static tables and working widget dependencies through HTML cleanup."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "normalize_docs_html",
    Path(__file__).resolve().parents[2] / "scripts/normalize_docs_html.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LOADER = '<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'
BRIDGE = "<script>define('jquery', [],function() {return window.jQuery;})</script>"
TABLE = '<table><tr><td>2.5</td></tr></table>'


def test_static_table_and_other_scripts_survive_without_unused_amd() -> None:
    other = '<script src="site_libs/quarto.js"></script>'
    result = MODULE.normalize(LOADER + BRIDGE + TABLE + other)
    assert result == TABLE + other
    assert MODULE.normalize(result) == result


def test_real_amd_and_jupyter_widgets_keep_their_loader() -> None:
    for widget in [
        '<script>require(["plot"], function(plot) { plot(); });</script>',
        '<script type="application/vnd.jupyter.widget-view+json">{}</script>',
        '<script src="custom-amd-widget.js"></script>',
    ]:
        original = LOADER + BRIDGE + TABLE + widget
        assert MODULE.normalize(original) == original
