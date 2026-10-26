"""RST literal-block markers must not leak into rendered API pages."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "normalize_quartodoc",
    Path(__file__).resolve().parents[2] / "scripts/normalize_quartodoc.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_literal_marker_becomes_colon_and_python_fence() -> None:
    text = (
        "Canonical form (matches R ``fmrireg::fmri_lm``)::\n"
        "\n"
        "    fit = fmri_lm(spec, dataset)\n"
        "    fit = fmri_lm(spec, dataset, ar='ar1')\n"
        "\n"
        "Next paragraph.\n"
    )
    result = MODULE.normalize_literal_blocks(text)
    assert result == (
        "Canonical form (matches R ``fmrireg::fmri_lm``):\n"
        "\n"
        "```python\n"
        "fit = fmri_lm(spec, dataset)\n"
        "fit = fmri_lm(spec, dataset, ar='ar1')\n"
        "```\n"
        "\n"
        "Next paragraph.\n"
    )
    assert MODULE.normalize_literal_blocks(result) == result


def test_bare_marker_line_and_fenced_code_are_handled() -> None:
    text = "Example\n\n::\n\n    x = 1\n\n```python\nmod::thing::\n```\n"
    result = MODULE.normalize_literal_blocks(text)
    assert "::\n\n" not in result
    assert "```python\nx = 1\n```" in result
    assert "mod::thing::" in result


def test_rst_directive_becomes_labelled_prose_not_code() -> None:
    text = ".. deprecated::\n    Use `fmrimod.single.estimate_single_trial` instead.\n\nDispatcher for OLS.\n"
    result = MODULE.normalize_literal_blocks(text)
    assert result == "**Deprecated**: Use `fmrimod.single.estimate_single_trial` instead.\n\nDispatcher for OLS.\n"
    assert "```" not in result


def test_doctest_block_is_not_labelled_runnable_python() -> None:
    text = "Example::\n\n    >>> import fmrimod as fm\n    >>> fm.hrf('spmg1')\n"
    result = MODULE.normalize_literal_blocks(text)
    assert "```pycon\n>>> import fmrimod as fm" in result
    assert "```python" not in result


def test_code_block_directive_stays_code() -> None:
    result = MODULE.normalize_literal_blocks(".. code-block:: python\n\n    x = 1\n")
    assert "```python\nx = 1\n```" in result
    assert "**" not in result


def test_math_directive_becomes_display_math() -> None:
    result = MODULE.normalize_literal_blocks(".. math::\n\n    y = X\\beta + e\n\nAfter.\n")
    assert result == "$$\ny = X\\beta + e\n$$\n\nAfter.\n"
