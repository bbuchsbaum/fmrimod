"""Regression guard for the documented first-level->group seam import story.

The golden-path tutorial (docs/tutorials/golden-path.qmd) teaches the
promoted top-level import for ``group_dataset_from_contrasts``::

    from fmrimod import group_dataset_from_contrasts

The helper was top-level promoted (tiered ``spine`` in
api_inventory_v1.json) so the tutorial no longer needs a namespace
caveat. The submodule form remains importable for back-compat, and the
``fmrimod.contrast`` package attribute is still the ``fm.contrast(...)``
builder function (so ``import fmrimod.contrast as c;
c.group_dataset_from_contrasts`` is still an AttributeError — asserted
below). This test fails the moment the documented story stops being
true, so the doc cannot rot unnoticed.

Owner: bd-01KRRNT5PQCYKT61F6EVNWQ95G (top-level promotion); supersedes
the submodule-only contract from bd-01KRRM5CY3J08X8N4V809XWTB9.
(Board source: tutorials-win-or-lose/post-01KRRM2J2ZREJSTBQHQPP1HT2E,
precondition for the golden-path tutorial bd-01KRRM5Y8B60G60KFBJ727FECS.)
"""

from __future__ import annotations

import sys
import types

import pytest


def test_canonical_form_works_and_is_callable() -> None:
    """The form the tutorial teaches must resolve and be callable."""
    from fmrimod.contrast import group_dataset_from_contrasts

    assert callable(group_dataset_from_contrasts)


def test_contrast_attribute_is_the_builder_function_not_the_submodule() -> None:
    """``import fmrimod.contrast as c`` binds the ``contrast`` builder
    *function*, so ``c.group_dataset_from_contrasts`` is an AttributeError.

    This is the first wrong turn a user takes; the tutorial's note exists
    precisely because of it. If this assertion ever fails, the shadow has
    changed and the tutorial prose must be revisited.
    """
    import fmrimod.contrast as c

    assert isinstance(c, types.FunctionType), (
        "fmrimod.contrast attribute is no longer the builder function; the "
        "golden-path tutorial's import note is now stale"
    )
    with pytest.raises(AttributeError):
        _ = c.group_dataset_from_contrasts  # type: ignore[attr-defined]


def test_real_submodule_still_reachable_via_sys_modules() -> None:
    """The genuine package is still importable through the import system and
    carries the helper in its ``__all__`` (the export is correct; only the
    attribute lookup is shadowed)."""
    from fmrimod.contrast import group_dataset_from_contrasts  # noqa: F401  (loads it)

    pkg = sys.modules.get("fmrimod.contrast")
    assert isinstance(pkg, types.ModuleType)
    assert "group_dataset_from_contrasts" in getattr(pkg, "__all__", ())
    assert hasattr(pkg, "group_dataset_from_contrasts")


def test_helper_is_top_level_promoted() -> None:
    """``from fmrimod import group_dataset_from_contrasts`` works.

    The deliberate namespace-policy promotion in
    bd-01KRRNT5PQCYKT61F6EVNWQ95G was taken (it is tiered ``spine`` in
    api_inventory_v1.json), so the tutorial teaches the short top-level
    form. The submodule form remains for back-compat (asserted above).
    """
    from fmrimod import group_dataset_from_contrasts

    assert callable(group_dataset_from_contrasts)
    from fmrimod.contrast import group_dataset_from_contrasts as _submodule

    assert group_dataset_from_contrasts is _submodule


def _python_cells(qmd_text: str) -> str:
    """Concatenate the executable ```{python}``` cells of a Quarto doc.

    The import guard is about *code the page runs*, not prose: the page is
    expected to discuss the shadowed forms in its explanatory callout, but
    must never execute them.
    """
    import re

    return "\n".join(re.findall(r"```\{python\}\n(.*?)```", qmd_text, re.S))


def test_golden_path_tutorial_teaches_the_canonical_form_only() -> None:
    """The tutorial's *code cells* teach the promoted top-level form and
    never execute a shadowed form.

    Skips until ``golden-path.qmd`` lands (owned by the golden-path tutorial
    bead bd-01KRRM5Y8B60G60KFBJ727FECS); the import-behaviour guards above
    are unconditional and carry the blocker on their own.
    """
    from pathlib import Path

    qmd = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "golden-path.qmd"
    if not qmd.exists():
        pytest.skip("golden-path.qmd not yet committed (bd-01KRRM5Y8B60G60KFBJ727FECS)")
    code = _python_cells(qmd.read_text())

    assert "from fmrimod import group_dataset_from_contrasts" in code, (
        "tutorial code no longer teaches the promoted top-level import form"
    )
    assert "import fmrimod.contrast as" not in code, (
        "tutorial executes the shadowed attribute form"
    )
