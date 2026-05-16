"""Pins the import contract the golden-path tutorial teaches.

``docs/tutorials/golden-path.qmd`` walks the full typed spine
``fmri_dataset -> fmri_lm -> contrast -> group_dataset_from_contrasts
-> ols_voxelwise``. The first-level -> group seam helper
``group_dataset_from_contrasts`` is reachable only as
``from fmrimod.contrast import group_dataset_from_contrasts``.

That is *not* obvious, because ``fmrimod.contrast`` is a ratified
rebind: the top-level attribute is a callable, not the submodule
(see ``tests/test_public_api/test_namespace_shadowing.py`` for the
general policy). The consequence the tutorial must explain in prose:

* ``from fmrimod.contrast import group_dataset_from_contrasts`` works
  (Python's import machinery resolves the real submodule).
* ``from fmrimod import group_dataset_from_contrasts`` does **not**
  work (the helper is deliberately not top-level).
* ``import fmrimod.contrast as c; c.group_dataset_from_contrasts``
  does **not** work (``c`` is the callable, not the submodule).

This test pins those three states so the tutorial's documented import
cannot silently rot. If a future change promotes the helper to the
top level (the recommended resolution in
bd-01KRRM5CY3J08X8N4V809XWTB9), this test fails loudly and forces the
tutorial prose to be updated in the same commit — which is the point.

Refs: bd-01KRRM5CY3J08X8N4V809XWTB9 (this precondition),
bd-01KRRM5Y8B60G60KFBJ727FECS (the golden-path page that depends on
it), board tutorials-win-or-lose/post-01KRRM2J2ZREJSTBQHQPP1HT2E.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def test_taught_import_form_works_and_is_callable() -> None:
    """The exact form the tutorial teaches must resolve to a callable."""
    from fmrimod.contrast import group_dataset_from_contrasts

    assert callable(group_dataset_from_contrasts), (
        "the golden-path tutorial teaches `from fmrimod.contrast import "
        "group_dataset_from_contrasts`; it must resolve to a callable."
    )


def test_helper_is_discoverable_via_submodule_all() -> None:
    """``importlib.import_module`` yields the real submodule exposing
    the helper in ``__all__`` — so it is discoverable from the
    submodule even though the top-level attribute is shadowed."""
    submodule = importlib.import_module("fmrimod.contrast")
    assert isinstance(submodule, ModuleType), (
        "importlib.import_module('fmrimod.contrast') must return the "
        f"submodule, got {type(submodule)!r}."
    )
    assert "group_dataset_from_contrasts" in getattr(submodule, "__all__", ()), (
        "group_dataset_from_contrasts must stay in fmrimod.contrast "
        "submodule __all__ so it is discoverable via dir()/autosummary; "
        "the tutorial relies on this."
    )


def test_top_level_import_is_pinned_unavailable() -> None:
    """``from fmrimod import group_dataset_from_contrasts`` must fail.

    Pinned as known-unavailable: the tutorial prose explicitly tells
    readers to use the submodule form. If the helper is ever promoted
    to the top level, this assertion fails and the tutorial prose must
    be revised in the same commit (bd-01KRRM5CY3J08X8N4V809XWTB9).
    """
    with pytest.raises(ImportError):
        from fmrimod import group_dataset_from_contrasts  # noqa: F401


def test_callable_shadow_blocks_attribute_access() -> None:
    """``import fmrimod.contrast as c`` binds the callable, so
    ``c.group_dataset_from_contrasts`` is *not* available — the
    misleading form the tutorial warns against."""
    import fmrimod

    contrast_attr = fmrimod.contrast
    assert callable(contrast_attr), (
        "fmrimod.contrast is a ratified rebind; the attribute must be "
        f"the callable, got {type(contrast_attr)!r}."
    )
    assert not hasattr(contrast_attr, "group_dataset_from_contrasts"), (
        "the callable rebind must NOT expose group_dataset_from_contrasts "
        "as an attribute; this is exactly the trap the tutorial documents."
    )
