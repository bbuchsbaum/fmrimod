"""Pins the import contract the golden-path tutorial teaches.

``docs/tutorials/golden-path.qmd`` walks the full typed workflow
``fmri_dataset -> fmri_lm -> contrast -> group_dataset_from_contrasts
-> ols_voxelwise``. The first-level -> group helper
``group_dataset_from_contrasts`` is part of the top-level public
surface, so the tutorial teaches the obvious form:

    from fmrimod import group_dataset_from_contrasts

It also remains importable from the ``fmrimod.contrast`` submodule for
back-compat. This test pins both so the tutorial cannot drift out of
sync with the code. (Previously the helper was submodule-only and the
tutorial had to warn readers about the ``fmrimod.contrast``
callable/submodule shadow; that paragraph was deleted when the helper
was promoted — bd-01KRRNT5PQCYKT61F6EVNWQ95G.)

Refs: bd-01KRRNT5PQCYKT61F6EVNWQ95G (top-level promotion),
bd-01KRRM5Y8B60G60KFBJ727FECS (the golden-path page that depends on
it), board tutorials-win-or-lose/post-01KRRM2J2ZREJSTBQHQPP1HT2E.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def test_top_level_import_is_the_taught_form() -> None:
    """``from fmrimod import group_dataset_from_contrasts`` must work.

    This is the exact line the golden-path tutorial teaches. If the
    promotion is ever reverted, this fails loudly and forces the
    tutorial to be revised in the same commit.
    """
    from fmrimod import group_dataset_from_contrasts

    assert callable(group_dataset_from_contrasts), (
        "the golden-path tutorial teaches `from fmrimod import "
        "group_dataset_from_contrasts`; it must resolve to a callable."
    )


def test_helper_is_in_top_level_all() -> None:
    """The helper is an advertised part of the public surface."""
    import fmrimod

    assert "group_dataset_from_contrasts" in fmrimod.__all__, (
        "group_dataset_from_contrasts must be in fmrimod.__all__ so it "
        "is discoverable via dir()/autosummary; the tutorial relies on it."
    )


def test_submodule_form_still_works_and_is_the_same_object() -> None:
    """Back-compat: ``from fmrimod.contrast import …`` still works and
    resolves to the *same* object as the top-level name."""
    from fmrimod import group_dataset_from_contrasts as top_level
    from fmrimod.contrast import group_dataset_from_contrasts as submodule

    assert submodule is top_level, (
        "the top-level and fmrimod.contrast forms must be the same "
        "object; a divergent re-export would be a silent fork."
    )

    module = importlib.import_module("fmrimod.contrast")
    assert isinstance(module, ModuleType)
    assert "group_dataset_from_contrasts" in getattr(module, "__all__", ()), (
        "group_dataset_from_contrasts must stay in the fmrimod.contrast "
        "submodule __all__ for back-compat discoverability."
    )
