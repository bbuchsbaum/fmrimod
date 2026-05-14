"""Executable artifact for ``docs/contracts/parametric_contrast_sugar_v1.md``.

These tests document the v1 call shapes agreed (or under vetting) on the
``parametric-contrast-sugar-vetting`` board thread. They skip until the
``fmrimod.contrast.modulator`` and
``fmrimod.contrast.group_dataset_from_contrasts`` surfaces exist, so the
file is safe to land before the implementation slice. They flip from
skipped to required as the implementation closes — that flip is the
mechanical signal that the bead's red check has been touched.

Owning beads:

- ``bd-01KRM9PVWWKTH7A0TJDYTZ9XB7`` — consolidated red check.
- ``bd-01KRM5FRQP16S05T3DA57DQY31`` — broader parametric semantic sugar gap.

Do *not* convert these to passing stubs by stubbing out ``modulator`` or
``group_dataset_from_contrasts`` to satisfy the import. The skip is the
intended state until the actual implementation lands.
"""

from __future__ import annotations

import importlib

import pytest


def _try_import(name: str) -> object | None:
    module = importlib.import_module("fmrimod.contrast")
    return getattr(module, name, None)


def test_modulator_v1_call_shape_compiles() -> None:
    """v1 t-contrast spelling: modulator(...).within(...).slope(level)."""

    modulator = _try_import("modulator")
    if modulator is None:
        pytest.skip(
            "fmrimod.contrast.modulator not yet implemented; "
            "see docs/contracts/parametric_contrast_sugar_v1.md"
        )

    rt = modulator("rt_z").within("trial_type")
    spec = rt.slope("word") - rt.slope("pseudoword")
    assert spec is not None


def test_modulator_rejects_missing_modulator_with_actionable_message() -> None:
    """Error policy from Q3: headline names the missing identifier."""

    modulator = _try_import("modulator")
    if modulator is None:
        pytest.skip(
            "fmrimod.contrast.modulator not yet implemented; "
            "see docs/contracts/parametric_contrast_sugar_v1.md"
        )

    with pytest.raises((KeyError, ValueError)) as excinfo:
        modulator("not_a_real_column").within("trial_type").slope("word")
    message = str(excinfo.value)
    assert "not_a_real_column" in message


def test_modulator_f_contrast_v1_raises_with_contract_pointer() -> None:
    """v1 ships t-only; F-contrast deferred to v2 with a clear error."""

    modulator = _try_import("modulator")
    if modulator is None:
        pytest.skip(
            "fmrimod.contrast.modulator not yet implemented; "
            "see docs/contracts/parametric_contrast_sugar_v1.md"
        )

    rt = modulator("rt_z").within("trial_type")
    omnibus = getattr(rt, "slopes", None) or getattr(rt, "omnibus", None)
    if omnibus is None:
        pytest.skip(
            "F-contrast spelling deferred to v2 per "
            "docs/contracts/parametric_contrast_sugar_v1.md"
        )
    with pytest.raises(NotImplementedError):
        omnibus("word", "pseudoword", "neutral")


def test_group_dataset_from_contrasts_v1_call_shape_compiles() -> None:
    """v1 single-contrast-per-subject dict constructor."""

    factory = _try_import("group_dataset_from_contrasts")
    if factory is None:
        pytest.skip(
            "fmrimod.contrast.group_dataset_from_contrasts not yet "
            "implemented; see docs/contracts/parametric_contrast_sugar_v1.md"
        )
    assert callable(factory)


def test_group_dataset_from_contrasts_preserves_provenance_fields() -> None:
    """Q5 metadata contract: required ContrastResult fields survive lowering."""

    factory = _try_import("group_dataset_from_contrasts")
    if factory is None:
        pytest.skip(
            "fmrimod.contrast.group_dataset_from_contrasts not yet "
            "implemented; see docs/contracts/parametric_contrast_sugar_v1.md"
        )

    pytest.skip(
        "Concrete provenance round-trip assertions require the "
        "implementation slice; this test exists to be filled in as the "
        "v1 implementation lands."
    )


def test_low_level_condition_escape_hatch_remains_valid() -> None:
    """The sugar must be additive: condition(..., term=...) still works."""

    from fmrimod.contrast import condition

    word = condition("word", term="trial_type:rt_z")
    pseudoword = condition("pseudoword", term="trial_type:rt_z")
    spec = word - pseudoword
    assert spec is not None
