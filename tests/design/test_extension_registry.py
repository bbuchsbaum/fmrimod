"""Python ports of the fmridesign HRF extension-registry contracts."""

from __future__ import annotations

from datetime import datetime

import pytest

from fmrimod import extension_registry as reg


@pytest.fixture(autouse=True)
def isolated_registry():
    saved = dict(reg._registry)
    reg._registry.clear()
    try:
        yield
    finally:
        reg._registry.clear()
        reg._registry.update(saved)


def test_register_hrfspec_extension_registers_class():
    result = reg.register_hrfspec_extension(
        spec_class="test_hrfspec_A",
        package="testPkg",
    )

    assert result is None
    assert "test_hrfspec_A" in reg.list_external_hrfspecs()


def test_register_hrfspec_extension_stores_all_fields():
    reg.register_hrfspec_extension(
        spec_class="test_hrfspec_B",
        package="testPkg",
        convolved_class="test_convolved_B",
        requires_external_processing=True,
        formula_functions=["test_hrf_B", "test_trial_B"],
    )

    info = reg.get_external_hrfspec_info("test_hrfspec_B")
    assert info is not None
    assert info.spec_class == "test_hrfspec_B"
    assert info.package == "testPkg"
    assert info.convolved_class == "test_convolved_B"
    assert info.requires_external_processing is True
    assert info.formula_functions == ["test_hrf_B", "test_trial_B"]
    assert isinstance(info.registered_at, datetime)


@pytest.mark.parametrize("spec_class", [123, "", ["a", "b"]])
def test_register_hrfspec_extension_validates_spec_class(spec_class):
    with pytest.raises(TypeError, match="spec_class must be a single"):
        reg.register_hrfspec_extension(spec_class=spec_class, package="pkg")


def test_register_hrfspec_extension_validates_package():
    with pytest.raises(TypeError, match="package must be a single"):
        reg.register_hrfspec_extension(spec_class="cls", package=42)


@pytest.mark.parametrize("formula_functions", [42, ["ok", 1], [""]])
def test_register_hrfspec_extension_validates_formula_functions(formula_functions):
    with pytest.raises(TypeError, match="formula_functions must be a character vector"):
        reg.register_hrfspec_extension(
            spec_class="cls",
            package="pkg",
            formula_functions=formula_functions,
        )


def test_register_hrfspec_extension_accepts_none_formula_functions():
    reg.register_hrfspec_extension(
        spec_class="test_hrfspec_C",
        package="testPkg",
        formula_functions=None,
    )

    assert reg.get_external_hrfspec_info("test_hrfspec_C").formula_functions is None


def test_is_external_hrfspec_detects_registered_class_by_name_and_object():
    class OtherClass:
        pass

    class TestHrfspecD(OtherClass):
        __fmrimod_hrfspec_classes__ = ["other_class", "test_hrfspec_D"]

    assert reg.is_external_hrfspec("test_hrfspec_D") is False

    reg.register_hrfspec_extension("test_hrfspec_D", "testPkg")

    assert reg.is_external_hrfspec("test_hrfspec_D") is True
    assert reg.is_external_hrfspec(TestHrfspecD()) is True
    assert reg.is_external_hrfspec("totally_unregistered_class") is False
    assert reg.is_external_hrfspec(OtherClass()) is False


def test_get_external_hrfspec_info_returns_registered_info_or_none():
    assert reg.get_external_hrfspec_info("nonexistent_class") is None

    reg.register_hrfspec_extension(
        spec_class="test_hrfspec_G",
        package="testPkg",
        requires_external_processing=True,
    )

    info = reg.get_external_hrfspec_info("test_hrfspec_G")
    assert info is not None
    assert info.spec_class == "test_hrfspec_G"
    assert info.package == "testPkg"
    assert info.requires_external_processing is True


def test_requires_external_processing_resolves_object_classes():
    class TestHrfspecH:
        __fmrimod_hrfspec_classes__ = ["test_hrfspec_H", "list"]

    class TestHrfspecI:
        __fmrimod_hrfspec_classes__ = ["test_hrfspec_I", "list"]

    reg.register_hrfspec_extension(
        "test_hrfspec_H",
        "testPkg",
        requires_external_processing=True,
    )
    reg.register_hrfspec_extension(
        "test_hrfspec_I",
        "testPkg",
        requires_external_processing=False,
    )

    assert reg.requires_external_processing(TestHrfspecH()) is True
    assert reg.requires_external_processing(TestHrfspecI()) is False
    assert reg.requires_external_processing("unregistered") is False


def test_requires_external_processing_accepts_registered_class_names():
    """Python accepts explicit class-name strings as first-class registry keys."""
    reg.register_hrfspec_extension(
        "test_hrfspec_J",
        "testPkg",
        requires_external_processing=True,
    )

    assert reg.requires_external_processing("test_hrfspec_J") is True
    assert reg.requires_external_processing("unregistered") is False


def test_get_external_hrfspec_functions_supports_registered_none_and_afni_fallback():
    reg.register_hrfspec_extension(
        "test_hrfspec_K",
        "testPkg",
        formula_functions=["my_hrf", "my_trialwise"],
    )
    reg.register_hrfspec_extension(
        "test_hrfspec_L",
        "testPkg",
        formula_functions=None,
    )
    reg.register_hrfspec_extension(
        "afni_hrfspec",
        "afnireg",
        formula_functions=None,
    )

    assert reg.get_external_hrfspec_functions("test_hrfspec_K") == [
        "my_hrf",
        "my_trialwise",
    ]
    assert reg.get_external_hrfspec_functions("test_hrfspec_L") is None
    assert reg.get_external_hrfspec_functions("nonexistent_class") is None
    assert reg.get_external_hrfspec_functions("afni_hrfspec") == ["afni_hrf"]


def test_get_all_external_hrf_functions_aggregates_unique_names():
    reg.register_hrfspec_extension(
        "test_hrfspec_M",
        "testPkgM",
        formula_functions=["hrf_m1", "shared_hrf"],
    )
    reg.register_hrfspec_extension(
        "test_hrfspec_N",
        "testPkgN",
        formula_functions=["shared_hrf", "hrf_n1"],
    )

    all_funcs = reg.get_all_external_hrf_functions()
    assert all_funcs == ["hrf_m1", "shared_hrf", "hrf_n1"]
    assert all_funcs.count("shared_hrf") == 1


def test_list_external_hrfspecs_returns_registered_classes():
    reg.register_hrfspec_extension("test_hrfspec_Q", "testPkgQ")
    reg.register_hrfspec_extension("test_hrfspec_R", "testPkgR")

    listed = reg.list_external_hrfspecs()
    assert isinstance(listed, list)
    assert "test_hrfspec_Q" in listed
    assert "test_hrfspec_R" in listed
