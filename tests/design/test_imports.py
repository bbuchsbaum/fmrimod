"""Test basic imports and re-exports from fmrimod."""

import importlib
import sys


def test_package_imports():
    """Test that the package can be imported."""
    import fmrimod
    assert hasattr(fmrimod, '__version__')
    assert fmrimod.__version__ == '0.1.0'


def test_pyfmrihrf_reexports():
    """Test that fmrimod HRF functions are properly exported."""
    # Test core HRF exports
    from fmrimod import (
        SamplingFrame,
        gen_hrf,
        regressor,
    )

    # Core exports must be callable
    assert callable(gen_hrf)
    assert callable(regressor)

    # SamplingFrame should be a class
    assert callable(SamplingFrame)

    # Test pre-defined HRFs
    from fmrimod import SPM_CANONICAL
    assert hasattr(SPM_CANONICAL, 'evaluate')


def test_submodule_imports():
    """Test that submodules can be imported."""
    import fmrimod.baseline
    import fmrimod.basis
    import fmrimod.design
    import fmrimod.events
    import fmrimod.formula
    import fmrimod.hrf
    import fmrimod.io
    import fmrimod.utils

    # All should have __all__ attribute
    for module in [
        fmrimod.design,
        fmrimod.events,
        fmrimod.basis,
        fmrimod.formula,
        fmrimod.utils,
        fmrimod.io,
        fmrimod.hrf,
        fmrimod.baseline,
    ]:
        assert hasattr(module, '__all__')


def test_module_import_order_robustness():
    """Test import-ordering that previously triggered circular-import crashes."""
    modules_to_clear = [
        name
        for name in list(sys.modules)
        if name == "fmrimod"
        or name.startswith("fmrimod.")
    ]
    for module_name in modules_to_clear:
        del sys.modules[module_name]

    importlib.import_module("fmrimod.utils.misc")
    importlib.import_module("fmrimod.regressor")
