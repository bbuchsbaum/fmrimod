"""Test basic imports and re-exports from fmrimod."""

import pytest


def test_package_imports():
    """Test that the package can be imported."""
    import fmrimod
    assert hasattr(fmrimod, '__version__')
    assert fmrimod.__version__ == '0.1.0'


def test_pyfmrihrf_reexports():
    """Test that fmrimod HRF functions are properly exported."""
    import fmrimod

    # Test core HRF exports
    from fmrimod import (
        HRF,
        gen_hrf,
        regressor,
        SamplingFrame,
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
    import fmrimod.design
    import fmrimod.events
    import fmrimod.basis
    import fmrimod.formula
    import fmrimod.utils
    import fmrimod.io
    import fmrimod.hrf
    import fmrimod.baseline

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