"""Test module-level exports and access patterns."""

import types


def test_events_generic_accessible():
    """events() generic is accessible from fmrimod.utils."""
    from fmrimod.utils.generics import events
    assert callable(events)


def test_events_subpackage_accessible():
    """pyfmridesign.events subpackage is still importable."""
    from fmrimod import events
    # This should be the subpackage (module), not the function
    assert isinstance(events, types.ModuleType)


def test_events_via_utils():
    """events() generic is accessible via pyfmridesign.utils import."""
    from fmrimod.utils import events
    assert callable(events)


def test_hrf_exported():
    """hrf() formula function is exported at top level as hrf_formula."""
    from fmrimod import hrf_formula
    assert callable(hrf_formula)


def test_hrf_spmg1_exported():
    """hrf_spmg1() partial is exported at top level."""
    from fmrimod import hrf_spmg1
    assert callable(hrf_spmg1)


def test_hrf_spmg1_is_partial():
    """hrf_spmg1 is a partial of hrf with spec='spmg1'."""
    from fmrimod import hrf_spmg1
    from functools import partial
    assert isinstance(hrf_spmg1, partial)
    assert hrf_spmg1.keywords.get('spec') == 'spmg1'
