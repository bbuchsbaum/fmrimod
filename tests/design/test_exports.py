"""Test module-level exports and access patterns."""

import types

import numpy as np
import pandas as pd


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


def test_fmridesign_top_level_contrast_exports():
    """F-contrast helpers documented in the migration guide are top-level."""
    import fmrimod
    from fmrimod.events import EventFactor

    event = EventFactor(
        name="condition",
        onsets=[0.0, 4.0, 8.0, 12.0],
        values=["A", "B", "A", "B"],
        levels=["A", "B"],
    )

    assert callable(fmrimod.Fcontrasts)
    assert callable(fmrimod.contrast_weights)
    assert "condition" in fmrimod.Fcontrasts(event)


def test_fmridesign_top_level_sampling_generics():
    """samples/global_onsets are available as R-parity convenience helpers."""
    import fmrimod

    sframe = fmrimod.SamplingFrame(blocklens=[2, 2], tr=2.0)

    np.testing.assert_allclose(fmrimod.samples(sframe), [1.0, 3.0, 5.0, 7.0])
    np.testing.assert_allclose(
        fmrimod.samples(sframe, global_time=False),
        [1.0, 3.0, 1.0, 3.0],
    )
    np.testing.assert_allclose(
        fmrimod.global_onsets(sframe, [0.5, 1.5], [0, 1]),
        [0.5, 5.5],
    )


def test_fmridesign_top_level_evaluate_helper():
    """evaluate delegates to objects with an evaluate method."""
    import fmrimod

    basis = fmrimod.Poly(degree=2)
    out = fmrimod.evaluate(basis, np.array([0.0, 1.0]))

    assert out.shape == (2, 2)


def test_condition_map_event_model_top_level():
    """condition_map exposes display/canonical names and model columns."""
    import fmrimod

    events = pd.DataFrame({
        "onset": [0.0, 4.0, 8.0, 12.0],
        "condition": pd.Categorical(["A", "B", "A", "B"]),
        "duration": [1.0, 1.0, 1.0, 1.0],
    })
    sframe = fmrimod.SamplingFrame(blocklens=20, tr=1.0)
    model = fmrimod.event_model(
        "onset ~ condition",
        data=events,
        sampling_frame=sframe,
        durations="duration",
    )

    cmap = fmrimod.condition_map(model)

    assert list(cmap.columns) == ["term", "display", "canonical", "column_name"]
    assert set(cmap["display"]) == {"A", "B"}
    assert set(cmap["canonical"]) == {"condition.A", "condition.B"}
    assert set(cmap["column_name"]) == set(model.column_names)
