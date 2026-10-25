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

    # Two points cannot determine the quadratic, but every basis column is kept.
    assert out.shape == (2, basis.n_basis) == (2, 3)


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


def _cm_events():
    return pd.DataFrame({"onset": [2.0, 20.0, 40.0, 60.0], "cond": ["A", "B", "A", "B"]})


def test_condition_map_multibasis_expand_basis_joins_columns():
    """Inventory v2 DES-13 (R 2368564/277ea71): join on condition and basis."""
    import fmrimod

    em = fmrimod.event_model(
        "onset ~ hrf(cond, basis='spmg3')", data=_cm_events(), tr=2.0, n_scans=50
    )
    cmap = fmrimod.condition_map(em, expand_basis=True)
    assert len(cmap) == 6
    assert list(cmap["canonical"]) == [
        f"cond.{lev}_b0{b}" for lev in "AB" for b in (1, 2, 3)
    ]
    assert list(cmap["display"]) == [f"{lev}_b0{b}" for lev in "AB" for b in (1, 2, 3)]
    assert list(cmap["column_name"]) == [f"cond_{c}" for c in cmap["canonical"]]
    assert list(cmap["column_name"]) == em.column_names

    # Without expand_basis a condition has no single column.
    plain = fmrimod.condition_map(em)
    assert len(plain) == 2
    assert plain["column_name"].isna().all()


def test_condition_map_repeated_term_name_uses_its_own_columns():
    import fmrimod

    em = fmrimod.event_model(
        "onset ~ hrf(cond) + hrf(cond, basis='spmg2')",
        data=_cm_events(), tr=2.0, n_scans=50,
    )
    cmap = fmrimod.condition_map(em, expand_basis=True)
    assert not cmap["column_name"].isna().any()
    assert sorted(cmap["column_name"]) == sorted(em.column_names)
    assert list(cmap["term"]) == ["cond"] * 2 + ["cond#1"] * 4
