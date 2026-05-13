"""Tests for HRF normalization modes (``hrf_norm`` / ``norm=``)."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.hrf import HRF_SPMG1, HRF_SPMG2, normalize


def test_spm_norm_yields_unit_integral_on_reference_grid():
    """`mode='spm'` makes the HRF sum to 1 on Nilearn's reference grid."""
    h = normalize(HRF_SPMG1, "spm")
    # Nilearn uses linspace(0, time_length, time_length / dt), including the
    # endpoint, which gives 1600 samples for time_length=32 and oversampling=50.
    t = np.linspace(0.0, 32.0, 1600)
    vals = h(t)
    assert vals.sum() == pytest.approx(1.0, rel=1e-12)


def test_spm_norm_matches_nilearn_reference_length_and_sum():
    nilearn_hemodynamic = pytest.importorskip(
        "nilearn.glm.first_level.hemodynamic_models"
    )

    h = normalize(HRF_SPMG1, "spm")
    t = np.linspace(0.0, 32.0, 1600)
    vals = h(t)
    nilearn_vals = nilearn_hemodynamic.spm_hrf(
        1.0, oversampling=50, time_length=32.0
    )

    assert len(vals) == len(nilearn_vals)
    assert vals.sum() == pytest.approx(nilearn_vals.sum(), rel=1e-12)


def test_unit_peak_norm_yields_peak_one():
    h = normalize(HRF_SPMG1, "unit_peak")
    t = np.linspace(0.0, 32.0, 1601)
    vals = h(t)
    assert np.max(np.abs(vals)) == pytest.approx(1.0, rel=1e-3)


def test_unit_integral_norm_yields_continuous_integral_one():
    h = normalize(HRF_SPMG1, "unit_integral")
    t = np.linspace(0.0, 32.0, 1601)
    vals = h(t)
    assert np.trapz(vals, t) == pytest.approx(1.0, rel=1e-3)


def test_normalize_is_grid_independent_scale_factor():
    """Normalization factor is fixed across evaluation grids."""
    h = normalize(HRF_SPMG1, "spm")
    # Compare a non-zero sample on two different grids.
    t1 = np.array([6.0])
    t2 = np.array([6.0, 8.0, 10.0])
    factor_1 = HRF_SPMG1(t1)[0] / h(t1)[0]
    factor_2 = HRF_SPMG1(t2)[0] / h(t2)[0]
    assert factor_1 == pytest.approx(factor_2, rel=1e-12)


def test_normalize_preserves_multibasis_shape():
    h = normalize(HRF_SPMG2, "spm")
    t = np.linspace(0.0, 32.0, 161)
    vals = h(t)
    assert vals.shape == (161, 2)


def test_normalize_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown HRF normalization"):
        normalize(HRF_SPMG1, "bogus")  # type: ignore[arg-type]


def test_hrf_builder_accepts_norm_kwarg():
    """The ``fmrimod.spec.hrf`` builder propagates ``norm`` into HrfTerm."""
    from fmrimod.spec import hrf

    term = hrf("trial_type", norm="spm")
    assert term.norm == "spm"
    assert term.variables == ("trial_type",)


def test_hrf_builder_rejects_unknown_norm_kwarg():
    from fmrimod.spec import hrf

    with pytest.raises(ValueError, match="Unknown HRF normalization mode"):
        hrf("trial_type", norm="bogus")  # type: ignore[arg-type]


def test_spec_compile_applies_norm_via_registry():
    """End-to-end: ``hrf("var", norm='spm')`` produces a design column at the
    Nilearn scale (sum ≈ 1 over the HRF support after convolution with a unit
    impulse)."""
    import neuroim
    import pandas as pd

    import fmrimod as fm
    from fmrimod.spec import hrf

    rng = np.random.default_rng(0)
    n_t = 60
    spatial = (1, 1, 1)
    space = neuroim.NeuroSpace(
        dim=spatial + (n_t,), spacing=(2, 2, 2, 1), origin=(0, 0, 0, 0)
    )
    data = rng.standard_normal(spatial + (n_t,))
    vec = neuroim.DenseNeuroVec(data, space)
    mask = np.ones(spatial, dtype=bool)
    events = pd.DataFrame({
        "onset": [10.0, 50.0],
        "duration": [0.0, 0.0],
        "trial_type": ["A", "B"],
    })
    ds = fm.fmri_dataset(vec, mask=mask, tr=2.0, events=events)

    fit_norm = fm.fmri_lm(hrf("trial_type", norm="spm"), ds, precision=0.1)
    fit_raw = fm.fmri_lm(hrf("trial_type"), ds, precision=0.1)

    col_norm = fit_norm.model.event_model.design_matrix[:, 0]
    col_raw = fit_raw.model.event_model.design_matrix[:, 0]
    # Normalized column should be ~2-3 orders of magnitude smaller than the
    # raw canonical (factor ~0.002 — matches Nilearn).
    scale = col_norm.max() / col_raw.max()
    assert scale < 0.01
    assert scale > 1e-4
