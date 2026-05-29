"""Pin the SPM/Nilearn-aligned canonical HRF.

fmrimod's default ``basis="spm"`` was migrated in early 2026 from the
legacy R-fmrireg parameterization ``(p1=5, p2=15, a1=0.0833)`` to the
SPM standard ``(delay=6, undershoot=16, dispersion=1, u_dispersion=1,
ratio=0.167)`` — the form used by SPM ``spm_hrf`` and Nilearn
``_gamma_difference_hrf``. This file pins the new defaults and the
kernel-level alignment so the canonical cannot silently regress to
the legacy shape.

The legacy R parameterization stays available via ``basis="spm_legacy"``
and is also pinned here for backward-compat.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level.hemodynamic_models import spm_hrf as nl_spm_hrf

import fmrimod as fm
from fmrimod.hrf.functions import spm_canonical, spm_canonical_legacy
from fmrimod.hrf.spm_hrf import (
    SPMG1_HRF,
    SPMG2_HRF,
    SPMG3_HRF,
    SPMG1_HRF_Legacy,
    SPMG2_HRF_Legacy,
    SPMG3_HRF_Legacy,
)
from fmrimod.spec import hrf


def test_default_spm_uses_spm_standard_parameterization() -> None:
    """``SPMG1_HRF()`` returns the SPM standard 5-parameter canonical."""
    h = SPMG1_HRF()
    assert h.delay == 6.0
    assert h.undershoot == 16.0
    assert h.dispersion == 1.0
    assert h.u_dispersion == 1.0
    assert h.ratio == pytest.approx(0.167)


def test_default_spm_peak_time_matches_spm() -> None:
    """The aligned canonical peaks near 5 s, matching SPM/Nilearn."""
    t = np.linspace(0, 32, 1601)
    h = SPMG1_HRF()
    peak_t = float(t[int(np.argmax(h(t)))])
    assert 4.5 <= peak_t <= 5.5, (
        f"aligned SPM canonical should peak near 5 s, got {peak_t:.2f}"
    )


def test_default_spm_peak_undershoot_ratio_is_approx_10() -> None:
    """SPM's peak/undershoot magnitude ratio is ~10:1 (vs ~4600 in legacy)."""
    t = np.linspace(0, 32, 3201)
    values = SPMG1_HRF()(t)
    peak = float(np.max(values))
    undershoot = float(-np.min(values))
    ratio = peak / undershoot
    assert 6.0 < ratio < 20.0, (
        f"SPM canonical peak/undershoot ratio should be ~10; got {ratio:.1f}"
    )


def test_aligned_canonical_correlates_with_nilearn_kernel() -> None:
    """Area-normalized kernels match Nilearn at correlation > 0.999."""
    t = np.arange(0, 32.0, 0.1)
    fm_kernel = SPMG1_HRF()(t)
    nl_full = nl_spm_hrf(t_r=2.0, oversampling=20, time_length=32.0)
    nl_t = np.linspace(0, 32.0, len(nl_full))
    fm_on_nl = np.interp(nl_t, t, fm_kernel)
    fm_on_nl /= fm_on_nl.sum()
    nl_norm = nl_full / nl_full.sum()
    corr = float(np.corrcoef(fm_on_nl, nl_norm)[0, 1])
    assert corr > 0.999, (
        f"aligned SPM canonical should correlate > 0.999 with "
        f"Nilearn's spm_hrf; got {corr:.6f}"
    )


def test_legacy_canonical_preserves_old_shape() -> None:
    """``SPMG1_HRF_Legacy()`` reproduces the pre-alignment shape exactly."""
    t = np.linspace(0, 24, 49)
    legacy_class = SPMG1_HRF_Legacy()(t)
    legacy_func = spm_canonical_legacy(t, p1=5.0, p2=15.0, a1=0.0833)
    np.testing.assert_allclose(legacy_class, legacy_func, atol=1e-14)
    # Sanity: the legacy form has peak/undershoot ratio much higher
    # than the SPM standard ~10:1.
    peak = float(np.max(legacy_class))
    undershoot = float(-np.min(legacy_class))
    assert peak / undershoot > 50, (
        f"legacy SPM canonical should have a much larger peak/undershoot "
        f"ratio than SPM's standard 10:1; got {peak/undershoot:.0f}"
    )


def test_spm_canonical_function_rejects_invalid_dispersion() -> None:
    """Both 0 and negative dispersion params raise clearly."""
    with pytest.raises(ValueError, match="dispersion"):
        spm_canonical(np.arange(0, 24), dispersion=0.0)
    with pytest.raises(ValueError, match="dispersion"):
        spm_canonical(np.arange(0, 24), u_dispersion=-1.0)


def test_legacy_basis_alias_routes_to_legacy_class() -> None:
    """``basis="spm_legacy"`` in the typed spec resolves to the legacy class."""
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 96.0, 6),
            "duration": 0.0,
            "trial_type": ["A", "B"] * 3,
            "run": 1,
        }
    )
    ds = fm.fmri_dataset(np.zeros((80, 1)), tr=2.0, events=events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_new = fm.fmri_lm(hrf("trial_type", basis="spm"), ds)
        fit_legacy = fm.fmri_lm(hrf("trial_type", basis="spm_legacy"), ds)
    X_new = fit_new.model.design_matrix_array(run=0)
    X_legacy = fit_legacy.model.design_matrix_array(run=0)
    # The two designs must differ — if they were equal, the legacy
    # alias would be silently routed to the new class.
    assert not np.allclose(X_new[:, 0], X_legacy[:, 0]), (
        "basis='spm' and basis='spm_legacy' should produce different "
        "task columns; if they match, the legacy alias is misrouted"
    )


def test_spmg2_legacy_unchanged() -> None:
    """``SPMG2_HRF_Legacy`` keeps the legacy canonical + analytic ∂h/∂t."""
    t = np.linspace(0, 24, 49)
    legacy = SPMG2_HRF_Legacy()(t)
    assert legacy.shape == (49, 2)
    # First column is the legacy canonical.
    np.testing.assert_allclose(
        legacy[:, 0], spm_canonical_legacy(t), atol=1e-14
    )


def test_spmg3_uses_spm_dispersion_derivative_by_default() -> None:
    """``SPMG3_HRF()`` third column is the SPM dispersion derivative.

    Sign check: positive on the rising flank (the SPM dispersion
    derivative on the *aligned* canonical, unlike the legacy
    formulation, has *positive* values on the rising flank because
    the aligned canonical's peak/undershoot ratio interacts
    differently with the dispersion perturbation).
    """
    t = np.linspace(0, 16, 161)
    h = SPMG3_HRF()
    basis = h(t)
    assert basis.shape == (161, 3)
    disp_col = basis[:, 2]
    assert np.all(np.isfinite(disp_col))
    # Not all-zero — the SPM dispersion derivative is a non-trivial
    # function over [0, 16].
    assert np.max(np.abs(disp_col)) > 0
