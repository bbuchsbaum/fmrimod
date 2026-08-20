"""Wave 2: PrewhitenConfig must forward ACVF bias-correction controls.

Guards fmrilss ``f851fb0`` / fmriAR 0.3.3 residual-bias correction.
Cheap pass disqualified: storing ``design`` / ``acvf_correction`` on
the config without forwarding them to ``fit_noise``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.ar.acvf import acvf_bias_matrix
from fmrimod.ar.estimation import fit_noise
from fmrimod.single import estimate_single_trial
from fmrimod.single._prewhiten import (
    PrewhitenConfig,
    _residuals_for_ar_estimation,
    prewhiten_matrices,
)


def _ar1_resid(n: int, v: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    eps = rng.standard_normal((n, v))
    out = np.empty_like(eps)
    out[0] = eps[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def _fat_design(n_time: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(n_time),
            np.polynomial.polynomial.polyvander(np.linspace(-1, 1, n_time), 3)[:, 1:],
            rng.standard_normal((n_time, 12)),
        ]
    )


def test_acvf_correction_changes_phi_on_fat_design() -> None:
    """Forwarded correction must move φ relative to the uncorrected fit.

    Cheap pass: field stored, not forwarded — both plans would share φ.
    """
    rng = np.random.default_rng(1515)
    n_time, n_voxels = 120, 6
    X = _fat_design(n_time, rng)
    Y = _ar1_resid(n_time, n_voxels, 0.5, rng)

    uncorrected = prewhiten_matrices(Y, X, None, PrewhitenConfig(method="ar", p=1))
    correction = acvf_bias_matrix(X, max_lag=8)
    corrected = prewhiten_matrices(
        Y,
        X,
        None,
        PrewhitenConfig(method="ar", p=1, acvf_correction=correction),
    )

    phi_u = np.asarray(uncorrected.plan.phi[0], dtype=np.float64)
    phi_c = np.asarray(corrected.plan.phi[0], dtype=np.float64)
    assert phi_u.shape == phi_c.shape
    assert np.max(np.abs(phi_c - phi_u)) > 1e-6

    resid_pw = _residuals_for_ar_estimation(Y, X)
    expected = fit_noise(
        resid=resid_pw,
        method="ar",
        p=1,
        exact_first="ar1",
        pooling="global",
        acvf_correction=correction,
    )
    assert_allclose(phi_c, np.asarray(expected.phi[0]), atol=0, rtol=0)


def test_design_kwarg_is_forwarded_like_acvf_correction() -> None:
    rng = np.random.default_rng(1516)
    n_time, n_voxels = 100, 4
    X = _fat_design(n_time, rng)
    Y = _ar1_resid(n_time, n_voxels, 0.45, rng)

    via_design = prewhiten_matrices(
        Y,
        X,
        None,
        PrewhitenConfig(method="ar", p=1, design=X, correction_max_lag=8),
    )
    via_cached = prewhiten_matrices(
        Y,
        X,
        None,
        PrewhitenConfig(
            method="ar",
            p=1,
            acvf_correction=acvf_bias_matrix(X, max_lag=8),
        ),
    )
    assert_allclose(
        np.asarray(via_design.plan.phi[0]),
        np.asarray(via_cached.plan.phi[0]),
        atol=1e-12,
    )


def test_whitening_plan_survives_new_prewhiten_options() -> None:
    rng = np.random.default_rng(20260514)
    n, n_trials, v = 80, 6, 8
    X = rng.standard_normal((n, n_trials))
    Y = rng.standard_normal((n, v))
    cfg = PrewhitenConfig(
        method="ar",
        p=1,
        design=np.column_stack([np.ones(n), X]),
        correction_max_lag=8,
    )
    result = estimate_single_trial(Y, X, method="lss", prewhiten=cfg)
    assert result.extra.whitening_plan is not None
    assert result.extra.whitening_plan.order == (1, 0)


def test_prewhiten_config_defaults_match_r() -> None:
    cfg = PrewhitenConfig()
    assert cfg.design is None
    assert cfg.acvf_correction is None
    assert cfg.correction_max_lag == 25


def test_prewhiten_config_rejects_nonsquare_correction() -> None:
    with pytest.raises(ValueError, match="square"):
        PrewhitenConfig(acvf_correction=np.ones((2, 3)))
