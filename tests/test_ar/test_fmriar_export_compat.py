"""Top-level fmriAR export compatibility checks."""

import numpy as np

import fmrimod
import fmrimod.ar as ar


FMRIAR_EXPORTS = [
    "fit_noise",
    "whiten",
    "whiten_apply",
    "acorr_diagnostics",
    "sandwich_from_whitened_resid",
    "afni_restricted_plan",
    "compat",
]


def test_fmriar_public_exports_exist_top_level_and_ar_namespace():
    for name in FMRIAR_EXPORTS:
        assert hasattr(fmrimod, name), name
        assert name in fmrimod.__all__, name
        assert hasattr(ar, name), name


def test_top_level_fit_noise_and_whiten_apply_delegate_to_ar_namespace():
    rng = np.random.default_rng(42)
    x = np.column_stack([np.ones(60), rng.normal(size=60)])
    beta = np.array([[1.0, -0.5], [0.25, 0.5]])
    y = x @ beta + rng.normal(size=(60, 2))

    plan = fmrimod.fit_noise(Y=y, X=x, method="ar", p=1, pooling="global")
    assert plan.method == "ar"

    top = fmrimod.whiten_apply(plan, x, y)
    direct = ar.whiten_apply(plan, x, y)
    np.testing.assert_allclose(top.X, direct.X)
    np.testing.assert_allclose(top.Y, direct.Y)


def test_top_level_fmriar_diagnostics_and_sandwich_helpers():
    rng = np.random.default_rng(123)
    x = np.column_stack([np.ones(50), rng.normal(size=50)])
    y = rng.normal(size=(50, 3))

    diag = fmrimod.acorr_diagnostics(y, max_lag=4)
    assert diag["acf"].shape == (4,)

    se = fmrimod.sandwich_from_whitened_resid(x, y, type="iid")
    assert se["se"].shape == (2, 3)
    assert se["type"] == "iid"


def test_top_level_fmriar_compat_module_exposes_expected_helpers():
    assert fmrimod.compat.plan_from_phi is ar.compat.plan_from_phi
    assert fmrimod.compat.whiten_with_phi is ar.compat.whiten_with_phi

    phi = np.array([0.2])
    plan = fmrimod.compat.plan_from_phi(phi)
    assert plan.order == (1, 0)
