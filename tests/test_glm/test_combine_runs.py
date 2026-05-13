"""Tests for ``fm.combine_runs`` / ``fm.combine_contrasts``."""

from __future__ import annotations

import numpy as np
import pytest

import fmrimod as fm
from fmrimod.glm import (
    CombinedFmriLm,
    combine_contrasts,
    combine_runs,
    fit_glm_from_suffstats,
)


def _make_run_fit(rng, n_t=80, n_p=4, n_v=20, sigma=1.0):
    """Build a synthetic FmriLm fit from a random design + noise."""
    X = rng.standard_normal((n_t, n_p))
    beta_true = rng.standard_normal((n_p, n_v))
    Y = X @ beta_true + sigma * rng.standard_normal((n_t, n_v))
    XtX = X.T @ X
    XtY = X.T @ Y
    YtY = np.sum(Y * Y, axis=0)
    df = float(n_t - n_p)
    fit = fit_glm_from_suffstats(model=None, XtX=XtX, XtS=XtY, StS=YtY, df=df)
    return fit, X, Y


def _fixed_pool(estimates, ses):
    """Reference Nilearn-style equal-weight pooling."""
    est = np.asarray(estimates)
    var = np.asarray(ses) ** 2
    n = est.shape[0]
    eff = est.mean(axis=0)
    v = var.sum(axis=0) / (n * n)
    return eff, np.sqrt(v)


def _ivw_pool(estimates, ses):
    """Reference IVW pooling for comparison."""
    var = np.maximum(np.asarray(ses) ** 2, np.finfo(np.float64).tiny)
    inv_var = 1.0 / var
    sum_inv_var = inv_var.sum(axis=0)
    eff = (np.asarray(estimates) * inv_var).sum(axis=0) / sum_inv_var
    v = 1.0 / sum_inv_var
    return eff, np.sqrt(v)


def test_combine_runs_basic_construction():
    rng = np.random.default_rng(0)
    fits = [_make_run_fit(rng)[0] for _ in range(3)]
    combined = combine_runs(fits)
    assert isinstance(combined, CombinedFmriLm)
    assert combined.n_runs == 3
    assert combined.n_voxels == fits[0].n_voxels
    assert combined.method == "fixed"


def test_combine_runs_rejects_empty():
    with pytest.raises(ValueError, match="at least one fit"):
        combine_runs([])


def test_combine_runs_rejects_voxel_mismatch():
    rng = np.random.default_rng(1)
    fit_a, _, _ = _make_run_fit(rng, n_v=20)
    fit_b, _, _ = _make_run_fit(rng, n_v=21)
    with pytest.raises(ValueError, match="voxels"):
        combine_runs([fit_a, fit_b])


def test_combine_runs_unknown_method():
    rng = np.random.default_rng(2)
    fit, _, _ = _make_run_fit(rng)
    with pytest.raises(ValueError, match="unknown method"):
        combine_runs([fit], method="random")


def test_combine_runs_fixed_matches_nilearn_equal_weight():
    rng = np.random.default_rng(3)
    fits = [_make_run_fit(rng)[0] for _ in range(4)]
    c = np.array([1.0, -1.0, 0.0, 0.0])

    per_run = [f.contrast(c) for f in fits]
    expected_eff, expected_se = _fixed_pool(
        [r.estimate for r in per_run],
        [r.se for r in per_run],
    )
    expected_df = sum(float(r.df) for r in per_run)
    expected_t = expected_eff / expected_se

    combined = combine_runs(fits)
    pooled = combined.contrast(c, name="con")
    assert pooled.name == "con"
    assert pooled.stat_type == "t"
    assert combined.method == "fixed"
    np.testing.assert_allclose(pooled.estimate, expected_eff, rtol=1e-12)
    np.testing.assert_allclose(pooled.se, expected_se, rtol=1e-12)
    np.testing.assert_allclose(pooled.stat, expected_t, rtol=1e-12)
    assert pooled.df == pytest.approx(expected_df)


def test_combine_runs_ivw_method_matches_reference():
    rng = np.random.default_rng(33)
    fits = [_make_run_fit(rng)[0] for _ in range(4)]
    c = np.array([1.0, -1.0, 0.0, 0.0])

    per_run = [f.contrast(c) for f in fits]
    expected_eff, expected_se = _ivw_pool(
        [r.estimate for r in per_run],
        [r.se for r in per_run],
    )
    pooled = combine_runs(fits, method="ivw").contrast(c)
    np.testing.assert_allclose(pooled.estimate, expected_eff, rtol=1e-12)
    np.testing.assert_allclose(pooled.se, expected_se, rtol=1e-12)


def test_combine_runs_fixed_matches_nilearn_compute_fixed_effect_contrast():
    """End-to-end check against Nilearn's compute_fixed_effect_contrast."""
    from nilearn.glm.contrasts import Contrast

    rng = np.random.default_rng(34)
    n_runs = 3
    fits = [_make_run_fit(rng)[0] for _ in range(n_runs)]
    c = np.array([1.0, 0.0, 1.0, -1.0])

    per_run = [f.contrast(c) for f in fits]
    # Reproduce nilearn behavior: sum Contrast then * (1/n).
    nilearn_total = None
    for r in per_run:
        cur = Contrast(
            effect=r.estimate.reshape(1, -1),
            variance=np.asarray(r.se) ** 2,
            dim=1,
            dof=float(r.df),
            stat_type="t",
        )
        nilearn_total = cur if nilearn_total is None else nilearn_total + cur
    nilearn_total = nilearn_total * (1.0 / n_runs)

    pooled = combine_runs(fits).contrast(c)
    np.testing.assert_allclose(
        pooled.estimate,
        nilearn_total.effect_size().ravel(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        pooled.se ** 2,
        nilearn_total.effect_variance(),
        rtol=1e-12,
    )


def test_combine_contrasts_directly():
    rng = np.random.default_rng(4)
    fits = [_make_run_fit(rng)[0] for _ in range(3)]
    c = np.array([0.5, 0.5, -0.5, -0.5])
    per_run = [f.contrast(c) for f in fits]
    pooled = combine_contrasts(per_run, name="custom")
    assert pooled.name == "custom"
    assert pooled.stat_type == "t"
    assert pooled.estimate.shape == per_run[0].estimate.shape


def test_combine_contrasts_rejects_f_contrasts():
    rng = np.random.default_rng(5)
    fits = [_make_run_fit(rng)[0] for _ in range(2)]
    fmat = np.eye(4)
    per_run = [f.contrast(fmat, name="f") for f in fits]
    with pytest.raises(NotImplementedError, match="t-contrasts only"):
        combine_contrasts(per_run)


def test_combine_runs_single_fit_is_identity():
    rng = np.random.default_rng(6)
    fit, _, _ = _make_run_fit(rng)
    c = np.array([1.0, 0.0, 0.0, 0.0])
    pooled = combine_runs([fit]).contrast(c)
    direct = fit.contrast(c)
    np.testing.assert_allclose(pooled.estimate, direct.estimate, rtol=1e-12)
    np.testing.assert_allclose(pooled.se, direct.se, rtol=1e-12)
    np.testing.assert_allclose(pooled.stat, direct.stat, rtol=1e-12)


def test_top_level_combine_runs_binding():
    rng = np.random.default_rng(7)
    fits = [_make_run_fit(rng)[0] for _ in range(2)]
    combined = fm.combine_runs(fits)
    assert isinstance(combined, CombinedFmriLm)

    c = np.array([1.0, 0.0, 0.0, 0.0])
    per_run = [f.contrast(c) for f in fits]
    pooled_top = fm.combine_contrasts(per_run)
    pooled_lib = combine_contrasts(per_run)
    np.testing.assert_allclose(pooled_top.estimate, pooled_lib.estimate, rtol=1e-12)


def test_combine_runs_ivw_dominates_under_unequal_variance():
    rng = np.random.default_rng(8)
    n_v = 8
    # Build two runs whose per-voxel SEs differ markedly so IVW != equal-weight.
    fits = []
    for sigma_scale in (1.0, 5.0):
        fit, _, _ = _make_run_fit(rng, n_v=n_v, sigma=sigma_scale)
        fits.append(fit)

    c = np.array([1.0, 0.0, 0.0, 0.0])
    se_runs = np.vstack([f.contrast(c).se for f in fits])
    ivw = combine_runs(fits, method="ivw").contrast(c)
    fixed = combine_runs(fits, method="fixed").contrast(c)

    # IVW gives smaller (or equal) variance than the better single run.
    assert np.all(ivw.se <= se_runs.min(axis=0) + 1e-12)
    # And IVW gives smaller (or equal) variance than equal-weight pooling
    # whenever variances differ across runs.
    assert np.all(ivw.se <= fixed.se + 1e-12)
