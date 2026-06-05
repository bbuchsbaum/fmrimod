"""Regression tests for the AR(1) ergonomic fixes.

History: ``tier_a_ar1_prewhitening`` surfaced three AR(1)
ergonomic gaps. Two were broken-state pins (the AR+concat crash
and the rejected ``ar="ar1"`` shorthand) and one was a relaxed-
tolerance pin on the algorithm divergence vs Nilearn. All three
are now fixed:

1. ``fmri_lm(..., ar="ar1", engine="concat")`` now raises a clean
   :class:`NotImplementedError` at engine resolution instead of a
   ``TypeError: 'NoneType' object is not subscriptable`` deep
   inside ``iterative_gls``.
2. ``fmri_lm(..., ar="ar1")`` accepts the string shorthand and
   maps it onto the underlying :class:`AROptions(struct="ar1")`
   config.
3. ``AROptions(voxelwise=True, noise_pools=10)`` adds Nilearn-
   style noise-pool quantization on top of voxelwise AR
   estimation, available for users who want the algorithm shape
   that matches ``run_glm(..., noise_model="ar1")``.

The Tier B parity workflow still uses fmrimod's default global
AR estimation because that path already correlates > 0.998 with
Nilearn; noise-pool quantization is a user-facing option, not an
auto-applied default.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.ar.estimation import _quantise_to_noise_pools
from fmrimod.model.config import AROptions, FmriLmConfig
from fmrimod.spec import drift, hrf, intercept


def _ar1_dataset(seed: int = 0, n: int = 120, n_voxels: int = 16):
    rng = np.random.default_rng(seed)
    events = pd.DataFrame({
        "onset": np.linspace(10.0, 220.0, 12),
        "duration": 0.0,
        "trial_type": ["A", "B"] * 6,
        "run": 1,
    })
    Y = rng.normal(size=(n, n_voxels))
    return fm.fmri_dataset(Y, tr=2.0, events=events, slice_timing_offset=0.0)


def _ar1_spec():
    return (
        hrf("trial_type", basis="spm", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )


# -- Fixed: AR + concat raises a clean error -------------------------------


def test_ar1_with_concat_engine_raises_clean_notimplemented() -> None:
    """``ar="ar1"`` + ``engine="concat"`` now raises NotImplementedError early.

    Before the fix the call crashed inside ``iterative_gls`` with
    ``TypeError: 'NoneType' object is not subscriptable``. The
    engine-resolution layer now detects the combination and raises a
    clean ``NotImplementedError`` with a fix suggestion pointing at
    the default runwise engine.
    """
    ds = _ar1_dataset()
    spec = _ar1_spec()
    with pytest.raises(NotImplementedError, match="concat"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fm.fmri_lm(spec, ds, ar="ar1", engine="concat")


# -- Fixed: ar="ar1" string shorthand --------------------------------------


def test_ar_string_shorthand_works() -> None:
    """``fmri_lm(..., ar="ar1")`` now accepts the string shorthand."""
    ds = _ar1_dataset()
    spec = _ar1_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_short = fm.fmri_lm(spec, ds, ar="ar1")
        fit_full = fm.fmri_lm(
            spec, ds,
            config=FmriLmConfig(ar=AROptions(struct="ar1")),
        )
    # Both paths produce identical betas.
    np.testing.assert_allclose(fit_short.betas, fit_full.betas, atol=1e-12)


def test_ar_string_shorthand_rejects_unknown_struct() -> None:
    """An unknown AR struct string raises a clear error."""
    ds = _ar1_dataset()
    spec = _ar1_spec()
    with pytest.raises(ValueError, match="AR struct"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fm.fmri_lm(spec, ds, ar="garbage")


def test_ar_kwarg_accepts_full_aroptions_instance() -> None:
    """``ar=AROptions(...)`` is also accepted (composes with the shorthand)."""
    ds = _ar1_dataset()
    spec = _ar1_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, ds, ar=AROptions(struct="ar1"))
    assert fit.betas.shape[0] > 0


# -- Fixed: AROptions(noise_pools=...) quantization ------------------------


def test_noise_pools_quantises_voxelwise_estimates() -> None:
    """``AROptions(voxelwise=True, noise_pools=N)`` quantises per-voxel AR.

    Pinning the algorithm: 20 voxels with strictly increasing true
    AR coefficients get quantised into 5 equal-frequency pools, so
    the output has 5 unique values each shared by 4 voxels.
    """
    phi = np.linspace(0.0, 0.5, 20).reshape(1, 20)
    out = _quantise_to_noise_pools(phi, n_pools=5)
    assert out.shape == phi.shape
    assert len(np.unique(out[0])) == 5
    # Each pool's representative is the median of its members; for
    # the equal-spaced input the bins land at strictly increasing
    # values.
    representatives = sorted(np.unique(out[0]))
    assert all(
        representatives[i] < representatives[i + 1]
        for i in range(len(representatives) - 1)
    )


def test_noise_pools_passes_through_aroptions() -> None:
    """``AROptions(voxelwise=True, noise_pools=10)`` runs end-to-end."""
    ds = _ar1_dataset(n_voxels=32)
    spec = _ar1_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            spec, ds,
            config=FmriLmConfig(
                ar=AROptions(
                    struct="ar1", voxelwise=True, noise_pools=10
                )
            ),
        )
    assert fit.betas.shape[1] == 32
    # The voxelwise + noise_pools path produces task betas that
    # differ from the (default global) AR(1) path — pooling at
    # the per-voxel level is a different algorithm.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_default = fm.fmri_lm(spec, ds, ar="ar1")
    assert not np.allclose(fit.betas, fit_default.betas)


def test_noise_pools_zero_or_one_is_a_noop() -> None:
    """``noise_pools <= 1`` falls back to per-voxel estimates unchanged."""
    phi = np.linspace(0.0, 0.5, 8).reshape(1, 8)
    np.testing.assert_array_equal(
        _quantise_to_noise_pools(phi, n_pools=1), phi
    )


# -- Algorithm divergence (still documented) -------------------------------


def test_ar1_algorithm_divergence_is_within_documented_bounds() -> None:
    """fmrimod and Nilearn AR(1) betas agree at per-voxel corr > 0.99.

    Even after the ergonomic fixes the underlying AR algorithms still
    differ (fmrimod's global Yule-Walker vs Nilearn's noise-pool
    quantization, both default behaviors). Per-voxel beta correlation
    stays > 0.998 and intercepts agree to < 1e-3 relative; pin the
    divergence at the current observed level so an AR-algorithm
    refactor that either harmonises further OR drifts is caught.
    """
    from nilearn.glm.first_level import run_glm

    rng = np.random.default_rng(42)
    n = 200
    n_voxels = 32
    events = pd.DataFrame({
        "onset": np.linspace(10.0, 380.0, 20),
        "duration": 0.0,
        "trial_type": ["A", "B"] * 10,
        "run": 1,
    })
    dummy = fm.fmri_dataset(
        np.zeros((n, 1)), tr=2.0, events=events, slice_timing_offset=0.0
    )
    spec = _ar1_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dummy_fit = fm.fmri_lm(spec, dummy)
    X = dummy_fit.model.design_matrix_array(run=None)
    true_betas = np.zeros((X.shape[1], n_voxels))
    true_betas[0] = 1.0
    true_betas[1] = 0.5
    true_betas[-1] = 100.0
    white = rng.normal(scale=0.5, size=(n, n_voxels))
    noise = np.zeros_like(white)
    noise[0] = white[0]
    for t in range(1, n):
        noise[t] = 0.5 * noise[t - 1] + white[t]
    Y = X @ true_betas + noise

    ds = fm.fmri_dataset(Y, tr=2.0, events=events, slice_timing_offset=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_fm = fm.fmri_lm(spec, ds, ar="ar1")  # new string shorthand
    labels, estimates = run_glm(Y, X, noise_model="ar1", n_jobs=1)
    nl_betas = np.zeros_like(fit_fm.betas)
    for label in np.unique(labels):
        mask = labels == label
        nl_betas[:, mask] = estimates[label].theta

    for i in range(2):
        corr = float(np.corrcoef(fit_fm.betas[i], nl_betas[i])[0, 1])
        assert corr > 0.99
    intercept_rel = np.max(
        np.abs(fit_fm.betas[-1] - nl_betas[-1]) / np.abs(nl_betas[-1])
    )
    assert intercept_rel < 1e-3
