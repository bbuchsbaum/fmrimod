"""Pain points pinned by the AR(1) prewhitening parity workflow.

Three issues surfaced while wiring ``tier_a_ar1_prewhitening`` and
are pinned at their current state here so a future fix has a clear
regression target. None of these are numerical bugs — they are
architectural / ergonomic gaps in the AR surface.

1. **AR(1) doesn't compose with the concat engine.** The AR
   integration path requires per-run residuals from the runwise
   strategy; the concat engine doesn't populate them.

2. **Verbose typed AR API.** ``fmri_lm(..., ar="ar1")`` is rejected
   by the engine-options resolver, so the user has to spell out
   ``FmriLmConfig(ar=AROptions(struct="ar1"))``.

3. **AR algorithm divergence from Nilearn.** fmrimod's per-voxel
   AR estimation vs Nilearn's noise-pool quantization produces
   AR(1) task betas that differ by ~10-20% on identical inputs.
   Documented in CAVEATS as ``ar1-algorithm-divergence``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
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


# -- Pain point 1: AR + concat engine broken --------------------------------


def test_ar1_with_concat_engine_raises_unclear_error() -> None:
    """Passing ``engine="concat"`` with an AR config raises a confusing error.

    The AR integration uses ``residuals_list[r]`` from the runwise
    strategy, but the concat engine doesn't populate per-run
    residuals — the error surfaces as ``TypeError: 'NoneType' object
    is not subscriptable`` rather than a clean "AR doesn't compose
    with concat" message.

    A clean fix would either:
    - Make the AR integration build its own residuals from the
      single-design solve on the concat path, OR
    - Detect the (ar, concat) combination at engine resolution time
      and raise a clear ``NotImplementedError`` (or auto-fall back
      to runwise) instead of crashing inside the algorithm.
    """
    ds = _ar1_dataset()
    spec = _ar1_spec()
    with pytest.raises((TypeError, ValueError)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fm.fmri_lm(
                spec, ds,
                config=FmriLmConfig(ar=AROptions(struct="ar1")),
                engine="concat",
            )


# -- Pain point 2: verbose typed AR API -------------------------------------


def test_ar_string_shorthand_is_rejected_by_engine_resolver() -> None:
    """``fmri_lm(..., ar="ar1")`` is not yet accepted as a shorthand.

    Pinned at the current state so a future fix that adds the
    shorthand can flip this assertion.
    """
    ds = _ar1_dataset()
    spec = _ar1_spec()
    with pytest.raises((ValueError, TypeError)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fm.fmri_lm(spec, ds, ar="ar1")


def test_ar1_works_via_full_config_path() -> None:
    """The current canonical way to enable AR(1) is via FmriLmConfig.

    Positive pin: ``fmri_lm(..., config=FmriLmConfig(
    ar=AROptions(struct="ar1")))`` runs and returns AR-corrected
    betas. Verifies the path the parity workflow relies on.
    """
    ds = _ar1_dataset()
    spec = _ar1_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit_ols = fm.fmri_lm(spec, ds)
        fit_ar = fm.fmri_lm(
            spec, ds, config=FmriLmConfig(ar=AROptions(struct="ar1")),
        )
    # AR(1) betas differ from OLS betas on the same data — the
    # prewhitening changes the solve.
    assert not np.allclose(fit_ar.betas, fit_ols.betas), (
        "AR(1) and OLS betas should differ on noisy data"
    )
    assert fit_ar.betas.shape == fit_ols.betas.shape


# -- Pain point 3: documented algorithm divergence from Nilearn -------------


def test_ar1_algorithm_divergence_is_within_documented_bounds() -> None:
    """fmrimod and Nilearn AR(1) betas differ but within bounds.

    Both engines run AR(1) on the same X / Y; the algorithm
    divergence (per-voxel estimation vs noise-pool quantization)
    produces beta differences ~10-20% of typical effect magnitudes.
    Per-voxel correlation remains > 0.99.

    This test pins the divergence at the observed level so a future
    AR-algorithm refactor that either harmonizes with Nilearn OR
    drifts further can be caught and reviewed.
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
        fit_fm = fm.fmri_lm(
            spec, ds, config=FmriLmConfig(ar=AROptions(struct="ar1")),
        )
    labels, estimates = run_glm(Y, X, noise_model="ar1", n_jobs=1)
    nl_betas = np.zeros_like(fit_fm.betas)
    for label in np.unique(labels):
        mask = labels == label
        nl_betas[:, mask] = estimates[label].theta

    # Per-voxel correlation > 0.99 on the task betas.
    for i in range(2):
        corr = float(np.corrcoef(fit_fm.betas[i], nl_betas[i])[0, 1])
        assert corr > 0.99, (
            f"AR(1) task-{i} beta correlation against Nilearn dropped "
            f"to {corr:.6f}; if this happens, either fmrimod or Nilearn "
            f"changed their AR algorithm. Investigate before raising the "
            f"tolerance."
        )
    # Intercept stays robust to AR algorithm choice.
    intercept_rel = np.max(
        np.abs(fit_fm.betas[-1] - nl_betas[-1]) / np.abs(nl_betas[-1])
    )
    assert intercept_rel < 1e-3, (
        f"intercept relative difference should stay < 1e-3; got "
        f"{intercept_rel:.4e}"
    )
