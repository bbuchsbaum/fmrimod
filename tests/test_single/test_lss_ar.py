"""Regression tests for AR-aware LSS prewhitening.

Bead: bd-01KRK97N4F0EZWSHTFG3GMCEJB. Mirrors the contract of R
``fmrilss::lss(Y, X, Z = Z, prewhiten = list(method = "ar", p = 1))``.

The dispatcher seam is :func:`fmrimod.single.estimate_single_trial`;
the new acceptance points are:

* ``baseline_regressors`` is whitened alongside ``Y / X / confounds``
  (the previous implementation silently skipped it).
* The :class:`~fmrimod.ar.plan.WhiteningPlan` is propagated into
  ``result.extra['whitening_plan']`` for diagnosability.
* Defaults preserved: when no ``prewhiten`` is given, the result is
  byte-equal to a call without the kwarg.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single import estimate_single_trial
from fmrimod.single._prewhiten import PrewhitenConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260514)


def _ar1_noise(n: int, v: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Generate AR(1) noise of shape ``(n, v)`` with persistence ``phi``."""
    eps = rng.standard_normal((n, v))
    out = np.empty_like(eps)
    out[0] = eps[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def _make_trial_design(n: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse boxcar-style trial regressors that don't align with AR structure."""
    X = np.zeros((n, n_trials))
    spacing = n // (n_trials + 1)
    width = max(2, spacing // 3)
    for k in range(n_trials):
        start = (k + 1) * spacing
        X[start : start + width, k] = 1.0
    return X


class TestPrewhitenPropagation:
    def test_whitening_plan_attached_to_extra(self, rng: np.random.Generator) -> None:
        n, n_trials, v = 80, 6, 12
        X = _make_trial_design(n, n_trials, rng)
        Y = rng.standard_normal((n, v))
        cfg = PrewhitenConfig(method="ar", p=1)

        result = estimate_single_trial(Y, X, method="lss", prewhiten=cfg)

        assert "whitening_plan" in result.extra
        plan = result.extra["whitening_plan"]
        assert plan.order == (1, 0)
        assert plan.pooling == "global"
        assert plan.phi is not None
        assert len(plan.phi) == 1
        assert plan.phi[0].shape == (1,)

    def test_no_prewhiten_leaves_extra_clean(self, rng: np.random.Generator) -> None:
        n, n_trials, v = 80, 6, 12
        X = _make_trial_design(n, n_trials, rng)
        Y = rng.standard_normal((n, v))

        result = estimate_single_trial(Y, X, method="lss")

        assert "whitening_plan" not in result.extra

    def test_method_none_is_byte_equal_to_omitted(
        self, rng: np.random.Generator
    ) -> None:
        n, n_trials, v = 80, 6, 12
        X = _make_trial_design(n, n_trials, rng)
        Y = rng.standard_normal((n, v))

        res_default = estimate_single_trial(Y, X, method="lss")
        res_none = estimate_single_trial(
            Y, X, method="lss", prewhiten=PrewhitenConfig(method="none")
        )

        assert_allclose(res_default.betas, res_none.betas, atol=0, rtol=0)
        assert "whitening_plan" not in res_none.extra


class TestBaselineRegressorsWhitened:
    """The baseline_regressors block must share the plan applied to X / Y.

    Before this bead, the dispatcher whitened Y / X / confounds but
    passed ``baseline_regressors`` straight through unwhitened, producing
    an analytically biased LSS adjustment under AR noise. The regression
    pins the parity: prewhitened LSS with baseline matches the result
    obtained by manually whitening the baseline upstream.
    """

    def test_dispatcher_whitens_baseline_alongside_x(
        self, rng: np.random.Generator
    ) -> None:
        n, n_trials, v = 96, 5, 8
        phi = 0.55
        X = _make_trial_design(n, n_trials, rng)
        noise = _ar1_noise(n, v, phi, rng)
        # Plant a real trial signal so the betas aren't pure noise.
        beta_true = rng.standard_normal((n_trials, v))
        Y = X @ beta_true + 1.5 * noise

        # Baseline = intercept + linear drift (typical fMRI nuisance).
        intercept = np.ones((n, 1))
        drift = np.linspace(-1.0, 1.0, n)[:, None]
        baseline = np.hstack([intercept, drift])

        cfg = PrewhitenConfig(method="ar", p=1)

        result_via_dispatcher = estimate_single_trial(
            Y, X, method="lss",
            prewhiten=cfg,
            baseline_regressors=baseline,
        )

        # The plan should have been applied to baseline as well: re-run
        # the dispatcher *without* baseline, then attach the same
        # (now-whitened) baseline upstream and re-fit with prewhiten off.
        # That manual upstream-whitening control must match the
        # dispatcher-whitened path within floating point.
        from fmrimod.ar.whitening import whiten_apply
        plan = result_via_dispatcher.extra["whitening_plan"]
        manual = whiten_apply(plan, np.hstack([baseline, X]), Y)
        assert manual.X is not None and manual.Y is not None
        baseline_w = manual.X[:, : baseline.shape[1]]
        X_w = manual.X[:, baseline.shape[1] :]
        Y_w = manual.Y

        result_manual = estimate_single_trial(
            Y_w, X_w, method="lss",
            baseline_regressors=baseline_w,
        )

        assert_allclose(
            result_via_dispatcher.betas, result_manual.betas,
            atol=1e-10, rtol=1e-10,
        )


class TestAr1BiasRecovery:
    """On AR(1) data, prewhitened LSS should be less biased than naive LSS.

    The test makes the bias direction explicit: we measure the
    mean-squared error of the recovered betas against the planted
    truth, under matched noise variance. AR(1) noise inflates the
    apparent SE of naive LSS estimates and biases the joint solve
    toward the high-autocorrelation regime; the whitened solve removes
    the cross-trial coupling and should sit closer to the truth.
    """

    def test_ar1_lss_recovers_truth_better_than_naive(
        self, rng: np.random.Generator
    ) -> None:
        n, n_trials, v = 160, 8, 6
        phi = 0.7
        X = _make_trial_design(n, n_trials, rng)
        beta_true = rng.standard_normal((n_trials, v))
        # Modest signal-to-noise so the AR structure dominates leftover variance.
        Y = X @ beta_true + 2.5 * _ar1_noise(n, v, phi, rng)

        cfg = PrewhitenConfig(method="ar", p=1)

        naive = estimate_single_trial(Y, X, method="lss")
        whitened = estimate_single_trial(Y, X, method="lss", prewhiten=cfg)

        mse_naive = float(np.mean((naive.betas - beta_true) ** 2))
        mse_whitened = float(np.mean((whitened.betas - beta_true) ** 2))

        assert mse_whitened < mse_naive, (
            f"prewhitened LSS should beat naive under AR(1) noise; "
            f"naive MSE={mse_naive:.4g}, whitened MSE={mse_whitened:.4g}"
        )
        # And the plan should have detected non-trivial positive AR.
        plan = whitened.extra["whitening_plan"]
        assert plan.phi is not None
        assert plan.phi[0].shape == (1,)
        assert plan.phi[0][0] > 0.3, (
            f"expected positive AR(1) coefficient near {phi}, got {plan.phi[0][0]:.3f}"
        )


class TestPrewhitenScopeBoundaries:
    def test_voxel_pooling_with_explicit_parcels_rejected(
        self, rng: np.random.Generator
    ) -> None:
        n, n_trials, v = 60, 4, 5
        X = _make_trial_design(n, n_trials, rng)
        Y = rng.standard_normal((n, v))
        cfg = PrewhitenConfig(
            method="ar",
            p=1,
            pooling="voxel",
            parcels=np.zeros(v, dtype=np.intp),
        )

        with pytest.raises(ValueError, match="incompatible with explicit parcels"):
            estimate_single_trial(Y, X, method="lss", prewhiten=cfg)
