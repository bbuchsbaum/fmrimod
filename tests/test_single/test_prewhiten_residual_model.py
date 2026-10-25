"""Single-trial noise model: aggregate-trial residuals and ``residual_model``.

Guards fmrilss ``a9d816e`` / ``37fa95a`` (``R/prewhitening.R:155-183,258-276``,
``R/lss_prewhiten_kernel.R:43-49,112-120,304-310``, ``R/oasis_backend.R:245-252``)
and transcribes ``test-lss-kernel.R:211-260,324-355``. The noise model is fitted
to residuals of the confounds plus one summed trial regressor (one sum per
basis function for multi-basis designs) unless ``residual_model="full"`` asks
for the trial-wise (LSA) residuals or ``"corrected"`` adds fmriAR's
residual-ACVF bias correction for that full design.

Cheap pass disqualified: dropping X from the residual design or clipping phi
could satisfy the 0.4 +/- 0.1 bound, but not the exact equality with a
``fit_noise`` on aggregate residuals; a row-sum-only summary fails the K=2
per-basis equality; accepting and ignoring ``residual_model`` fails the
full-vs-aggregate and corrected-vs-manual-design equalities.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import gamma

from fmrimod.ar.estimation import fit_noise
from fmrimod.single import estimate_single_trial
from fmrimod.single._prewhiten import PrewhitenConfig


def _ar1(e: np.ndarray, phi: float) -> np.ndarray:
    """``stats::filter(e, phi, "recursive")`` column-wise."""
    out = np.empty_like(e)
    acc = np.zeros(e.shape[1:])
    for t in range(e.shape[0]):
        acc = e[t] + phi * acc
        out[t] = acc
    return out


def _residualize(Y: np.ndarray, design: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(design)
    return Y - Q @ (Q.T @ Y)


def _orthonormal_basis(design: np.ndarray) -> np.ndarray:
    U, s, _ = np.linalg.svd(design, full_matrices=False)
    return U[:, s > max(design.shape) * np.finfo(float).eps * s[0]]


def _phi(plan) -> float:
    return float(np.asarray(plan.phi[0])[0])


def _fit_ar1(resid: np.ndarray, **kwargs) -> float:
    plan = fit_noise(
        resid=resid, method="ar", p=1, pooling="global", exact_first="ar1", **kwargs
    )
    return _phi(plan)


@pytest.fixture(scope="module")
def rapid_design():
    """R ``test-lss-kernel.R:211-230``: 90 trials in 240 scans, AR(1) = 0.4."""
    rng = np.random.default_rng(21)
    n, n_trials, V = 240, 90, 200
    onsets = np.sort(rng.choice(np.arange(3, n - 18 + 1), n_trials, replace=False))
    t = np.arange(16)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    X = np.zeros((n, n_trials))
    for j, onset in enumerate(onsets):
        stop = min(n, onset - 1 + h.size)
        X[onset - 1 : stop, j] = h[: stop - onset + 1]
    noise = _ar1(rng.standard_normal((n, V)), 0.4)
    Y = X @ rng.normal(1.0, 0.3, size=(n_trials, V)) + noise
    return Y, X, np.ones((n, 1))


class TestAggregateResidualModel:
    def test_default_noise_model_uses_aggregate_trial_residuals(self, rapid_design):
        Y, X, Z = rapid_design
        result = estimate_single_trial(
            Y, X, prewhiten=PrewhitenConfig(p=1), baseline_regressors=Z
        )
        phi_agg = _phi(result.extra.whitening_plan)
        assert abs(phi_agg - 0.4) < 0.1

        resid = _residualize(Y, np.hstack([Z, X.sum(axis=1, keepdims=True)]))
        assert abs(phi_agg - _fit_ar1(resid)) < 1e-10

    def test_full_residual_model_keeps_trialwise_residuals(self, rapid_design):
        Y, X, Z = rapid_design
        agg = estimate_single_trial(
            Y, X, prewhiten=PrewhitenConfig(p=1), baseline_regressors=Z
        )
        full = estimate_single_trial(
            Y,
            X,
            prewhiten=PrewhitenConfig(p=1, residual_model="full"),
            baseline_regressors=Z,
        )
        phi_agg = _phi(agg.extra.whitening_plan)
        phi_full = _phi(full.extra.whitening_plan)
        assert abs(phi_full - _fit_ar1(_residualize(Y, np.hstack([Z, X])))) < 1e-10
        # 90 trial columns in 240 scans bias the full-model AR estimate down.
        assert phi_full < phi_agg - 0.15
        assert np.max(np.abs(full.betas - agg.betas)) > 1e-3

    def test_lsa_shares_the_aggregate_noise_model(self, rapid_design):
        Y, X, Z = rapid_design
        lss = estimate_single_trial(
            Y, X, prewhiten=PrewhitenConfig(p=1), baseline_regressors=Z
        )
        lsa = estimate_single_trial(
            Y, X, method="lsa", prewhiten=PrewhitenConfig(p=1), baseline_regressors=Z
        )
        assert abs(
            _phi(lsa.extra.whitening_plan) - _phi(lss.extra.whitening_plan)
        ) < 1e-12

    def test_oasis_multi_basis_noise_design_sums_each_basis(self):
        rng = np.random.default_rng(7)
        n, n_trials, K, V = 200, 30, 2, 40
        onsets = np.sort(rng.choice(np.arange(2, n - 20), n_trials, replace=False))
        lag = np.arange(16, dtype=np.float64)
        canon = gamma.pdf(lag, 6) - gamma.pdf(lag, 16) / 6
        basis = np.column_stack([canon, np.gradient(canon)])
        X = np.zeros((n, n_trials * K))
        for j, onset in enumerate(onsets):
            X[onset : onset + lag.size, j * K : (j + 1) * K] = basis
        amps = rng.normal(1.0, 0.3, size=(n_trials * K, V))
        Y = X @ amps + _ar1(rng.standard_normal((n, V)), 0.4)

        result = estimate_single_trial(
            Y, X, method="oasis", n_basis=K, prewhiten=PrewhitenConfig(p=1)
        )
        phi = _phi(result.extra.whitening_plan)

        per_basis = np.column_stack([X[:, k::K].sum(axis=1) for k in range(K)])
        ones = np.ones((n, 1))
        expected = _fit_ar1(_residualize(Y, np.hstack([ones, per_basis])))
        row_sum = _fit_ar1(_residualize(Y, np.hstack([ones, X.sum(1, keepdims=True)])))
        assert abs(phi - expected) < 1e-10
        assert abs(phi - row_sum) > 1e-6

    def test_sbhm_noise_design_is_the_aggregate_basis_design(self):
        """R ``lss_sbhm`` fits the noise model with ``X_noise = A_agg``."""
        from fmrimod.single.sbhm.library import build_sbhm_library

        rng = np.random.default_rng(33)
        library = build_sbhm_library(rng.standard_normal((20, 50)), r=3)
        n, n_trials, V = 120, 12, 15
        K = library.B.shape[1]
        X = rng.standard_normal((n, n_trials * K))
        confounds = rng.standard_normal((n, 2))
        Y = _ar1(rng.standard_normal((n, V)), 0.4)

        result = estimate_single_trial(
            Y,
            X,
            method="sbhm",
            confounds=confounds,
            sbhm_library=library,
            prewhiten=PrewhitenConfig(p=1),
        )
        a_agg = np.column_stack([X[:, k::K].sum(axis=1) for k in range(K)])
        expected = _fit_ar1(
            _residualize(Y, np.hstack([a_agg, confounds, np.ones((n, 1))]))
        )
        assert abs(_phi(result.extra.whitening_plan) - expected) < 1e-10


class TestResidualModelOptions:
    def test_defaults_resolve_like_r(self):
        assert PrewhitenConfig().residual_model is None
        assert PrewhitenConfig().resolved_residual_model == "aggregate"
        assert PrewhitenConfig(design=np.eye(3)).resolved_residual_model == "full"
        assert (
            PrewhitenConfig(acvf_correction=np.eye(3)).resolved_residual_model
            == "full"
        )
        assert (
            PrewhitenConfig(design=np.eye(3), residual_model="full").residual_model
            == "full"
        )

    @pytest.mark.parametrize("model", ["aggregate", "corrected"])
    def test_explicit_correction_requires_full_residuals(self, model):
        with pytest.raises(ValueError, match="residual_model='full'"):
            PrewhitenConfig(design=np.eye(3), residual_model=model)
        with pytest.raises(ValueError, match="residual_model='full'"):
            PrewhitenConfig(acvf_correction=np.eye(3), residual_model=model)

    def test_unknown_residual_model_rejected(self):
        with pytest.raises(ValueError, match="residual_model must be one of"):
            PrewhitenConfig(residual_model="bogus")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"pooling": "voxel"}, "pooling='global' or 'run'"),
            (
                {"pooling": "parcel", "parcels": np.zeros(4, dtype=int)},
                "pooling='global' or 'run'",
            ),
            ({"method": "arma", "q": 1}, "method='ar'"),
        ],
    )
    def test_corrected_requires_global_or_run_ar(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PrewhitenConfig(residual_model="corrected", **kwargs)

    def test_corrected_equals_manual_orthonormal_design(self):
        """R ``test-lss-kernel.R:324-343``."""
        rng = np.random.default_rng(4)
        n, V = 160, 400
        X = rng.standard_normal((n, 40))
        Z = np.column_stack([np.ones(n), np.arange(1, n + 1) / n])
        Y = _ar1(rng.standard_normal((n, V)), 0.4)

        auto = estimate_single_trial(
            Y,
            X,
            baseline_regressors=Z,
            prewhiten=PrewhitenConfig(p=1, residual_model="corrected"),
        )
        Q = _orthonormal_basis(np.hstack([np.ones((n, 1)), Z, X]))
        manual = estimate_single_trial(
            Y, X, baseline_regressors=Z, prewhiten=PrewhitenConfig(p=1, design=Q)
        )
        full = estimate_single_trial(
            Y,
            X,
            baseline_regressors=Z,
            prewhiten=PrewhitenConfig(p=1, residual_model="full"),
        )
        phi_auto = _phi(auto.extra.whitening_plan)
        assert abs(phi_auto - _phi(manual.extra.whitening_plan)) < 1e-10
        assert_allclose(auto.betas, manual.betas, atol=1e-10, rtol=0)
        # The correction must actually move phi away from the plain full fit.
        assert abs(phi_auto - _phi(full.extra.whitening_plan)) > 1e-3

    def test_corrected_adds_intercept_to_noise_basis(self):
        rng = np.random.default_rng(5)
        n, V = 120, 30
        X = rng.standard_normal((n, 10))
        Y = _ar1(rng.standard_normal((n, V)), 0.3) + 50.0

        auto = estimate_single_trial(
            Y, X, prewhiten=PrewhitenConfig(p=1, residual_model="corrected")
        )
        Q = _orthonormal_basis(np.hstack([np.ones((n, 1)), X]))
        manual = estimate_single_trial(
            Y, X, prewhiten=PrewhitenConfig(p=1, design=Q)
        )
        assert abs(
            _phi(auto.extra.whitening_plan) - _phi(manual.extra.whitening_plan)
        ) < 1e-10
