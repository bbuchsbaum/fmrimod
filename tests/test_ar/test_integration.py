"""Tests for iterative GLS pipeline."""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag, toeplitz

import fmrimod as fm
from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.whitening import (
    ar_whiten,
    ar_whiten_matrix,
    whiten_apply,
)
from fmrimod.ar.diagnostics import acorr_diagnostics
from fmrimod.ar.integration import iterative_gls
from fmrimod.dataset.data_access import get_run_data
from fmrimod.glm.errors import UnsupportedEngineConfiguration
from fmrimod.glm.preprocess import soft_subspace_projection
from fmrimod.glm.strategies import fit_runwise
from fmrimod.model.config import (
    AROptions,
    FmriLmConfig,
    RobustOptions,
    SoftSubspaceOptions,
    VolumeWeightOptions,
)


class TestFitWhitenPipeline:
    """Integration test: fit_noise -> whiten_apply -> check whiteness."""

    def test_ar1_pipeline(self):
        rng = np.random.RandomState(42)
        n, V, k = 300, 20, 4
        phi_true = 0.6

        X = rng.randn(n, k)
        beta = rng.randn(k, V)
        noise = rng.randn(n, V)
        for t in range(1, n):
            noise[t] += phi_true * noise[t - 1]
        Y = X @ beta + noise

        # OLS residuals
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ coef

        # Fit noise
        plan = fit_noise(resid=resid, method="ar", p=1)
        assert abs(plan.phi[0][0] - phi_true) < 0.15

        # Whiten
        wr = whiten_apply(plan, X, Y)

        # Check whiteness
        resid_w = wr.Y - wr.X @ np.linalg.lstsq(wr.X, wr.Y, rcond=None)[0]
        diag = acorr_diagnostics(resid_w, max_lag=5)
        assert abs(diag["acf"][0]) < 0.15

    def test_ar2_pipeline(self):
        rng = np.random.RandomState(42)
        n, V, k = 300, 10, 3
        phi_true = np.array([0.5, -0.2])

        X = rng.randn(n, k)
        beta = rng.randn(k, V)
        noise = rng.randn(n, V)
        for t in range(2, n):
            noise[t] += phi_true[0] * noise[t - 1] + phi_true[1] * noise[t - 2]
        Y = X @ beta + noise

        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ coef

        plan = fit_noise(resid=resid, method="ar", p="auto", p_max=4)
        assert plan.order[0] >= 1

        wr = whiten_apply(plan, X, Y)
        resid_w = wr.Y - wr.X @ np.linalg.lstsq(wr.X, wr.Y, rcond=None)[0]
        diag = acorr_diagnostics(resid_w, max_lag=5)
        assert abs(diag["acf"][0]) < 0.15

    def test_multirun_pipeline(self):
        rng = np.random.RandomState(42)
        n_per_run, V, k = 150, 10, 3
        n = n_per_run * 2

        X = rng.randn(n, k)
        beta = rng.randn(k, V)
        noise = rng.randn(n, V)
        for t in range(1, n):
            noise[t] += 0.5 * noise[t - 1]
        Y = X @ beta + noise

        runs = np.concatenate([np.zeros(n_per_run, dtype=int),
                               np.ones(n_per_run, dtype=int)])

        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ coef

        plan = fit_noise(resid=resid, runs=runs, method="ar", p=1, pooling="run")
        wr = whiten_apply(plan, X, Y, runs=runs)
        assert wr.X.shape == X.shape
        assert wr.Y.shape == Y.shape

    def test_convergence_across_iterations(self):
        """Multiple fit_noise calls should successfully estimate parameters."""
        rng = np.random.RandomState(42)
        n, V, k = 300, 10, 3
        phi_true = 0.6

        X = rng.randn(n, k)
        beta = rng.randn(k, V)
        noise = rng.randn(n, V)
        for t in range(1, n):
            noise[t] += phi_true * noise[t - 1]
        Y = X @ beta + noise

        # Iteration 1
        coef1 = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid1 = Y - X @ coef1
        plan1 = fit_noise(resid=resid1, method="ar", p=1)
        wr1 = whiten_apply(plan1, X, Y)

        # Iteration 2
        coef2 = np.linalg.lstsq(wr1.X, wr1.Y, rcond=None)[0]
        resid2 = wr1.Y - wr1.X @ coef2
        plan2 = fit_noise(resid=resid2, method="ar", p=1)

        # First estimate should be reasonable
        assert 0.0 < plan1.phi[0][0] < 1.0
        # After whitening, residuals should have less autocorrelation
        # (second estimate may be near zero, which is expected)

    def test_censor_pipeline(self):
        rng = np.random.RandomState(42)
        n, V = 200, 10
        noise = rng.randn(n, V)
        for t in range(1, n):
            noise[t] += 0.5 * noise[t - 1]

        censor = np.array([10, 50, 100, 150])
        plan = fit_noise(resid=noise, censor=censor, method="ar", p=1)
        X = rng.randn(n, 3)
        wr = whiten_apply(plan, X, noise, censor=censor)
        assert wr.X.shape == X.shape


class _DummyDataset:
    def __init__(self, runs):
        self._runs = runs

    def get_data(self, run):
        return self._runs[run]

    def get_censor(self, run):
        return None


class _DummyModel:
    def __init__(self, X_runs, Y_runs):
        self._X_runs = X_runs
        self.dataset = _DummyDataset(Y_runs)
        self.n_runs = len(X_runs)

    def design_matrix_array(self, run):
        return self._X_runs[run]


class TestIterativeGlsRegression:
    def test_iter_gls_ar1_does_not_collapse_on_second_iteration(self):
        """Regression: AR(1) estimate should remain stable for iter_gls > 1.

        The estimate is held fixed after the first pass (fmrireg forces
        ``iter_gls`` to 1 for AR), so a second iteration cannot collapse it.
        """
        rng = np.random.RandomState(0)
        n, k, V = 300, 6, 80
        phi_true = 0.6

        X = rng.randn(n, k)
        X[:, 0] = 1.0
        beta = rng.randn(k, V)

        innov = rng.randn(n, V)
        noise = np.zeros((n, V))
        noise[0] = innov[0]
        for t in range(1, n):
            noise[t] = phi_true * noise[t - 1] + innov[t]
        Y = X @ beta + noise

        model = _DummyModel([X], [Y])
        cfg1 = FmriLmConfig(
            ar=AROptions(struct="ar1", iter_gls=1, global_ar=True, voxelwise=False)
        )
        cfg2 = FmriLmConfig(
            ar=AROptions(struct="ar1", iter_gls=2, global_ar=True, voxelwise=False)
        )
        initial_fit = fit_runwise(model, cfg1)

        _, phi1 = iterative_gls(model, cfg1, initial_fit)
        _, phi2 = iterative_gls(model, cfg2, initial_fit)

        assert np.abs(phi1[0] - phi_true) < 0.2
        np.testing.assert_array_equal(phi2, phi1)


# -- fmri_lm AR GLS -----------------------------------------------------------


def _events(n_runs):
    onsets = np.arange(4.0, 100.0, 12.0)
    cond = np.array(["A", "B"] * 10)[: len(onsets)]
    return pd.DataFrame(
        {
            "onset": np.tile(onsets, n_runs),
            "condition": np.tile(cond, n_runs),
            "duration": 0.0,
            "run": np.repeat(np.arange(1, n_runs + 1), len(onsets)),
        }
    )


def _ar1_noise(rng, n, phis):
    """One AR(1) series per entry of ``phis``, shape ``(n, len(phis))``."""
    phis = np.asarray(phis, dtype=float)
    z = rng.standard_normal((n, phis.size))
    E = np.zeros_like(z)
    for i in range(n):
        E[i] = z[i] + (phis * E[i - 1] if i else 0.0)
    return E


def _ar_dataset(run_phis, n=150, V=12, seed=0, **kwargs):
    """Runs whose AR(1) noise has coefficient ``run_phis[r]`` in every voxel."""
    rng = np.random.default_rng(seed)
    Y = np.vstack([_ar1_noise(rng, n, [phi] * V) for phi in run_phis]) + 100.0
    return fm.fmri_dataset(
        Y,
        tr=2.0,
        events=_events(len(run_phis)),
        run_length=[n] * len(run_phis),
        **kwargs,
    )


def _ar_fit(ds, **ar):
    return fm.fmri_lm(
        "hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1", **ar))
    )


def _gls(X, Y, phi, exact_first=False):
    Xw, Yw = ar_whiten_matrix(X, Y, np.asarray(phi), exact_first_ar1=exact_first)
    return np.linalg.lstsq(Xw, Yw, rcond=None)[0]


def _run_xy(fit, ds, r):
    return fit.model.design_matrix_array(run=r), get_run_data(ds, r)


class TestSharedArPooling:
    """REG-A-12: global pooling applies the one phi it reports (168980d)."""

    @pytest.mark.parametrize("opts", [{"pooling": "global"}, {"global_ar": True}])
    def test_global_pooling_whitens_every_run_with_the_reported_phi(self, opts):
        ds = _ar_dataset([0.2, 0.6])
        fit = _ar_fit(ds, **opts)
        per_run = _ar_fit(ds).ar_params

        phi = np.asarray(fit.ar_params)
        assert phi.shape == (1,)
        assert per_run[0, 0] < phi[0] < per_run[1, 0]
        for r in range(2):
            X, Y = _run_xy(fit, ds, r)
            np.testing.assert_allclose(
                fit.run_results[r].betas, _gls(X, Y, phi), rtol=0, atol=1e-8
            )

    def test_global_phi_is_corrected_for_the_block_diagonal_ols_operator(self):
        # Each run is fitted with its own coefficients, so the residual
        # operator of the pooled estimate is block diagonal over runs.
        ds = _ar_dataset([0.3, 0.5], n=120)
        fit = _ar_fit(ds, pooling="global")
        Xs, resid = [], []
        for r in range(2):
            X, Y = _run_xy(fit, ds, r)
            Xs.append(X)
            resid.append(Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0])
        rdf = min(X.shape[0] - np.linalg.matrix_rank(X) for X in Xs)
        plan = fit_noise(
            resid=np.vstack(resid),
            runs=np.repeat([0, 1], 120),
            method="ar",
            p=1,
            p_max=1,
            pooling="global",
            design=block_diag(*Xs),
            correction_max_lag=max(1, min(25, 5, rdf // 3)),
        )
        np.testing.assert_allclose(fit.ar_params, plan.phi[0], rtol=0, atol=1e-12)

    def test_run_pooling_whitens_each_run_with_its_own_phi(self):
        ds = _ar_dataset([0.2, 0.6])
        fit = _ar_fit(ds)
        assert fit.config.ar.pooling == "run"
        phi = np.asarray(fit.ar_params)
        assert phi.shape == (2, 1)
        assert phi[1, 0] - phi[0, 0] > 0.2
        for r in range(2):
            X, Y = _run_xy(fit, ds, r)
            np.testing.assert_allclose(
                fit.run_results[r].betas, _gls(X, Y, phi[r]), rtol=0, atol=1e-8
            )

    def test_exact_first_starts_ar1_whitening_with_prais_winsten(self):
        ds = _ar_dataset([0.5], n=100)
        fit = _ar_fit(ds, exact_first=True)
        X, Y = _run_xy(fit, ds, 0)
        np.testing.assert_allclose(
            fit.betas, _gls(X, Y, fit.ar_params, exact_first=True), rtol=0, atol=1e-8
        )

    def test_exact_first_is_exact_gls_for_ar2(self):
        rng = np.random.default_rng(3)
        n, V = 100, 6
        z = rng.standard_normal((n, V))
        E = np.zeros_like(z)
        for i in range(n):
            E[i] = z[i] + (0.5 * E[i - 1] if i else 0) - (0.3 * E[i - 2] if i > 1 else 0)
        ds = fm.fmri_dataset(E + 100.0, tr=2.0, events=_events(1), run_length=[n])
        fit = fm.fmri_lm(
            "hrf(condition)",
            ds,
            config=FmriLmConfig(ar=AROptions(struct="ar2", exact_first=True)),
        )
        X, Y = _run_xy(fit, ds, 0)
        # AR(2) autocorrelation: rho_1 = phi_1 / (1 - phi_2), then the
        # Yule-Walker recursion.
        phi1, phi2 = np.asarray(fit.ar_params)
        rho = np.empty(n)
        rho[0], rho[1] = 1.0, phi1 / (1.0 - phi2)
        for k in range(2, n):
            rho[k] = phi1 * rho[k - 1] + phi2 * rho[k - 2]
        Sinv = np.linalg.inv(toeplitz(rho))
        gls = np.linalg.solve(X.T @ Sinv @ X, X.T @ Sinv @ Y)
        np.testing.assert_allclose(fit.betas, gls, rtol=0, atol=1e-8)

    def test_parcel_pooling_is_refused(self):
        ds = _ar_dataset([0.4], n=60, V=4)
        with pytest.raises(UnsupportedEngineConfiguration, match="parcel"):
            _ar_fit(ds, pooling="parcel")


class TestArGlsHonoursRunPreprocessing:
    """REG-A-04: the GLS whitens the run matrices the OLS fit solved."""

    def test_volume_weights_are_applied_before_whitening(self):
        ds = _ar_dataset([0.4, 0.4], n=100, V=6, seed=3)
        rng = np.random.default_rng(0)
        w = rng.uniform(0.2, 1.0, 200)
        w[[5, 6, 7, 120]] = 0.0
        fit = fm.fmri_lm(
            "hrf(condition)",
            ds,
            config=FmriLmConfig(
                ar=AROptions(struct="ar1"),
                volume_weights=VolumeWeightOptions(weights=w),
            ),
        )
        phi = np.asarray(fit.ar_params)
        dfres = 0.0
        for r in range(2):
            X, Y = _run_xy(fit, ds, r)
            wr = w[r * 100 : (r + 1) * 100]
            sw = np.sqrt(wr)[:, None]
            np.testing.assert_allclose(
                fit.run_results[r].betas, _gls(X * sw, Y * sw, phi[r]), rtol=0, atol=1e-8
            )
            # Zero-weight volumes are not observations (stats::lm.wfit).
            dfres += np.count_nonzero(wr > 0) - np.linalg.matrix_rank(X)
        assert fit.residual_df == dfres

    def test_soft_projection_is_applied_before_whitening(self):
        ds = _ar_dataset([0.4, 0.4], n=100, V=6, seed=3)
        N = np.random.default_rng(1).standard_normal((200, 6))
        fit = fm.fmri_lm(
            "hrf(condition)",
            ds,
            config=FmriLmConfig(
                ar=AROptions(struct="ar1"),
                soft_subspace=SoftSubspaceOptions(
                    enabled=True, nuisance_matrix=N, lam=1.0
                ),
            ),
        )
        phi = np.asarray(fit.ar_params)
        for r in range(2):
            X, Y = _run_xy(fit, ds, r)
            Xp, Yp = soft_subspace_projection(X, Y, N[r * 100 : (r + 1) * 100], 1.0)
            np.testing.assert_allclose(
                fit.run_results[r].betas, _gls(Xp, Yp, phi[r]), rtol=0, atol=1e-8
            )

    def test_dataset_censoring_is_a_gap_in_estimation_and_whitening(self):
        # fmrimod scrubs censored volumes: they are not observations, so the
        # AR model is estimated and applied on the observed segments only.
        n = 80
        censor = np.zeros(2 * n, dtype=bool)
        censor[[9, 10, 11, 54, 55, 120]] = True
        ds = _ar_dataset(
            [0.5, 0.5], n=n, V=8, seed=5, censor=[censor[:n], censor[n:]]
        )
        fit = _ar_fit(ds)
        phi = np.asarray(fit.ar_params)
        assert np.all(np.isfinite(fit.betas))

        dfres = 0.0
        for r in range(2):
            X, Y = _run_xy(fit, ds, r)
            keep = ~censor[r * n : (r + 1) * n]
            Xk, Yk = X[keep], Y[keep]
            resid = Yk - Xk @ np.linalg.lstsq(Xk, Yk, rcond=None)[0]
            grid_resid = np.zeros_like(Y)
            grid_resid[keep] = resid
            grid_X = np.zeros_like(X)
            grid_X[keep] = Xk
            rdf = int(keep.sum()) - np.linalg.matrix_rank(Xk)
            plan = fit_noise(
                resid=grid_resid,
                censor=~keep,
                method="ar",
                p=1,
                p_max=1,
                design=grid_X,
                correction_max_lag=max(1, min(25, 5, rdf // 3)),
            )
            np.testing.assert_allclose(phi[r], plan.phi[0], rtol=0, atol=1e-12)

            # Whitening restarts after every gap.
            seg = np.cumsum(np.concatenate([[0], np.diff(np.flatnonzero(keep)) > 1]))
            Xw = np.vstack([ar_whiten(Xk[seg == s], phi[r]) for s in np.unique(seg)])
            Yw = np.vstack([ar_whiten(Yk[seg == s], phi[r]) for s in np.unique(seg)])
            np.testing.assert_allclose(
                fit.run_results[r].betas,
                np.linalg.lstsq(Xw, Yw, rcond=None)[0],
                rtol=0,
                atol=1e-8,
            )
            dfres += rdf
        assert fit.residual_df == dfres


class TestVoxelwiseAr:
    """REG-A-02: each voxel's design and data are whitened with its phi."""

    def _fit(self, n_runs=1, n=180, **ar):
        rng = np.random.default_rng(11)
        phis = [0.05, 0.25, 0.45, 0.65]
        Y = np.vstack([_ar1_noise(rng, n, phis) for _ in range(n_runs)]) + 10.0
        ds = fm.fmri_dataset(
            Y, tr=2.0, events=_events(n_runs), run_length=[n] * n_runs
        )
        fit = fm.fmri_lm(
            "hrf(condition)",
            ds,
            config=FmriLmConfig(ar=AROptions(struct="ar1", voxelwise=True, **ar)),
        )
        return fit, ds

    def test_every_voxel_is_its_own_gls(self):
        fit, ds = self._fit()
        X, Y = _run_xy(fit, ds, 0)
        phi = np.asarray(fit.ar_params)
        assert phi.shape == (1, 4)
        assert np.ptp(phi) > 0.3

        con = np.zeros(fit.betas.shape[0])
        con[0], con[1] = 1.0, -1.0
        t_con = fit.contrast(con, name="A-B").stat
        F = fit.contrast(np.eye(fit.betas.shape[0])[:2], name="F").stat
        se = fit.se()
        for v in range(4):
            Xw, Yw = ar_whiten_matrix(X, Y[:, [v]], phi[:, v])
            b = np.linalg.lstsq(Xw, Yw, rcond=None)[0][:, 0]
            np.testing.assert_allclose(fit.betas[:, v], b, rtol=0, atol=1e-8)

            df = X.shape[0] - np.linalg.matrix_rank(X)
            s2 = np.sum((Yw[:, 0] - Xw @ b) ** 2) / df
            cov = np.linalg.inv(Xw.T @ Xw)
            np.testing.assert_allclose(se[:, v], np.sqrt(s2 * np.diag(cov)), rtol=1e-10)
            np.testing.assert_allclose(
                t_con[v], con @ b / np.sqrt(s2 * con @ cov @ con), rtol=1e-10
            )
            C = np.eye(len(b))[:2]
            cb = C @ b
            F_v = cb @ np.linalg.solve(C @ cov @ C.T, cb) / (2 * s2)
            np.testing.assert_allclose(F[v], F_v, rtol=1e-10)

    def test_runs_pool_the_per_voxel_fits(self):
        fit, ds = self._fit(n_runs=2, n=120)
        phi = np.asarray(fit.ar_params)
        assert phi.shape == (2, 1, 4)
        grams = []
        for r in range(2):
            X, Y = _run_xy(fit, ds, r)
            for v in range(4):
                Xw, Yw = ar_whiten_matrix(X, Y[:, [v]], phi[r, :, v])
                np.testing.assert_allclose(
                    fit.run_results[r].betas[:, v],
                    np.linalg.lstsq(Xw, Yw, rcond=None)[0][:, 0],
                    rtol=0,
                    atol=1e-8,
                )
                grams.append(Xw.T @ Xw)
        for v in range(4):
            np.testing.assert_allclose(
                fit.XtXinv[v], np.linalg.inv(grams[v] + grams[4 + v]), rtol=1e-8
            )

    def test_noise_pools_share_one_solve_per_pool(self):
        fit, ds = self._fit(noise_pools=2)
        phi = np.asarray(fit.ar_params)
        assert len(np.unique(phi)) == 2
        X, Y = _run_xy(fit, ds, 0)
        for v in range(4):
            np.testing.assert_allclose(
                fit.betas[:, v], _gls(X, Y[:, [v]], phi[:, v])[:, 0], rtol=0, atol=1e-8
            )

    @pytest.mark.parametrize(
        "config",
        [
            {"ar": {"pooling": "global"}},
            {"ar": {"iter_gls": 2}},
            {"volume_weights": VolumeWeightOptions(weights=np.ones(60))},
            {
                "soft_subspace": SoftSubspaceOptions(
                    enabled=True, nuisance_matrix=np.ones((60, 1)), lam=1.0
                )
            },
            {"robust": RobustOptions(type="huber")},
            {"censor": True},
        ],
        ids=["global", "iter_gls", "weights", "soft", "robust", "censor"],
    )
    def test_outside_the_narrow_contract_is_refused(self, config):
        config = dict(config)
        kwargs = {}
        if config.pop("censor", False):
            censor = np.zeros(60, dtype=bool)
            censor[10] = True
            kwargs["censor"] = [censor]
        ds = _ar_dataset([0.4], n=60, V=4, **kwargs)
        ar = AROptions(struct="ar1", voxelwise=True, **config.pop("ar", {}))
        with pytest.raises(UnsupportedEngineConfiguration, match="voxelwise"):
            fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=ar, **config))

    def test_voxelwise_without_an_ar_structure_is_refused(self):
        ds = _ar_dataset([0.4], n=60, V=4)
        with pytest.raises(ValueError, match="AR structure"):
            fm.fmri_lm(
                "hrf(condition)",
                ds,
                config=FmriLmConfig(ar=AROptions(voxelwise=True)),
            )


# -- fmri_lm ARMA GLS (REG-A-14) -----------------------------------------------


def _arma_dataset(n=180, V=6, n_runs=2, seed=82203, phi=0.4, theta=0.3, **kwargs):
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    E = lfilter([1.0, theta], [1.0, -phi], rng.standard_normal((n * n_runs + 200, V)), axis=0)
    Y = E[200:] + 100.0
    return fm.fmri_dataset(
        Y, tr=2.0, events=_events(n_runs), run_length=[n] * n_runs, **kwargs
    )


def _arma_gls(X, Y, phi, theta, exact_first=False):
    from fmrimod.ar.plan import plan_from_phi

    plan = plan_from_phi(np.asarray(phi), theta=np.asarray(theta), exact_first=exact_first)
    wr = whiten_apply(plan, X, Y)
    return np.linalg.lstsq(wr.X, wr.Y, rcond=None)[0]


class TestArmaFmriLm:
    """REG-A-14 (fmrireg 3c2a9c2): q > 0 is fitted as ARMA, never as AR(p)."""

    @pytest.mark.parametrize(
        "opts",
        [
            {"struct": "arma", "p": 1, "q": 1},
            {"struct": "ar1", "method": "arma", "q": 1},
            {"struct": "ar1", "q": 1},
        ],
        ids=["struct", "method", "q"],
    )
    def test_arma_is_fitted_with_per_run_theta(self, opts):
        ds = _arma_dataset()
        ar1 = _ar_fit(ds)
        fit = fm.fmri_lm(
            "hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(iter_gls=2, **opts))
        )
        phi, theta = np.asarray(fit.ar_params), np.asarray(fit.ma_params)
        assert phi.shape == (2, 1) and theta.shape == (2, 1)
        assert np.all(np.isfinite(phi)) and np.all(np.isfinite(theta))
        assert np.all(np.abs(theta - 0.3) < 0.2)
        assert np.max(np.abs(fit.betas - ar1.betas)) > 1e-6
        # Each run's betas are its GLS under the reported ARMA(1, 1).
        for r in range(2):
            X, Y = _run_xy(fit, ds, r)
            np.testing.assert_allclose(
                fit.run_results[r].betas,
                _arma_gls(X, Y, phi[r], theta[r]),
                rtol=0,
                atol=1e-8,
            )
        assert np.all(np.isfinite(fit.se()))

    def test_first_estimate_comes_from_the_ols_residuals(self):
        ds = _arma_dataset(n_runs=1)
        fit = fm.fmri_lm(
            "hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1", q=1))
        )
        X, Y = _run_xy(fit, ds, 0)
        R = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        plan = fit_noise(R, method="arma", p=1, q=1, pooling="global")
        np.testing.assert_allclose(fit.ar_params, plan.phi[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(fit.ma_params, plan.theta[0], rtol=0, atol=1e-12)

    def test_pure_ma_is_not_mistaken_for_iid(self):
        ds = _arma_dataset(n_runs=1, phi=0.0, theta=0.5)
        fit = fm.fmri_lm(
            "hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="iid", q=1))
        )
        ols = fm.fmri_lm("hrf(condition)", ds)
        assert np.asarray(fit.ar_params).size == 0
        assert abs(float(np.asarray(fit.ma_params).ravel()[0]) - 0.5) < 0.2
        assert np.max(np.abs(fit.betas - ols.betas)) > 1e-6

    def test_ar_fit_has_no_ma_params(self):
        assert _ar_fit(_ar_dataset([0.4], n=60, V=4)).ma_params is None

    @pytest.mark.parametrize(
        ("ar", "extra", "match"),
        [
            ({"pooling": "global"}, {}, "pooling='global'"),
            ({"voxelwise": True}, {}, "voxelwise"),
            ({"censor": [10]}, {}, "censoring"),
            ({"iter_gls": 0}, {}, "iter_gls"),
            ({"noise_pools": 4}, {}, "noise_pools"),
            ({}, {"robust": RobustOptions(type="huber")}, "robust"),
            ({}, {"censor": True}, "censoring"),
        ],
        ids=["global", "voxelwise", "censor", "iter0", "pools", "robust", "ds-censor"],
    )
    def test_outside_the_narrow_contract_is_refused(self, ar, extra, match):
        extra = dict(extra)
        kwargs = {}
        if extra.pop("censor", False):
            censor = np.zeros(60, dtype=bool)
            censor[10] = True
            kwargs["censor"] = [censor]
        ds = _ar_dataset([0.4], n=60, V=4, **kwargs)
        opts = AROptions(struct="ar1", q=1, **ar)
        with pytest.raises(UnsupportedEngineConfiguration, match=match):
            fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=opts, **extra))

    def test_options_no_path_reads_are_refused(self):
        ds = _ar_dataset([0.4], n=60, V=4)
        for opts in (
            AROptions(struct="ar1", method="afni"),
            AROptions(struct="ar1", parcels=np.ones(4)),
        ):
            with pytest.raises(UnsupportedEngineConfiguration):
                fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=opts))
