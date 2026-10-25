"""Tests for diagnostics: acorr_diagnostics, sandwich SEs."""

import numpy as np
import pytest

from fmrimod.ar.diagnostics import acorr_diagnostics, sandwich_from_whitened_resid


class TestAcorrDiagnostics:
    def test_basic(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 10)
        result = acorr_diagnostics(resid, max_lag=10)
        assert "lags" in result
        assert "acf" in result
        assert "ci" in result
        assert len(result["lags"]) == 10
        assert len(result["acf"]) == 10
        assert result["ci"] > 0

    def test_white_noise_within_ci(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(1000, 10)
        result = acorr_diagnostics(resid, max_lag=5)
        # Most ACF values should be within CI for white noise
        assert np.all(np.abs(result["acf"]) < 3 * result["ci"])

    def test_detects_ar1(self):
        rng = np.random.RandomState(42)
        n = 500
        resid = rng.randn(n, 5)
        for t in range(1, n):
            resid[t] += 0.5 * resid[t - 1]
        result = acorr_diagnostics(resid, max_lag=5)
        assert abs(result["acf"][0]) > result["ci"]

    def test_aggregate_none(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 5)
        result = acorr_diagnostics(resid, max_lag=5, aggregate="none")
        assert result["acf"].shape == (5, 5)

    def test_aggregate_median(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 5)
        result = acorr_diagnostics(resid, max_lag=5, aggregate="median")
        assert len(result["acf"]) == 5

    def test_runs_exclude_boundary_lag_products(self):
        """fmriAR 0.3.3: ``runs=`` must change lag-1 vs a global ACF.

        Cheap pass: accepting ``runs`` while still spanning the join.
        Two constant runs of opposite sign are white after per-run
        centering; the concatenated series is highly autocorrelated.
        """
        rng = np.random.default_rng(0)
        resid = np.concatenate(
            [
                5.0 + 0.05 * rng.standard_normal((50, 3)),
                -5.0 + 0.05 * rng.standard_normal((50, 3)),
            ],
            axis=0,
        )
        runs = np.array([1] * 50 + [2] * 50)
        ignored = acorr_diagnostics(resid, max_lag=3)
        honoured = acorr_diagnostics(resid, runs=runs, max_lag=3)
        assert abs(ignored["acf"][0]) > 0.8
        assert abs(honoured["acf"][0]) < 0.15


class TestSandwichSEs:
    def test_iid(self):
        rng = np.random.RandomState(42)
        n, p, v = 100, 3, 5
        Xw = rng.randn(n, p)
        Yw = rng.randn(n, v)
        result = sandwich_from_whitened_resid(Xw, Yw, type="iid")
        assert result["se"].shape == (p, v)
        assert result["type"] == "iid"
        assert np.all(result["se"] > 0)
        assert np.all(result["sigma2"] > 0)

    def test_hc0(self):
        rng = np.random.RandomState(42)
        n, p, v = 100, 3, 5
        Xw = rng.randn(n, p)
        Yw = rng.randn(n, v)
        result = sandwich_from_whitened_resid(Xw, Yw, type="hc0")
        assert result["se"].shape == (p, v)
        assert result["type"] == "hc0"
        assert np.all(result["se"] > 0)

    def test_iid_matches_manual(self):
        rng = np.random.RandomState(42)
        n, p = 200, 2
        X = rng.randn(n, p)
        beta_true = np.array([[1.0], [2.0]])
        Y = X @ beta_true + rng.randn(n, 1)
        result = sandwich_from_whitened_resid(X, Y, type="iid")
        # Manual computation
        XtX_inv = np.linalg.inv(X.T @ X)
        beta_hat = XtX_inv @ (X.T @ Y)
        e = Y - X @ beta_hat
        sigma2 = np.sum(e**2) / (n - p)
        se_manual = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))
        np.testing.assert_allclose(result["se"], se_manual, atol=1e-10)

    def test_with_beta(self):
        rng = np.random.RandomState(42)
        Xw = rng.randn(100, 2)
        Yw = rng.randn(100, 3)
        beta = np.linalg.lstsq(Xw, Yw, rcond=None)[0]
        result = sandwich_from_whitened_resid(Xw, Yw, beta=beta, type="iid")
        assert result["se"].shape == (2, 3)


class TestAcorrAggregatesPerVoxel:
    """fmriAR 617bd87 (test-audit-regressions.R:337-355), row AR-09.

    Cheap pass disqualified: averaging per-voxel values while keeping the
    pair-count normalisation and PSD shrink fails the ``runs`` assertion.
    """

    @staticmethod
    def _per_voxel_acf(R, max_lag):
        """``stats::acf`` per column: biased, demeaned."""
        Rc = R - R.mean(axis=0)
        den = np.sum(Rc * Rc, axis=0)
        return np.vstack([np.sum(Rc[k:] * Rc[:-k], axis=0) / den
                          for k in range(1, max_lag + 1)])

    def test_aggregates_per_voxel_autocorrelations(self):
        from scipy.signal import lfilter

        rng = np.random.default_rng(9121)
        n, v = 300, 50
        shared = lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))
        R = rng.standard_normal((n, v)) + 0.5 * shared[:, None]
        per_vox = self._per_voxel_acf(R, 3)
        np.testing.assert_allclose(
            acorr_diagnostics(R, max_lag=3, aggregate="none")["acf"], per_vox,
            rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(acorr_diagnostics(R, max_lag=3)["acf"],
                                   per_vox.mean(axis=1), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            acorr_diagnostics(R, max_lag=3, aggregate="median")["acf"],
            np.median(per_vox, axis=1), rtol=0, atol=1e-12,
        )

    def test_runs_use_biased_within_run_normalisation(self):
        rng = np.random.default_rng(5)
        n, v = 60, 4
        runs = np.repeat([1, 2], 30)
        R = rng.standard_normal((n, v)) + np.where(runs == 1, -5.0, 5.0)[:, None]
        Rc = R.copy()
        for r in (1, 2):
            Rc[runs == r] -= Rc[runs == r].mean(axis=0)
        den = np.sum(Rc * Rc, axis=0)
        ref = []
        for k in (1, 2):
            ok = runs[k:] == runs[:-k]
            ref.append(np.sum(Rc[k:][ok] * Rc[:-k][ok], axis=0) / den)
        out = acorr_diagnostics(R, runs=runs, max_lag=2, aggregate="none")["acf"]
        np.testing.assert_allclose(out, np.vstack(ref), rtol=0, atol=1e-12)
        # Two independent constant-offset runs of white noise: no lag-1 structure.
        W = rng.standard_normal((300, 50)) + np.repeat([-5.0, 5.0], 150)[:, None]
        honoured = acorr_diagnostics(W, runs=np.repeat([1, 2], 150), max_lag=1)
        assert abs(honoured["acf"][0]) < 0.02

    def test_zero_variance_voxels_are_missing_not_zero(self):
        rng = np.random.default_rng(6)
        R = rng.standard_normal((100, 3))
        R[:, 1] = 2.0
        out = acorr_diagnostics(R, max_lag=2, aggregate="none")["acf"]
        assert np.all(np.isnan(out[:, 1]))
        mean = acorr_diagnostics(R, max_lag=2)["acf"]
        np.testing.assert_allclose(mean, out[:, [0, 2]].mean(axis=1), atol=1e-12)


class TestSandwichRankDeficient:
    """AR-18 (fmriAR 617bd87): aliased columns get NaN SEs, like ``lm()``."""

    @staticmethod
    def _aliased(seed=9122, n=100):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x, 2 * x])
        y = 1 + x + rng.standard_normal(n)
        return X, y

    def test_iid_matches_lm_on_the_estimable_columns(self):
        X, y = self._aliased()
        out = sandwich_from_whitened_resid(X, y[:, None])
        Xe = X[:, :2]
        b = np.linalg.solve(Xe.T @ Xe, Xe.T @ y)
        s2 = np.sum((y - Xe @ b) ** 2) / 98
        ref = np.sqrt(np.diag(np.linalg.inv(Xe.T @ Xe)) * s2)
        assert out["df"] == 98
        assert np.isnan(out["se"][2, 0])
        np.testing.assert_allclose(out["se"][:2, 0], ref, rtol=1e-10)
        assert np.all(np.isnan(out["XtX_inv"][2])) and np.all(
            np.isnan(out["XtX_inv"][:, 2])
        )
        np.testing.assert_allclose(
            out["XtX_inv"][:2, :2], np.linalg.inv(Xe.T @ Xe), rtol=1e-10
        )

    def test_hc0_rank_deficient_and_vectorised(self):
        X, y = self._aliased()
        out = sandwich_from_whitened_resid(X, y[:, None], type="hc0")
        assert np.isnan(out["se"][2, 0]) and np.all(np.isfinite(out["se"][:2]))
        rng = np.random.default_rng(9123)
        X = np.column_stack([np.ones(200), rng.standard_normal((200, 3))])
        Y = rng.standard_normal((200, 7)) * (1 + np.abs(X[:, [1]]))
        out = sandwich_from_whitened_resid(X, Y, type="hc0")
        XtXi = np.linalg.inv(X.T @ X)
        E = Y - X @ XtXi @ X.T @ Y
        ref = np.column_stack(
            [
                np.sqrt(np.diag(XtXi @ ((X * E[:, [j]]).T @ (X * E[:, [j]])) @ XtXi))
                for j in range(7)
            ]
        )
        np.testing.assert_allclose(out["se"], ref, rtol=1e-10)

    @pytest.mark.parametrize("se_type", ["iid", "hc0"])
    def test_supplied_beta_keeps_aliased_columns_fit(self, se_type):
        # lstsq spreads the fit over x and 2x; NaN at the aliased entry is the
        # lm() convention. Both must give the residuals of the estimated fit.
        X, y = self._aliased()
        ref = sandwich_from_whitened_resid(X, y[:, None], type=se_type)
        b_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
        assert abs(b_lstsq[2]) > 0.1
        b_na = np.append(np.linalg.lstsq(X[:, :2], y, rcond=None)[0], np.nan)
        for beta in (b_lstsq, b_na):
            out = sandwich_from_whitened_resid(X, y[:, None], beta=beta, type=se_type)
            np.testing.assert_allclose(out["sigma2"], ref["sigma2"], rtol=1e-10)
            np.testing.assert_allclose(out["se"], ref["se"], rtol=1e-10, equal_nan=True)
