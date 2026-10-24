"""Tests for whitening: whiten_apply, segment-aware, exact first."""

import numpy as np
import pytest
import typing

import fmrimod.ar.whitening as whitening_mod
from fmrimod.ar.plan import WhiteningPlan, WhitenResult
from fmrimod.ar.whitening import (
    ar_whiten,
    ar_whiten_matrix,
    arma_whiten_segments,
    whiten,
    whiten_apply,
)
from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.diagnostics import acorr_diagnostics


def _sim_ar1(n, phi, V=10, seed=42):
    rng = np.random.RandomState(seed)
    e = rng.randn(n, V)
    for t in range(1, n):
        e[t] += phi * e[t - 1]
    return e


class TestArmaWhitenSegments:
    def test_pure_ar1(self):
        rng = np.random.RandomState(42)
        y = rng.randn(200, 5)
        phi = np.array([0.5])
        result = arma_whiten_segments(y, phi, np.array([]), np.array([0]))
        assert result.shape == y.shape

    def test_segment_resets(self):
        rng = np.random.RandomState(42)
        y = rng.randn(200, 1)
        phi = np.array([0.5])
        # Single segment
        r1 = arma_whiten_segments(y, phi, np.array([]), np.array([0]))
        # Two segments
        r2 = arma_whiten_segments(y, phi, np.array([]), np.array([0, 100]))
        # Results should differ at segment boundary
        assert not np.allclose(r1[100], r2[100])

    def test_exact_first_ar1(self):
        rng = np.random.RandomState(42)
        y = rng.randn(100, 1)
        phi = np.array([0.8])
        r_no = arma_whiten_segments(y, phi, np.array([]), np.array([0]),
                                     exact_first_ar1=False)
        r_ex = arma_whiten_segments(y, phi, np.array([]), np.array([0]),
                                     exact_first_ar1=True)
        # First sample should differ
        assert not np.allclose(r_no[0], r_ex[0])
        # Exact scaling: y[0] * sqrt(1 - phi^2)
        expected_scale = np.sqrt(1 - 0.8 ** 2)
        np.testing.assert_allclose(r_ex[0], y[0] * expected_scale, atol=1e-10)

    def test_arma11(self):
        rng = np.random.RandomState(42)
        y = rng.randn(200, 3)
        phi = np.array([0.5])
        theta = np.array([0.3])
        result = arma_whiten_segments(y, phi, theta, np.array([0]))
        assert result.shape == y.shape

    def test_1d_input(self):
        rng = np.random.RandomState(42)
        y = rng.randn(100)
        phi = np.array([0.5])
        result = arma_whiten_segments(y, phi, np.array([]), np.array([0]))
        assert result.shape == (100,)

    def test_empty_phi(self):
        rng = np.random.RandomState(42)
        y = rng.randn(100, 3)
        result = arma_whiten_segments(y, np.array([]), np.array([]), np.array([0]))
        np.testing.assert_allclose(result, y)


class TestSimpleArWhiten:
    def test_ar_whiten_default_leaves_first_sample_unscaled(self):
        y = np.arange(1.0, 6.0)[:, np.newaxis]
        phi = np.array([0.6])

        out = ar_whiten(y, phi)

        np.testing.assert_allclose(out[0], y[0])
        np.testing.assert_allclose(out[1:, 0], y[1:, 0] - 0.6 * y[:-1, 0])

    def test_ar_whiten_exact_first_ar1_scales_first_sample_when_requested(self):
        y = np.arange(1.0, 6.0)[:, np.newaxis]
        phi = np.array([0.6])

        out = ar_whiten(y, phi, exact_first_ar1=True)

        np.testing.assert_allclose(out[0], y[0] * np.sqrt(1.0 - 0.6**2))
        np.testing.assert_allclose(out[1:, 0], y[1:, 0] - 0.6 * y[:-1, 0])

    def test_ar_whiten_matrix_voxelwise_exact_first_is_optional(self):
        X = np.column_stack([np.ones(5), np.arange(5.0)])
        Y = np.column_stack([np.arange(1.0, 6.0), np.arange(2.0, 7.0)])
        phi = np.array([[0.3, 0.6]])

        Xw_plain, Yw_plain = ar_whiten_matrix(X, Y, phi)
        Xw_exact, Yw_exact = ar_whiten_matrix(X, Y, phi, exact_first_ar1=True)

        np.testing.assert_allclose(Xw_plain[0], X[0])
        np.testing.assert_allclose(Yw_plain[0], Y[0])
        np.testing.assert_allclose(Xw_exact[0], X[0] * np.sqrt(1.0 - np.mean(phi) ** 2))
        np.testing.assert_allclose(
            Yw_exact[0],
            Y[0] * np.sqrt(1.0 - phi.ravel() ** 2),
        )


class TestArmaBackendDispatch:
    def test_prefers_numba_backend_over_c_and_scipy(self, monkeypatch):
        y = np.random.RandomState(0).randn(40, 4)
        phi = np.array([0.4])
        theta = np.array([0.2])
        seg = np.array([0], dtype=np.intp)
        c_out = np.full_like(y, 11.0)
        numba_out = np.full_like(y, 22.0)
        calls = []

        def fake_c(y, phi, theta, seg_starts, do_exact):
            calls.append("c")
            return c_out

        def fake_numba(y, phi, theta, seg_starts, do_exact):
            calls.append("numba")
            return numba_out

        def fail_lfilter(*args, **kwargs):
            raise AssertionError("SciPy fallback should not be used when numba backend succeeds")

        monkeypatch.setattr(whitening_mod, "_USE_C_ARMA", True)
        monkeypatch.setattr(whitening_mod, "_USE_NUMBA_ARMA", True)
        monkeypatch.setattr(whitening_mod, "arma_whiten_segments_c", fake_c)
        monkeypatch.setattr(whitening_mod, "_arma_whiten_segments_numba_core", fake_numba)
        monkeypatch.setattr(whitening_mod, "lfilter", fail_lfilter)

        out = whitening_mod.arma_whiten_segments(y, phi, theta, seg)
        np.testing.assert_allclose(out, numba_out)
        assert calls == ["numba"]

    def test_uses_c_backend_when_numba_unavailable(self, monkeypatch):
        y = np.random.RandomState(1).randn(40, 4)
        phi = np.array([0.4])
        theta = np.array([0.2])
        seg = np.array([0], dtype=np.intp)
        c_out = np.full_like(y, 11.0)
        calls = []

        def fake_c(y, phi, theta, seg_starts, do_exact):
            calls.append("c")
            return c_out

        def fail_lfilter(*args, **kwargs):
            raise AssertionError("SciPy fallback should not be used when C backend succeeds")

        monkeypatch.setattr(whitening_mod, "_USE_C_ARMA", True)
        monkeypatch.setattr(whitening_mod, "_USE_NUMBA_ARMA", False)
        monkeypatch.setattr(whitening_mod, "arma_whiten_segments_c", fake_c)
        monkeypatch.setattr(whitening_mod, "lfilter", fail_lfilter)

        out = whitening_mod.arma_whiten_segments(y, phi, theta, seg)
        np.testing.assert_allclose(out, c_out)
        assert calls == ["c"]

    def test_falls_back_to_scipy_when_c_and_numba_unavailable(self, monkeypatch):
        y = np.random.RandomState(2).randn(40, 4)
        phi = np.array([0.4])
        theta = np.array([0.2])
        seg = np.array([0], dtype=np.intp)
        calls = []

        def fake_c(y, phi, theta, seg_starts, do_exact):
            calls.append("c")
            return None

        def fake_lfilter(b, a, x, axis=0):
            calls.append("lfilter")
            return np.full_like(x, 33.0)

        monkeypatch.setattr(whitening_mod, "_USE_C_ARMA", True)
        monkeypatch.setattr(whitening_mod, "_USE_NUMBA_ARMA", False)
        monkeypatch.setattr(whitening_mod, "arma_whiten_segments_c", fake_c)
        monkeypatch.setattr(whitening_mod, "lfilter", fake_lfilter)

        out = whitening_mod.arma_whiten_segments(y, phi, theta, seg)
        np.testing.assert_allclose(out, np.full_like(y, 33.0))
        assert calls == ["c", "lfilter"]


class TestWhitenApply:
    def test_whitening_type_hints_resolve(self):
        """Regression: forward-referenced plan/result annotations resolve."""
        import importlib

        mod = importlib.import_module("fmrimod.ar.whitening")
        typing.get_type_hints(mod.whiten_apply)
        typing.get_type_hints(mod.whiten)

    def test_global_plan(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        Y = _sim_ar1(200, 0.6, V=10)
        plan = WhiteningPlan(
            phi=[np.array([0.6])],
            theta=[np.array([])],
            order=(1, 0),
            pooling="global",
        )
        wr = whiten_apply(plan, X, Y)
        assert isinstance(wr, WhitenResult)
        assert wr.X.shape == X.shape
        assert wr.Y.shape == Y.shape

    def test_reduces_autocorrelation(self):
        Y = _sim_ar1(300, 0.6, V=10)
        plan = fit_noise(resid=Y, method="ar", p=1)
        X = np.random.RandomState(42).randn(300, 3)
        wr = whiten_apply(plan, X, Y)
        diag = acorr_diagnostics(wr.Y, max_lag=5)
        assert abs(diag["acf"][0]) < 0.15

    def test_run_plan(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        Y = _sim_ar1(200, 0.5, V=5)
        runs = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
        plan = fit_noise(resid=Y, runs=runs, method="ar", p=1, pooling="run")
        wr = whiten_apply(plan, X, Y, runs=runs)
        assert wr.X.shape == X.shape
        assert wr.Y.shape == Y.shape

    def test_parcel_plan(self):
        rng = np.random.RandomState(42)
        n, V = 200, 20
        X = rng.randn(n, 3)
        Y = _sim_ar1(n, 0.5, V=V)
        parcels = np.array([1]*5 + [2]*5 + [3]*5 + [4]*5)
        plan = fit_noise(resid=Y, method="ar", p=1, pooling="parcel", parcels=parcels)
        wr = whiten_apply(plan, X, Y, parcels=parcels)
        assert wr.X is None
        assert wr.X_by is not None
        assert wr.Y.shape == Y.shape


class TestWhitenConvenience:
    def test_basic(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        Y = _sim_ar1(200, 0.5, V=10)
        wr = whiten(X, Y, method="ar", p=1)
        assert isinstance(wr, WhitenResult)
        assert wr.X.shape == X.shape
        assert wr.Y.shape == Y.shape


class TestLogicalCensorMask:
    """fmriAR 617bd87 (test-audit-regressions.R:45-61), row AR-06.

    Cheap pass disqualified: converting the mask only inside ``whiten()``
    leaves the direct ``whiten_apply`` assertions red, and the parcel branch
    builds its segment starts separately from the global/run branch.
    """

    def _fixture(self):
        rng = np.random.default_rng(9103)
        R = _sim_ar1(200, 0.5, V=4, seed=9103)
        X = np.column_stack([np.ones(200), rng.standard_normal(200)])
        mask = np.zeros(200, dtype=bool)
        mask[[49, 119, 120]] = True
        return X, R, mask

    def test_whiten_apply_mask_equals_indices(self):
        X, R, mask = self._fixture()
        plan = fit_noise(R, p=1, censor=mask)
        np.testing.assert_array_equal(plan.censor, np.flatnonzero(mask))
        a = whiten_apply(plan, X, R, censor=mask)
        b = whiten_apply(plan, X, R, censor=np.flatnonzero(mask))
        np.testing.assert_array_equal(a.Y, b.Y)
        np.testing.assert_array_equal(a.X, b.X)
        with pytest.raises(ValueError, match="one entry per timepoint"):
            whiten_apply(plan, X, R, censor=mask[1:])

    def test_parcel_plan_mask_equals_indices(self):
        X, R, mask = self._fixture()
        parcels = np.array([1, 1, 2, 2])
        plan = fit_noise(R, p=1, censor=mask, pooling="parcel", parcels=parcels)
        a = whiten_apply(plan, X, R, censor=mask)
        b = whiten_apply(plan, X, R, censor=np.flatnonzero(mask))
        np.testing.assert_array_equal(a.Y, b.Y)
        for key in b.X_by:
            np.testing.assert_array_equal(a.X_by[key], b.X_by[key])

    def test_whiten_forwards_mask_to_both_steps(self):
        X, R, mask = self._fixture()
        w1 = whiten(X, R, censor=mask, p=1)
        w2 = whiten(X, R, censor=np.flatnonzero(mask), p=1)
        np.testing.assert_array_equal(w1.Y, w2.Y)

    def test_wrong_length_mask_is_refused_when_fitting(self):
        X, R, mask = self._fixture()
        with pytest.raises(ValueError, match="one entry per timepoint"):
            fit_noise(R, p=1, censor=mask[:-1])
        from fmrimod.ar.acvf import noise_acvf

        with pytest.raises(ValueError, match="one entry per timepoint"):
            noise_acvf(R, censor=mask[:-1])


def _arma_acvf_oracle(phi, theta, max_lag, n_ma=5000):
    """Unit-innovation ARMA autocovariance from the MA(inf) weights."""
    from scipy.signal import lfilter

    imp = np.zeros(n_ma)
    imp[0] = 1.0
    psi = lfilter(
        np.concatenate([[1.0], np.asarray(theta, dtype=float)]),
        np.concatenate([[1.0], -np.asarray(phi, dtype=float)]),
        imp,
    )
    return np.array([psi[: n_ma - k] @ psi[k:] for k in range(max_lag + 1)])


def _gls_oracle(y, phi, theta):
    """Exact GLS whitening: forward-solve with chol of the stationary Sigma."""
    n = len(y)
    g = _arma_acvf_oracle(phi, theta, n - 1)
    S = g[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    return np.linalg.solve(np.linalg.cholesky(S), y)


class TestExactStationaryStart:
    """fmriAR 617bd87: Ansley exact start-up (test-audit-regressions.R:118-180).

    Row AR-02. Cheap pass disqualified: rescaling only the first sample by a
    variance ratio leaves samples 1..m-1 and the MA tail wrong, which the
    dense-Cholesky GLS oracle catches.
    """

    CASES = [
        ([0.5], []),
        ([1.2, -0.5], []),
        ([0.3, 0.2, 0.1], []),
        ([0.5], [0.4]),
        ([0.6, 0.2], [0.3, -0.2]),
        ([], [0.7]),
        ([0.3, 0.2, 0.1], [0.8]),
    ]

    def test_matches_dense_cholesky_gls_per_segment(self):
        rng = np.random.default_rng(9108)
        starts = [0, 30, 33]  # includes a 3-frame segment
        for phi, theta in self.CASES:
            y = rng.standard_normal(70)
            out = arma_whiten_segments(
                y, np.array(phi), np.array(theta), np.array(starts),
                exact_first_ar1=True,
            )
            ref = np.concatenate(
                [_gls_oracle(y[a:b], phi, theta)
                 for a, b in zip(starts, starts[1:] + [70], strict=True)]
            )
            np.testing.assert_allclose(out, ref, rtol=0, atol=1e-10,
                                       err_msg=f"phi={phi} theta={theta}")

    def test_columns_are_whitened_independently(self):
        rng = np.random.default_rng(1)
        Y = rng.standard_normal((300, 4))
        phi, theta = np.array([0.6, 0.2]), np.array([0.3, -0.2])
        starts = np.array([0, 120, 121, 200])
        joint = arma_whiten_segments(Y, phi, theta, starts, exact_first_ar1=True)
        for j in range(Y.shape[1]):
            np.testing.assert_allclose(
                joint[:, j],
                arma_whiten_segments(Y[:, j], phi, theta, starts, exact_first_ar1=True),
                rtol=0, atol=1e-12,
            )
        # Long segments go past the factor's converged rows.
        ref = _gls_oracle(Y[200:, 0], phi, theta)
        np.testing.assert_allclose(joint[200:, 0], ref, rtol=0, atol=1e-10)

    def test_ar1_start_is_prais_winsten(self):
        rng = np.random.default_rng(9109)
        y = rng.standard_normal(50)
        out = arma_whiten_segments(y, np.array([0.7]), np.array([]), np.array([0]),
                                   exact_first_ar1=True)
        np.testing.assert_allclose(out[0], y[0] * np.sqrt(1 - 0.49), rtol=1e-14)
        np.testing.assert_allclose(out[1:], y[1:] - 0.7 * y[:-1], rtol=0, atol=1e-14)

    def test_whitened_noise_has_unit_variance_at_run_starts(self):
        from scipy.signal import lfilter

        from fmrimod.ar.plan import plan_from_phi

        rng = np.random.default_rng(9110)
        for phi, theta in (([1.2, -0.5], []), ([0.6], [0.5])):
            e = rng.standard_normal((230, 3000))
            Y = lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi)], e, axis=0)[200:]
            plan = plan_from_phi(np.array(phi), np.array(theta), exact_first=True)
            W = whiten_apply(plan, np.ones((30, 1)), Y).Y
            v = W.var(axis=1, ddof=1)
            assert np.all(np.abs(v - 1) < 0.1), np.round(v[:4], 2)
            assert abs(np.corrcoef(W[0], W[1])[0, 1]) < 0.06

    def test_nonstationary_filter_falls_back_to_conditional_recursion(self):
        y = np.arange(1.0, 6.0)
        out = arma_whiten_segments(y, np.array([1.0]), np.array([]), np.array([0]),
                                   exact_first_ar1=True)
        np.testing.assert_allclose(out, np.ones(5))
