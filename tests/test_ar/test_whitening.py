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
