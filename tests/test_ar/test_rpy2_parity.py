"""Optional cross-language parity tests: Python fmrimod.ar vs R fmriAR.

These tests execute R's fmriAR directly from Python via rpy2 and compare
the resulting outputs against Python implementations.

They are marked with ``@pytest.mark.rpy2`` and skip automatically when rpy2
or the fmriAR R package is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pytest

from fmrimod.ar.numhelpers import (
    ar_to_pacf,
    enforce_invertible_ma,
    enforce_stationary_ar,
    pacf_to_ar,
)
from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.hr_arma import hr_arma
from fmrimod.ar.whitening import whiten_apply
from fmrimod.ar.diagnostics import sandwich_from_whitened_resid


@dataclass
class ARContext:
    """Container for lazily imported R bridge objects."""

    fmriAR: object
    FloatVector: object
    IntVector: object
    r: object  # rpy2.robjects.r


@pytest.fixture(scope="module")
def rctx() -> ARContext:
    """Provide an R fmriAR interop context or skip when unavailable."""
    pytest.importorskip("rpy2")

    from rpy2.robjects.packages import PackageNotInstalledError, importr
    from rpy2.robjects.vectors import FloatVector, IntVector
    import rpy2.robjects as ro

    try:
        fmriAR = importr("fmriAR")
    except PackageNotInstalledError as exc:
        pytest.skip(f"fmriAR R package not installed: {exc}")

    return ARContext(
        fmriAR=fmriAR,
        FloatVector=FloatVector,
        IntVector=IntVector,
        r=ro.r,
    )


def _to_r_matrix(arr, rctx):
    """Convert numpy array to R matrix."""
    import rpy2.robjects as ro

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n, p = arr.shape
    return ro.r.matrix(
        rctx.FloatVector(arr.ravel(order="F")), nrow=n, ncol=p
    )


def _r_null_to_none(val):
    """Convert R NULL to Python None."""
    import rpy2.rinterface_lib.sexp

    if isinstance(val, rpy2.rinterface_lib.sexp.NULLType):
        return None
    return val


# ---------------------------------------------------------------------------
# Helper function parity
# ---------------------------------------------------------------------------


@pytest.mark.rpy2
class TestPACFParity:
    """pacf_to_ar / ar_to_pacf match R fmriAR implementations."""

    @pytest.mark.parametrize(
        "kappa",
        [
            np.array([0.6]),
            np.array([0.6, -0.3]),
            np.array([0.6, -0.3, 0.1]),
            np.array([0.9, -0.5, 0.3, -0.1]),
        ],
    )
    def test_pacf_to_ar(self, rctx, kappa):
        py_phi = pacf_to_ar(kappa)
        r_phi = np.array(rctx.fmriAR.pacf_to_ar(rctx.FloatVector(kappa)))
        np.testing.assert_allclose(py_phi, r_phi, atol=1e-12)

    @pytest.mark.parametrize(
        "phi",
        [
            np.array([0.7]),
            np.array([0.5, -0.2]),
            np.array([0.6, -0.3, 0.1]),
        ],
    )
    def test_ar_to_pacf(self, rctx, phi):
        py_kappa = ar_to_pacf(phi)
        r_kappa = np.array(rctx.fmriAR.ar_to_pacf(rctx.FloatVector(phi)))
        np.testing.assert_allclose(py_kappa, r_kappa, atol=1e-12)

    @pytest.mark.parametrize(
        "kappa",
        [
            np.array([0.6, -0.3]),
            np.array([0.9, -0.5, 0.3, -0.1]),
        ],
    )
    def test_round_trip_matches_r(self, rctx, kappa):
        """Full round-trip: Python pacf->ar->pacf matches R pacf->ar->pacf."""
        py_phi = pacf_to_ar(kappa)
        r_phi = np.array(rctx.fmriAR.pacf_to_ar(rctx.FloatVector(kappa)))
        np.testing.assert_allclose(py_phi, r_phi, atol=1e-12)

        py_back = ar_to_pacf(py_phi)
        r_back = np.array(rctx.fmriAR.ar_to_pacf(rctx.FloatVector(r_phi)))
        np.testing.assert_allclose(py_back, r_back, atol=1e-12)
        np.testing.assert_allclose(py_back, kappa, atol=1e-10)


@pytest.mark.rpy2
class TestEnforceParity:
    """enforce_stationary_ar / enforce_invertible_ma match R."""

    @pytest.mark.parametrize(
        "phi",
        [
            np.array([0.5]),
            np.array([1.5]),
            np.array([-1.2]),
            np.array([1.5, -0.8, 0.3]),
        ],
    )
    def test_enforce_stationary_ar(self, rctx, phi):
        py_result = enforce_stationary_ar(phi)
        r_result = np.array(
            rctx.fmriAR.enforce_stationary_ar(rctx.FloatVector(phi))
        )
        np.testing.assert_allclose(py_result, r_result, atol=1e-10)

    def test_enforce_stationary_ar_bound(self, rctx):
        phi = np.array([0.95])
        py_result = enforce_stationary_ar(phi, bound=0.9)
        r_result = np.array(
            rctx.fmriAR.enforce_stationary_ar(rctx.FloatVector(phi), bound=0.9)
        )
        np.testing.assert_allclose(py_result, r_result, atol=1e-10)

    @pytest.mark.parametrize(
        "theta",
        [
            np.array([0.3]),
            np.array([2.0]),
            np.array([1.5, 0.8]),
        ],
    )
    def test_enforce_invertible_ma(self, rctx, theta):
        py_result = enforce_invertible_ma(theta)
        r_result = np.array(
            rctx.fmriAR.enforce_invertible_ma(rctx.FloatVector(theta))
        )
        np.testing.assert_allclose(py_result, r_result, atol=1e-10)


# ---------------------------------------------------------------------------
# fit_noise parity
# ---------------------------------------------------------------------------


def _generate_ar1_data(n, phi, seed=42):
    """Generate AR(1) time series."""
    rng = np.random.RandomState(seed)
    y = np.zeros(n)
    e = rng.randn(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


def _generate_ar2_data(n, phi, seed=42):
    """Generate AR(2) time series."""
    rng = np.random.RandomState(seed)
    y = np.zeros(n)
    e = rng.randn(n)
    for t in range(2, n):
        y[t] = phi[0] * y[t - 1] + phi[1] * y[t - 2] + e[t]
    return y


def _generate_arma11_data(n, phi, theta, seed=42):
    """Generate ARMA(1,1) time series."""
    rng = np.random.RandomState(seed)
    y = np.zeros(n)
    e = rng.randn(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t] + theta * e[t - 1]
    return y


@pytest.mark.rpy2
class TestFitNoiseParity:
    """fit_noise() output matches R fmriAR::fit_noise()."""

    def test_ar1_global(self, rctx):
        """AR(1) with global pooling."""
        y = _generate_ar1_data(300, 0.6, seed=42)

        # Python
        py_plan = fit_noise(y[:, np.newaxis], method="ar", p="auto", p_max=6,
                            exact_first=True, pooling="global")

        # R
        r_resid = _to_r_matrix(y, rctx)
        r_plan = rctx.fmriAR.fit_noise(
            resid=r_resid, method="ar", p="auto", p_max=6,
            exact_first="ar1", pooling="global"
        )

        r_phi = np.array(r_plan.rx2("phi")).ravel()
        r_order = np.array(r_plan.rx2("order"))

        py_phi = py_plan.phi[0]

        # Orders should match
        assert py_plan.order[0] == int(r_order[0]), (
            f"AR order mismatch: Python {py_plan.order[0]} vs R {int(r_order[0])}"
        )
        # Coefficients should be close
        np.testing.assert_allclose(py_phi, r_phi, atol=0.05,
                                   err_msg="AR coefficients differ")

    def test_ar2_global(self, rctx):
        """AR(2) with global pooling."""
        y = _generate_ar2_data(400, [0.5, -0.3], seed=123)

        py_plan = fit_noise(y[:, np.newaxis], method="ar", p="auto", p_max=6,
                            exact_first=True, pooling="global")

        r_resid = _to_r_matrix(y, rctx)
        r_plan = rctx.fmriAR.fit_noise(
            resid=r_resid, method="ar", p="auto", p_max=6,
            exact_first="ar1", pooling="global"
        )

        r_phi = np.array(r_plan.rx2("phi")).ravel()
        r_order = np.array(r_plan.rx2("order"))

        py_phi = py_plan.phi[0]

        assert py_plan.order[0] == int(r_order[0])
        np.testing.assert_allclose(py_phi, r_phi, atol=0.05)

    def test_ar_fixed_order(self, rctx):
        """Fixed AR(1) order (no BIC selection)."""
        y = _generate_ar1_data(200, 0.7, seed=99)

        py_plan = fit_noise(y[:, np.newaxis], method="ar", p=1,
                            exact_first=True, pooling="global")

        r_resid = _to_r_matrix(y, rctx)
        r_plan = rctx.fmriAR.fit_noise(
            resid=r_resid, method="ar", p=1,
            exact_first="ar1", pooling="global"
        )

        r_phi = np.array(r_plan.rx2("phi")).ravel()
        py_phi = py_plan.phi[0]

        np.testing.assert_allclose(py_phi, r_phi, atol=0.02,
                                   err_msg="Fixed AR(1) phi mismatch")

    def test_multirun_global(self, rctx):
        """Multi-run data with global pooling."""
        y1 = _generate_ar1_data(150, 0.5, seed=10)
        y2 = _generate_ar1_data(150, 0.5, seed=20)
        y = np.concatenate([y1, y2])
        runs = np.concatenate([np.zeros(150), np.ones(150)]).astype(int)

        py_plan = fit_noise(y[:, np.newaxis], method="ar", p=1,
                            runs=runs, exact_first=True, pooling="global")

        r_resid = _to_r_matrix(y, rctx)
        r_runs = rctx.IntVector(runs + 1)  # R is 1-based
        r_plan = rctx.fmriAR.fit_noise(
            resid=r_resid, method="ar", p=1, runs=r_runs,
            exact_first="ar1", pooling="global"
        )

        r_phi = np.array(r_plan.rx2("phi")).ravel()
        py_phi = py_plan.phi[0]

        np.testing.assert_allclose(py_phi, r_phi, atol=0.05)

    def test_multirun_per_run(self, rctx):
        """Multi-run data with per-run pooling (fixed order to avoid BIC differences)."""
        y1 = _generate_ar1_data(200, 0.4, seed=10)
        y2 = _generate_ar1_data(200, 0.7, seed=20)
        y = np.concatenate([y1, y2])
        runs = np.concatenate([np.zeros(200), np.ones(200)]).astype(int)

        # Use fixed p=1 to compare coefficients directly
        py_plan = fit_noise(y[:, np.newaxis], method="ar", p=1,
                            runs=runs, exact_first=True, pooling="run")

        r_resid = _to_r_matrix(y, rctx)
        r_runs = rctx.IntVector(runs + 1)
        r_plan = rctx.fmriAR.fit_noise(
            resid=r_resid, method="ar", p=1, runs=r_runs,
            exact_first="ar1", pooling="run"
        )

        # R returns a list of per-run phi vectors
        r_phi_list = r_plan.rx2("phi")
        for i in range(2):
            r_phi_i = np.array(r_phi_list.rx2(i + 1)).ravel()  # R is 1-based
            np.testing.assert_allclose(
                py_plan.phi[i], r_phi_i, atol=0.05,
                err_msg=f"Run {i} phi mismatch"
            )


# ---------------------------------------------------------------------------
# hr_arma parity
# ---------------------------------------------------------------------------


@pytest.mark.rpy2
class TestHrArmaParity:
    """hr_arma() matches R fmriAR::hr_arma()."""

    def test_arma11(self, rctx):
        """ARMA(1,1) estimation."""
        y = _generate_arma11_data(500, 0.6, 0.4, seed=42)

        py_result = hr_arma(y, p=1, q=1)
        r_result = rctx.fmriAR.hr_arma(rctx.FloatVector(y), p=1, q=1)

        r_phi = np.array(r_result.rx2("phi"))
        r_theta = np.array(r_result.rx2("theta"))

        np.testing.assert_allclose(py_result["phi"], r_phi, atol=0.05,
                                   err_msg="HR phi mismatch")
        np.testing.assert_allclose(py_result["theta"], r_theta, atol=0.05,
                                   err_msg="HR theta mismatch")

    def test_arma21(self, rctx):
        """ARMA(2,1) estimation."""
        rng = np.random.RandomState(123)
        n = 600
        y = np.zeros(n)
        e = rng.randn(n)
        phi1, phi2, theta1 = 0.5, -0.2, 0.3
        for t in range(2, n):
            y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + e[t] + theta1 * e[t - 1]

        py_result = hr_arma(y, p=2, q=1)
        r_result = rctx.fmriAR.hr_arma(rctx.FloatVector(y), p=2, q=1)

        r_phi = np.array(r_result.rx2("phi"))
        r_theta = np.array(r_result.rx2("theta"))

        np.testing.assert_allclose(py_result["phi"], r_phi, atol=0.08,
                                   err_msg="HR(2,1) phi mismatch")
        np.testing.assert_allclose(py_result["theta"], r_theta, atol=0.08,
                                   err_msg="HR(2,1) theta mismatch")

    def test_sigma2(self, rctx):
        """Innovation variance matches."""
        y = _generate_arma11_data(500, 0.6, 0.4, seed=42)

        py_result = hr_arma(y, p=1, q=1)
        r_result = rctx.fmriAR.hr_arma(rctx.FloatVector(y), p=1, q=1)

        r_sigma2 = np.array(r_result.rx2("sigma2")).item()
        np.testing.assert_allclose(py_result["sigma2"], r_sigma2, rtol=0.1,
                                   err_msg="HR sigma2 mismatch")


# ---------------------------------------------------------------------------
# whiten_apply parity
# ---------------------------------------------------------------------------


@pytest.mark.rpy2
class TestWhitenApplyParity:
    """whiten_apply() matches R fmriAR::whiten_apply()."""

    def test_ar1_whitening(self, rctx):
        """AR(1) whitening of X and Y."""
        np.random.seed(42)
        n = 200
        y = _generate_ar1_data(n, 0.6, seed=42)
        X = np.column_stack([np.ones(n), np.random.RandomState(42).randn(n)])

        # Fit with both
        r_resid = _to_r_matrix(y, rctx)
        r_plan = rctx.fmriAR.fit_noise(
            resid=r_resid, method="ar", p=1,
            exact_first="ar1", pooling="global"
        )

        # R whiten
        r_X = _to_r_matrix(X, rctx)
        r_Y = _to_r_matrix(y, rctx)
        r_result = rctx.fmriAR.whiten_apply(plan=r_plan, X=r_X, Y=r_Y)
        r_Xw = np.array(r_result.rx2("X"))
        r_Yw = np.array(r_result.rx2("Y")).ravel()

        # Python whiten with same phi
        r_phi = np.array(r_plan.rx2("phi")).ravel()
        from fmrimod.ar.plan import WhiteningPlan
        py_plan = WhiteningPlan(
            phi=[r_phi],
            theta=[np.array([])],
            order=(len(r_phi), 0),
            method="ar",
            pooling="global",
            exact_first=True,
        )
        py_result = whiten_apply(py_plan, X, y[:, np.newaxis])

        np.testing.assert_allclose(py_result.X, r_Xw, atol=1e-10,
                                   err_msg="Whitened X mismatch")
        np.testing.assert_allclose(py_result.Y.ravel(), r_Yw, atol=1e-10,
                                   err_msg="Whitened Y mismatch")

    def test_ar2_whitening(self, rctx):
        """AR(2) whitening."""
        np.random.seed(123)
        n = 300
        y = _generate_ar2_data(n, [0.5, -0.3], seed=123)
        X = np.column_stack([np.ones(n), np.random.RandomState(123).randn(n)])

        r_resid = _to_r_matrix(y, rctx)
        r_plan = rctx.fmriAR.fit_noise(
            resid=r_resid, method="ar", p=2,
            exact_first="none", pooling="global"
        )

        r_X = _to_r_matrix(X, rctx)
        r_Y = _to_r_matrix(y, rctx)
        r_result = rctx.fmriAR.whiten_apply(plan=r_plan, X=r_X, Y=r_Y)
        r_Xw = np.array(r_result.rx2("X"))
        r_Yw = np.array(r_result.rx2("Y")).ravel()

        r_phi = np.array(r_plan.rx2("phi")).ravel()
        from fmrimod.ar.plan import WhiteningPlan
        py_plan = WhiteningPlan(
            phi=[r_phi],
            theta=[np.array([])],
            order=(len(r_phi), 0),
            method="ar",
            pooling="global",
            exact_first=False,
        )
        py_result = whiten_apply(py_plan, X, y[:, np.newaxis])

        np.testing.assert_allclose(py_result.X, r_Xw, atol=1e-10,
                                   err_msg="Whitened X mismatch")
        np.testing.assert_allclose(py_result.Y.ravel(), r_Yw, atol=1e-10,
                                   err_msg="Whitened Y mismatch")

    def test_multirun_whitening(self, rctx):
        """Multi-run whitening with per-run phi."""
        y1 = _generate_ar1_data(150, 0.4, seed=10)
        y2 = _generate_ar1_data(150, 0.7, seed=20)
        y = np.concatenate([y1, y2])
        runs = np.concatenate([np.zeros(150), np.ones(150)]).astype(int)
        X = np.column_stack([np.ones(300), np.random.RandomState(42).randn(300)])

        # Fit per-run in R
        r_resid = _to_r_matrix(y, rctx)
        r_runs = rctx.IntVector(runs + 1)
        r_plan = rctx.fmriAR.fit_noise(
            resid=r_resid, method="ar", p=1, runs=r_runs,
            exact_first="ar1", pooling="run"
        )

        r_X = _to_r_matrix(X, rctx)
        r_Y = _to_r_matrix(y, rctx)
        r_result = rctx.fmriAR.whiten_apply(
            plan=r_plan, X=r_X, Y=r_Y, runs=r_runs
        )
        r_Xw = np.array(r_result.rx2("X"))
        r_Yw = np.array(r_result.rx2("Y")).ravel()

        # Build Python plan from R phi (R returns a list of per-run vectors)
        r_phi_list = r_plan.rx2("phi")
        n_runs = len(r_phi_list)
        from fmrimod.ar.plan import WhiteningPlan
        py_plan = WhiteningPlan(
            phi=[np.array(r_phi_list.rx2(i + 1)).ravel() for i in range(n_runs)],
            theta=[np.array([]) for _ in range(n_runs)],
            order=(1, 0),
            method="ar",
            pooling="run",
            exact_first=True,
            runs=runs,
        )
        py_result = whiten_apply(py_plan, X, y[:, np.newaxis], runs=runs)

        np.testing.assert_allclose(py_result.X, r_Xw, atol=1e-10,
                                   err_msg="Multi-run whitened X mismatch")
        np.testing.assert_allclose(py_result.Y.ravel(), r_Yw, atol=1e-10,
                                   err_msg="Multi-run whitened Y mismatch")


# ---------------------------------------------------------------------------
# sandwich SE parity
# ---------------------------------------------------------------------------


@pytest.mark.rpy2
class TestSandwichParity:
    """sandwich_from_whitened_resid() matches R fmriAR."""

    def test_iid(self, rctx):
        """IID sandwich SEs match R."""
        rng = np.random.RandomState(42)
        n, p, v = 100, 3, 5
        Xw = rng.randn(n, p)
        Yw = rng.randn(n, v)

        py_result = sandwich_from_whitened_resid(Xw, Yw, type="iid")

        r_Xw = _to_r_matrix(Xw, rctx)
        r_Yw = _to_r_matrix(Yw, rctx)
        r_result = rctx.fmriAR.sandwich_from_whitened_resid(
            Xw=r_Xw, Yw=r_Yw, type="iid"
        )

        r_se = np.array(r_result.rx2("se"))
        r_sigma2 = np.array(r_result.rx2("sigma2"))
        r_XtX_inv = np.array(r_result.rx2("XtX_inv"))

        np.testing.assert_allclose(py_result["se"], r_se, atol=1e-10,
                                   err_msg="IID SE mismatch")
        np.testing.assert_allclose(py_result["sigma2"], r_sigma2, atol=1e-10,
                                   err_msg="IID sigma2 mismatch")
        np.testing.assert_allclose(py_result["XtX_inv"], r_XtX_inv, atol=1e-10,
                                   err_msg="IID XtX_inv mismatch")

    def test_hc0(self, rctx):
        """HC0 sandwich SEs match R."""
        rng = np.random.RandomState(99)
        n, p, v = 80, 2, 4
        Xw = rng.randn(n, p)
        Yw = rng.randn(n, v)

        py_result = sandwich_from_whitened_resid(Xw, Yw, type="HC0")

        r_Xw = _to_r_matrix(Xw, rctx)
        r_Yw = _to_r_matrix(Yw, rctx)
        r_result = rctx.fmriAR.sandwich_from_whitened_resid(
            Xw=r_Xw, Yw=r_Yw, type="hc0"  # R uses lowercase
        )

        r_se = np.array(r_result.rx2("se"))
        np.testing.assert_allclose(py_result["se"], r_se, atol=1e-10,
                                   err_msg="HC0 SE mismatch")
