"""Tests for AFNI restricted AR parameterisation."""

import numpy as np
import pytest

from fmrimod.ar.afni import afni_phi_ar3, afni_phi_ar5, afni_restricted_plan
from fmrimod.ar.plan import WhiteningPlan


class TestAfniPhiAR3:
    def test_basic(self):
        phi = afni_phi_ar3(0.5, 0.8, 1.0)
        assert len(phi) == 3
        assert phi.dtype == np.float64

    def test_zero_damping(self):
        phi = afni_phi_ar3(0.0, 0.8, 1.0)
        assert len(phi) == 3
        # With a=0: p1 = 2*r1*cos(t1), p2 = -r1^2, p3 = 0
        np.testing.assert_allclose(phi[2], 0.0, atol=1e-10)

    def test_clipping(self):
        phi = afni_phi_ar3(2.0, 2.0, 5.0)
        assert len(phi) == 3  # should not crash

    def test_known_values(self):
        # a=0.5, r1=0.0, t1=0 => p1=0.5, p2=0, p3=0
        phi = afni_phi_ar3(0.5, 0.0, 0.0)
        np.testing.assert_allclose(phi, [0.5, 0.0, 0.0], atol=1e-10)


class TestAfniPhiAR5:
    def test_basic(self):
        phi = afni_phi_ar5(0.3, 0.7, 1.2, 0.5, 0.8)
        assert len(phi) == 5
        assert phi.dtype == np.float64

    def test_reduces_to_ar3(self):
        # r2=0 should make AR(5) mostly like AR(3) in first 3 coefficients
        phi3 = afni_phi_ar3(0.5, 0.8, 1.0)
        phi5 = afni_phi_ar5(0.5, 0.8, 1.0, 0.0, 0.0)
        np.testing.assert_allclose(phi5[:3], phi3, atol=1e-10)
        np.testing.assert_allclose(phi5[3:], [0.0, 0.0], atol=1e-10)


class TestAfniRestrictedPlan:
    def test_global_plan(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 10)
        roots = {"a": 0.5, "r1": 0.7, "t1": 1.0}
        plan = afni_restricted_plan(resid, p=3, roots=roots, estimate_ma1=False)
        assert isinstance(plan, WhiteningPlan)
        assert plan.method == "afni"
        assert plan.order[0] == 3

    def test_with_ma1(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 10)
        roots = {"a": 0.3, "r1": 0.5, "t1": 1.0}
        plan = afni_restricted_plan(resid, p=3, roots=roots, estimate_ma1=True)
        assert plan.method == "afni"

    def test_parcel_plan(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 6)
        parcels = np.array([1, 1, 2, 2, 3, 3])
        roots = {"a": 0.3, "r1": 0.5, "t1": 1.0}
        plan = afni_restricted_plan(resid, p=3, roots=roots,
                                     parcels=parcels, estimate_ma1=False)
        assert plan.pooling == "parcel"
        assert plan.phi_by_parcel is not None

    def test_invalid_order(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 10)
        with pytest.raises(ValueError, match="p must be 3 or 5"):
            afni_restricted_plan(resid, p=4, roots={"a": 0.5, "r1": 0.5, "t1": 1.0})


class TestAfniMa1FromVoxels:
    """fmriAR 617bd87 + 5f5fc32 (test-audit-regressions.R:255-266), row AR-11.

    Cheap pass disqualified: returning theta = 0 passes the first fixture
    only; the second has a true MA(1) of 0.4 after the AR(3) filter.
    """

    def test_shared_component_does_not_leak_into_the_ma_term(self):
        from scipy.signal import lfilter

        rng = np.random.default_rng(9115)
        n, v = 300, 100
        # Weak shared component: negligible in each voxel, dominant in the mean.
        shared = lfilter([1.0], [1.0, -0.95], rng.standard_normal(n))
        R = rng.standard_normal((n, v)) + 0.05 * shared[:, None]
        plan = afni_restricted_plan(R, p=3, roots={"a": 0, "r1": 0, "t1": 0})
        assert abs(plan.theta[0][0]) < 0.06
        pplan = afni_restricted_plan(R, p=3, roots={"a": 0, "r1": 0, "t1": 0},
                                     parcels=np.repeat([1, 2], 50))
        for th in pplan.theta_by_parcel.values():
            assert abs(th[0]) < 0.08

    def test_recovers_the_ma1_left_by_the_ar_filter_in_every_run(self):
        from scipy.signal import lfilter

        rng = np.random.default_rng(4)
        phi = afni_phi_ar3(0.5, 0.6, 1.0)
        runs = np.repeat([1, 2], 150)
        # ARMA(3, 1) per run: the AR(3) filter leaves an exact MA(1) of 0.4.
        R = np.vstack([
            lfilter([1.0, 0.4], np.r_[1.0, -phi],
                    rng.standard_normal((350, 40)), axis=0)[200:]
            for _ in (1, 2)
        ])
        plan = afni_restricted_plan(R, p=3, roots={"a": 0.5, "r1": 0.6, "t1": 1.0},
                                    runs=runs)
        assert abs(plan.theta[0][0] - 0.4) < 0.05


class TestAfniSpecValidation:
    """AR-20 (fmriAR 5f5fc32): AFNI rejects bad specs instead of clamping."""

    R = np.random.default_rng(0).standard_normal((100, 2))

    @pytest.mark.parametrize(
        ("roots", "match"),
        [
            ({"a": -0.2, "r1": 0.3, "t1": 1.0}, ">= 0"),
            ({"a": 0.5, "r1": -0.3, "t1": 1.0}, ">= 0"),
            ({"a": 0.5, "r1": 0.3}, "missing: t1"),
            ({"a": 0.5, "r1": np.nan, "t1": 1.0}, "finite"),
            ({"a": 0.5, "r1": 0.3, "t1": 1.0, "vrt": 0}, "vrt"),
            ({"a": 0.5, "r1": 0.3, "t1": 1.0, "vrt": np.inf}, "vrt"),
        ],
    )
    def test_bad_specs_raise(self, roots, match):
        with pytest.raises(ValueError, match=match):
            afni_restricted_plan(self.R, roots=roots, estimate_ma1=False)
        parcels = np.array([1, 2])
        with pytest.raises(ValueError, match=match):
            afni_restricted_plan(
                self.R, roots=roots, parcels=parcels, estimate_ma1=False
            )

    def test_p5_requires_second_pair_and_nonnegative_r2(self):
        with pytest.raises(ValueError, match="missing: r2, t2"):
            afni_restricted_plan(self.R, p=5, roots={"a": 0.5, "r1": 0.3, "t1": 1})
        bad = {"a": 0.5, "r1": 0.3, "t1": 1, "r2": -0.1, "t2": 2}
        with pytest.raises(ValueError, match=">= 0"):
            afni_restricted_plan(self.R, p=5, roots=bad)

    def test_parcel_order_q_reflects_the_estimated_theta(self):
        roots = {"a": 0.5, "r1": 0.3, "t1": 1.0}
        plan = afni_restricted_plan(
            self.R, roots=roots, parcels=np.array([1, 2]), estimate_ma1=False
        )
        assert plan.order == (3, 0)
        # MA(1) estimation that cannot run (runs too short) leaves q = 0.
        short = self.R[:12]
        plan = afni_restricted_plan(
            short, roots=roots, parcels=np.array([1, 2]), estimate_ma1=True
        )
        q_actual = max(len(t) for t in plan.theta_by_parcel.values())
        assert plan.order[1] == q_actual
