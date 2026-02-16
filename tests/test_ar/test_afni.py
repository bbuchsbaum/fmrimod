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
