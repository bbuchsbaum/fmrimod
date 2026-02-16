"""Tests for WhiteningPlan and WhitenResult dataclasses."""

import numpy as np
import pytest

from fmrimod.ar.plan import WhiteningPlan, WhitenResult


class TestWhiteningPlan:
    def test_default_creation(self):
        plan = WhiteningPlan()
        assert plan.order == (0, 0)
        assert plan.method == "ar"
        assert plan.pooling == "global"
        assert plan.exact_first is False
        assert plan.phi is None
        assert plan.theta is None

    def test_global_plan(self):
        phi = [np.array([0.5])]
        plan = WhiteningPlan(
            phi=phi,
            theta=[np.array([])],
            order=(1, 0),
            method="ar",
            pooling="global",
            exact_first=True,
        )
        assert plan.order == (1, 0)
        assert plan.exact_first is True
        assert len(plan.phi) == 1
        np.testing.assert_array_equal(plan.phi[0], [0.5])

    def test_run_plan(self):
        phi = [np.array([0.4]), np.array([0.6])]
        plan = WhiteningPlan(
            phi=phi,
            theta=[np.array([]), np.array([])],
            order=(1, 0),
            pooling="run",
            runs=np.concatenate([np.zeros(100), np.ones(100)]).astype(int),
        )
        assert plan.pooling == "run"
        assert len(plan.phi) == 2

    def test_parcel_plan(self):
        plan = WhiteningPlan(
            order=(2, 0),
            pooling="parcel",
            parcels=np.array([1, 1, 2, 2, 3, 3]),
            parcel_ids=["1", "2", "3"],
            phi_by_parcel={
                "1": np.array([0.5, -0.1]),
                "2": np.array([0.4, -0.2]),
                "3": np.array([0.3, -0.1]),
            },
            theta_by_parcel={
                "1": np.array([]),
                "2": np.array([]),
                "3": np.array([]),
            },
        )
        assert plan.pooling == "parcel"
        assert len(plan.phi_by_parcel) == 3
        assert plan.order == (2, 0)

    def test_arma_plan(self):
        plan = WhiteningPlan(
            phi=[np.array([0.5])],
            theta=[np.array([0.3])],
            order=(1, 1),
            method="arma",
        )
        assert plan.method == "arma"
        assert plan.order == (1, 1)

    def test_repr_global(self):
        plan = WhiteningPlan(
            phi=[np.array([0.5, -0.2])],
            theta=[np.array([])],
            order=(2, 0),
            method="ar",
            pooling="global",
        )
        r = repr(plan)
        assert "WhiteningPlan" in r
        assert "AR" in r
        assert "p=2" in r
        assert "global" in r

    def test_repr_parcel(self):
        plan = WhiteningPlan(
            order=(1, 0),
            pooling="parcel",
            phi_by_parcel={"1": np.array([0.5]), "2": np.array([0.3])},
        )
        r = repr(plan)
        assert "parcel" in r
        assert "2 parcels" in r


class TestWhitenResult:
    def test_creation(self):
        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 10)
        wr = WhitenResult(X=X, Y=Y)
        assert wr.X.shape == (100, 3)
        assert wr.Y.shape == (100, 10)
        assert wr.X_by is None

    def test_parcel_result(self):
        Y = np.random.randn(100, 10)
        X_by = {"1": np.random.randn(100, 3), "2": np.random.randn(100, 3)}
        wr = WhitenResult(X=None, Y=Y, X_by=X_by)
        assert wr.X is None
        assert len(wr.X_by) == 2
