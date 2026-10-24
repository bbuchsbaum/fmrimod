"""Tests for multi-scale spatial pooling."""

import numpy as np
import pytest

from fmrimod.ar.multiscale import (
    ms_combine_to_fine,
    ms_dispersion,
    ms_estimate_scale,
    ms_parent_maps,
    ms_weights,
    parcel_means,
)
from fmrimod.ar.numhelpers import segmented_acvf


class TestParcelMeans:
    def test_basic(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(100, 6)
        parcels = np.array([1, 1, 2, 2, 3, 3])
        M = parcel_means(resid, parcels)
        assert len(M) == 3
        for key in ["1", "2", "3"]:
            assert key in M
            assert M[key].shape == (100,)

    def test_matches_manual(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(50, 4)
        parcels = np.array([1, 1, 2, 2])
        M = parcel_means(resid, parcels)
        expected_1 = resid[:, :2].mean(axis=1)
        expected_2 = resid[:, 2:].mean(axis=1)
        np.testing.assert_allclose(M["1"], expected_1)
        np.testing.assert_allclose(M["2"], expected_2)

    def test_single_voxel_parcels(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(50, 3)
        parcels = np.array([1, 2, 3])
        M = parcel_means(resid, parcels)
        np.testing.assert_allclose(M["1"], resid[:, 0])


class TestMsDispersion:
    def test_basic(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(100, 6)
        parcels = np.array([1, 1, 2, 2, 3, 3])
        disp = ms_dispersion(resid, parcels)
        assert len(disp) == 3
        for key in ["1", "2", "3"]:
            assert key in disp
            assert disp[key] >= 0

    def test_identical_columns_zero_dispersion(self):
        resid = np.tile(np.random.randn(100, 1), (1, 4))
        parcels = np.array([1, 1, 1, 1])
        disp = ms_dispersion(resid, parcels)
        assert disp["1"] == pytest.approx(0.0, abs=1e-10)


class TestMsWeights:
    def test_basic(self):
        sizes = np.array([100.0, 50.0, 10.0])
        disp = np.array([0.1, 0.2, 0.5])
        w = ms_weights(200, 2, sizes, disp, beta=0.5)
        assert len(w) == 3
        assert np.all(w > 0)

    def test_larger_size_more_weight(self):
        sizes = np.array([100.0, 10.0])
        disp = np.array([0.0, 0.0])
        w = ms_weights(100, 1, sizes, disp, beta=0.5)
        assert w[0] > w[1]


class TestMsParentMaps:
    def test_basic(self):
        fine = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        medium = np.array([10, 10, 10, 10, 20, 20, 20, 20])
        coarse = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        parents = ms_parent_maps(fine, medium, coarse)
        assert parents["parent_medium"][1] == 10
        assert parents["parent_medium"][3] == 20
        assert parents["parent_coarse"][1] == 100


class TestMsEstimateScale:
    def test_basic(self):
        rng = np.random.RandomState(42)
        M = {"1": rng.randn(100), "2": rng.randn(100)}

        def estimator(y):
            return {"phi": np.array([0.5]), "order": (1, 0)}

        result = ms_estimate_scale(M, estimator)
        assert "phi" in result
        assert "acvf" in result
        assert len(result["phi"]) == 2


class TestMsCombineToFine:
    def test_pacf_weighted(self):
        phi_c = {"100": np.array([0.5])}
        phi_m = {"10": np.array([0.4]), "20": np.array([0.6])}
        phi_f = {"1": np.array([0.3]), "2": np.array([0.5]),
                 "3": np.array([0.7]), "4": np.array([0.4])}
        parents = {
            "parent_medium": {1: 10, 2: 10, 3: 20, 4: 20},
            "parent_coarse": {1: 100, 2: 100, 3: 100, 4: 100},
        }
        sizes = {
            "n_t": 200, "n_runs": 1, "beta": 0.5,
            "coarse": {"100": 8},
            "medium": {"10": 4, "20": 4},
            "fine": {"1": 2, "2": 2, "3": 2, "4": 2},
        }
        disp = {
            "coarse": {"100": 0.1},
            "medium": {"10": 0.1, "20": 0.1},
            "fine": {"1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1},
        }
        result = ms_combine_to_fine(
            phi_c, phi_m, phi_f,
            parents=parents, sizes=sizes, disp_list=disp,
            p_target=1, mode="pacf_weighted",
        )
        assert len(result) == 4
        for key in ["1", "2", "3", "4"]:
            assert key in result
            assert len(result[key]) == 1


def _ar_cols(n, v, phi, rng):
    from scipy.signal import lfilter

    e = rng.standard_normal((n + 200, v))
    return lfilter([1.0], np.r_[1.0, -np.asarray(phi, dtype=float)], e, axis=0)[200:]


class TestMultiscaleFitNoise:
    """fmriAR 617bd87 (test-audit-regressions.R:296-321, :799-807), row AR-10."""

    def test_pooling_is_invariant_to_the_units_of_the_data(self):
        """Cheap pass disqualified: unit-free dispersion alone does not fix
        the white-noise order below, and a raw-variance MAD moves phi."""
        from fmrimod.ar.estimation import fit_noise

        rng = np.random.default_rng(9118)
        n, v = 200, 64
        R = _ar_cols(n, v, [0.4], rng) * rng.uniform(0.5, 2.0, v)[None, :]
        ps = {
            "coarse": np.repeat([1, 2], 32),
            "medium": np.repeat([1, 2, 3, 4], 16),
            "fine": np.repeat(np.arange(1, 9), 8),
        }
        for mode in ("pacf_weighted", "acvf_pooled"):
            a = fit_noise(R, pooling="parcel", parcels=ps["fine"], parcel_sets=ps,
                          p=1, multiscale=mode)
            b = fit_noise(R * 100, pooling="parcel", parcels=ps["fine"],
                          parcel_sets=ps, p=1, multiscale=mode)
            for k in a.phi_by_parcel:
                np.testing.assert_allclose(b.phi_by_parcel[k], a.phi_by_parcel[k],
                                           rtol=0, atol=1e-8, err_msg=mode)

    def test_acvf_pooled_auto_fits_the_selected_order(self):
        from fmrimod.ar.estimation import fit_noise

        ps = {
            "coarse": np.repeat([1, 2], 16),
            "medium": np.repeat([1, 2, 3, 4], 8),
            "fine": np.repeat(np.arange(1, 9), 4),
        }
        white = np.random.default_rng(9310).standard_normal((300, 32))
        pl = fit_noise(white, pooling="parcel", parcels=ps["fine"], parcel_sets=ps,
                       p="auto", multiscale="acvf_pooled")
        assert pl.order == (0, 0)
        assert all(len(v) == 0 for v in pl.phi_by_parcel.values())
        ar1 = _ar_cols(300, 32, [0.5], np.random.default_rng(9119))
        pl1 = fit_noise(ar1, pooling="parcel", parcels=ps["fine"], parcel_sets=ps,
                        p="auto", p_max=6, multiscale="acvf_pooled")
        assert pl1.order[0] < 6
        # Without parcel_sets the same rule applies.
        pl2 = fit_noise(white, pooling="parcel", parcels=ps["fine"], p="auto",
                        multiscale="acvf_pooled")
        assert pl2.order == (0, 0)

    def test_dispersion_is_relative_to_the_median_voxel_variance(self):
        rng = np.random.default_rng(3)
        resid = rng.standard_normal((100, 6)) * np.array([1, 2, 1, 3, 1, 1.5])
        parcels = np.array([1, 1, 2, 2, 3, 3])
        d1 = ms_dispersion(resid, parcels)
        d2 = ms_dispersion(resid * 10, parcels)
        for k in d1:
            assert d1[k] == pytest.approx(d2[k], rel=1e-12)
