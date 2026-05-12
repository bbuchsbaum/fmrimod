"""Focused ports of fmrihrf testthat cases not covered elsewhere."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.hrf.derivatives import deriv, spmg1_derivative, spmg1_second_derivative
from fmrimod.hrf.functions import hrf_basis_lwu, hrf_lwu
from fmrimod.hrf.generators import fourier_generator
from fmrimod.hrf.library import (
    BSPLINE_HRF,
    GAUSSIAN_HRF,
    SPM_CANONICAL,
    SPM_WITH_DERIVATIVE,
    SPM_WITH_DISPERSION,
)
from fmrimod.hrf.penalty import penalty_matrix
from fmrimod.sampling import SamplingFrame


def test_acquisition_onsets_matches_r_edge_cases():
    sf = SamplingFrame(blocklens=[100, 120, 80], tr=2)
    onsets = sf.acquisition_onsets

    assert len(onsets) == 300
    assert onsets[99] == pytest.approx(199)
    assert onsets[100] == pytest.approx(201)
    assert onsets[219] == pytest.approx(439)
    assert onsets[220] == pytest.approx(441)

    short_tr = SamplingFrame(blocklens=5, tr=0.5)
    np.testing.assert_allclose(
        short_tr.acquisition_onsets,
        [0.25, 0.75, 1.25, 1.75, 2.25],
    )

    standard = SamplingFrame(blocklens=[150, 150], tr=2, start_time=0)
    standard_onsets = standard.acquisition_onsets
    assert standard_onsets[0] == pytest.approx(0)
    assert standard_onsets[149] == pytest.approx(298)
    assert standard_onsets[150] == pytest.approx(300)
    assert standard_onsets[299] == pytest.approx(598)
    assert np.max(standard_onsets) == pytest.approx((300 - 1) * 2)


def test_deriv_matches_spmg_analytic_contracts():
    t = np.arange(0, 20.5, 0.5)

    d1 = deriv(SPM_CANONICAL, t)
    assert d1.shape == t.shape
    assert d1[0] == pytest.approx(0)
    np.testing.assert_allclose(d1, spmg1_derivative(t))

    d2 = deriv(SPM_WITH_DERIVATIVE, t)
    assert d2.shape == (len(t), 2)
    np.testing.assert_allclose(d2[:, 0], spmg1_derivative(t))
    np.testing.assert_allclose(d2[:, 1], spmg1_second_derivative(t))

    d3 = deriv(SPM_WITH_DISPERSION, t)
    assert d3.shape == (len(t), 3)
    np.testing.assert_allclose(d3[:, 0], spmg1_derivative(t))
    np.testing.assert_allclose(d3[:, 1], spmg1_second_derivative(t))
    assert np.issubdtype(d3[:, 2].dtype, np.floating)


def test_deriv_handles_numeric_and_edge_cases():
    t = np.arange(0, 10.5, 0.5)
    assert deriv(GAUSSIAN_HRF, t).shape == t.shape

    bspline_deriv = deriv(BSPLINE_HRF, np.arange(0, 21, 1))
    assert bspline_deriv.shape == (21, BSPLINE_HRF.nbasis)

    assert deriv(SPM_CANONICAL, [5]).shape == (1,)
    assert deriv(SPM_CANONICAL, []).shape == (0,)

    t_neg = np.array([-5, -2, 0, 2, 5])
    d_neg = deriv(SPM_CANONICAL, t_neg)
    np.testing.assert_allclose(d_neg[:2], [0, 0])


def test_lwu_response_normalisation_and_derivative_basis():
    t = np.arange(0, 20.5, 0.5)
    raw = hrf_lwu(t, tau=6, sigma=2, rho=0.4)
    assert raw.shape == t.shape

    height = hrf_lwu(t, tau=6, sigma=2, rho=0.4, normalize="height")
    assert np.max(np.abs(height)) == pytest.approx(1)

    with pytest.warns(UserWarning, match="area"):
        area = hrf_lwu(t, tau=6, sigma=2, rho=0.4, normalize="area")
    np.testing.assert_allclose(area, raw)

    theta = np.array([6.0, 2.0, 0.4])
    t_basis = np.arange(0, 21, 1)
    basis = hrf_basis_lwu(theta, t_basis)
    assert basis.shape == (len(t_basis), 4)
    np.testing.assert_allclose(
        basis[:, 0],
        hrf_lwu(t_basis, tau=theta[0], sigma=theta[1], rho=theta[2]),
    )

    delta = 1e-4
    t0 = 4.0
    fd_tau = (
        hrf_lwu(t0, tau=theta[0] + delta, sigma=theta[1], rho=theta[2])
        - hrf_lwu(t0, tau=theta[0] - delta, sigma=theta[1], rho=theta[2])
    ) / (2 * delta)
    idx = np.where(t_basis == t0)[0][0]
    assert basis[idx, 1] == pytest.approx(float(fd_tau), abs=1e-3)


def test_penalty_matrix_r_contracts_for_fourier_and_default_hrfs():
    fourier = fourier_generator(n_basis=4, span=24)
    np.testing.assert_allclose(np.diag(penalty_matrix(fourier)), [1, 1, 4, 4])

    np.testing.assert_allclose(
        penalty_matrix(GAUSSIAN_HRF),
        np.eye(GAUSSIAN_HRF.nbasis),
    )
