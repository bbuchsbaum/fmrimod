"""Compatibility coverage for fmrihrf-exported helper names."""

import numpy as np

import fmrimod
from fmrimod.hrf.generators import gamma_generator


def test_sampling_accessors_match_fmrihrf_exports():
    sframe = fmrimod.SamplingFrame(blocklens=[3, 2], tr=2.0)

    np.testing.assert_allclose(
        fmrimod.acquisition_onsets(sframe),
        fmrimod.samples(sframe, global_time=True),
    )
    np.testing.assert_allclose(
        fmrimod.samples(sframe, blockids=[1], global_time=False),
        [1.0, 3.0],
    )


def test_regressor_amplitudes_and_shift_helpers():
    reg = fmrimod.regressor(
        onsets=[1.0, 3.0, 5.0],
        hrf="spmg1",
        amplitude=[1.0, 2.0, 3.0],
    )

    np.testing.assert_allclose(fmrimod.amplitudes(reg), [1.0, 2.0, 3.0])

    shifted = fmrimod.shift(reg, offset=2.5)
    np.testing.assert_allclose(shifted.onsets, [3.5, 5.5, 7.5])
    np.testing.assert_allclose(fmrimod.amplitudes(shifted), fmrimod.amplitudes(reg))


def test_hrf_generator_aliases_accept_fmrihrf_argument_names():
    times = np.linspace(0, 24, 9)

    generators = [
        fmrimod.hrf_bspline_generator(nbasis=4),
        fmrimod.hrf_tent_generator(nbasis=4),
        fmrimod.hrf_fourier_generator(nbasis=4),
        fmrimod.hrf_daguerre_generator(nbasis=4),
        fmrimod.hrf_fir_generator(nbasis=4),
    ]

    for hrf in generators:
        assert hrf.nbasis == 4
        assert fmrimod.evaluate(hrf, times).shape == (len(times), 4)


def test_hrf_decorator_aliases_handle_scalar_inputs():
    times = np.linspace(0, 24, 25)

    lagged = fmrimod.hrf_lagged(fmrimod.SPM_CANONICAL, lag=2.0)
    np.testing.assert_allclose(
        np.asarray(fmrimod.evaluate(lagged, times)).squeeze(),
        fmrimod.SPM_CANONICAL(times - 2.0),
    )

    blocked = fmrimod.hrf_blocked(fmrimod.SPM_CANONICAL, width=3.0)
    assert blocked.nbasis == 1
    assert np.max(fmrimod.evaluate(blocked, times)) > 0


def test_empirical_set_and_library_aliases():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 1.0, 0.0])
    empirical = fmrimod.gen_empirical_hrf(times, values)

    np.testing.assert_allclose(fmrimod.evaluate(empirical, [0.5, 1.5]), [0.5, 0.5])

    hrf_set = fmrimod.hrf_set(fmrimod.SPM_CANONICAL, empirical)
    assert hrf_set.name == "hrf_set"
    assert hrf_set.nbasis == 2

    library = fmrimod.gen_hrf_library(
        gamma_generator,
        {"shape": [4.0, 6.0], "rate": [1.0]},
    )
    assert library.nbasis == 2
