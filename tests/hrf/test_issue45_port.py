"""Regression tests for the fmrihrf#45 defects ported into fmrimod.

Covers fmrimod issues #7-#14 and #17. Each test is written so that it fails
against the pre-fix implementation, not merely passes against the current one:

* the quadrature tests assert *convergence* and agreement with an independent
  ``scipy.integrate`` ground truth, both of which the old ``1/precision``
  scaling violated by construction;
* the ``summate`` tests assert an exact ratio (``duration``) that neither an
  unweighted sum nor a pointwise maximum can produce;
* the branch test asserts two precisions agree, where the old code jumped 5x;
* the basis tests assert the value at a fixed time is independent of the grid
  the caller passed.

The R reference for all of this is fmrihrf >= 0.4.0 (``tests/testthat/
test_issue45.R`` there).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from fmrimod.hrf import functions as hrf_functions
from fmrimod.hrf.decorators import block_hrf
from fmrimod.hrf.registry import get_hrf
from fmrimod.regressor.convolution import convolve_hrf_per_event
from fmrimod.regressor.core import regressor

GRID = np.arange(0, 82, 2.0)


def _hrf():
    return get_hrf("spmg1")


def _integral_truth(grid, onsets, duration, hrf, amplitude=1.0):
    """amplitude * sum_events integral_0^duration h(t - onset - u) du."""

    def h_at(x):
        return float(np.asarray(hrf(np.array([x]))).ravel()[0])

    out = np.zeros(len(grid), dtype=np.float64)
    for i, t in enumerate(grid):
        total = 0.0
        for onset in onsets:
            total += (
                amplitude
                * quad(
                    lambda u, t=t, onset=onset: h_at(t - onset - u),
                    0.0,
                    duration,
                    limit=200,
                )[0]
            )
        out[i] = total
    return out


# --------------------------------------------------------------------------
# #7 - epoch amplitude no longer scales with 1/precision
# --------------------------------------------------------------------------


def test_epoch_amplitude_converges_instead_of_scaling_with_inverse_precision():
    """Peak must converge as precision shrinks, not grow like 1/precision.

    Pre-fix peaks were 1.305 / 1.970 / 6.048 / 11.97 at precision
    0.5 / 0.33 / 0.1 / 0.05 -- a ~2x jump for every halving of the step.
    """
    hrf = _hrf()
    peaks = {}
    for p in (0.5, 0.33, 0.1, 0.05, 0.01):
        reg = regressor(onsets=[10.0, 30.0], hrf=hrf, duration=4.0)
        peaks[p] = float(np.abs(np.asarray(reg.evaluate(GRID, precision=p))).max())

    limit = peaks[0.01]
    for p, peak in peaks.items():
        assert peak == pytest.approx(limit, rel=0.01), (p, peaks)

    # Cheap-pass disqualifier: under the old code this ratio was ~50.
    assert peaks[0.01] / peaks[0.5] == pytest.approx(1.0, abs=0.01), peaks


def test_epoch_matches_independent_numerical_integration():
    """The block response is the integral over the block, to ~1e-3."""
    hrf = _hrf()
    onsets = [10.3, 30.7]  # deliberately off-grid
    truth = _integral_truth(GRID, onsets, 4.0, hrf)

    reg = regressor(onsets=onsets, hrf=hrf, duration=4.0)
    got = np.asarray(reg.evaluate(GRID, precision=0.1))

    rel = np.abs(got - truth).max() / np.abs(truth).max()
    assert rel < 1e-3, rel


def test_epoch_error_shrinks_as_precision_shrinks():
    """Refining the quadrature must reduce the error against ground truth."""
    hrf = _hrf()
    onsets = [10.3, 30.7]
    truth = _integral_truth(GRID, onsets, 4.0, hrf)

    errs = []
    for p in (0.33, 0.1, 0.05):
        reg = regressor(onsets=onsets, hrf=hrf, duration=4.0)
        got = np.asarray(reg.evaluate(GRID, precision=p))
        errs.append(np.abs(got - truth).max() / np.abs(truth).max())

    assert errs[1] < errs[0], errs
    assert errs[2] <= errs[1] * 1.05, errs


def test_point_event_onset_is_not_snapped_to_the_fine_grid():
    """An onset falling between bins keeps its sub-bin position."""
    hrf = _hrf()
    onsets = [10.15, 30.15]  # 10.15 is mid-bin at precision 0.33

    def truth_at(t):
        return sum(float(np.asarray(hrf(np.array([t - o]))).ravel()[0]) for o in onsets)

    truth = np.array([truth_at(t) for t in GRID])
    reg = regressor(onsets=onsets, hrf=hrf, duration=0.0)
    got = np.asarray(reg.evaluate(GRID, precision=0.33))

    rel = np.abs(got - truth).max() / np.abs(truth).max()
    # Snapping to the nearest bin cost ~12% of peak at this precision.
    assert rel < 0.02, rel


def test_point_event_amplitude_is_precision_invariant():
    """Point events were already precision-invariant; keep them that way."""
    hrf = _hrf()
    peaks = []
    for p in (0.5, 0.33, 0.1):
        reg = regressor(onsets=[10.0, 30.0], hrf=hrf, duration=0.0)
        peaks.append(float(np.abs(np.asarray(reg.evaluate(GRID, precision=p))).max()))
    assert peaks[0] == pytest.approx(peaks[-1], rel=0.02), peaks


def test_regressor_path_agrees_with_hrf_evaluate_path():
    """The two block implementations must land on the same scale.

    This is the defect that made the bug survivable: ``HRF.evaluate`` used
    trapezoid quadrature while the regressor path summed unweighted samples,
    so the two disagreed by a factor of ``1/precision`` (100x at 0.01).
    """
    hrf = _hrf()
    grid = np.arange(0, 60, 1.0)
    onset, duration = 10.0, 5.0

    for summate in (True, False):
        reg = regressor(onsets=[onset], hrf=hrf, duration=duration, summate=summate)
        via_regressor = np.asarray(reg.evaluate(grid, precision=0.01))
        via_evaluate = np.asarray(
            hrf.evaluate(
                grid - onset, duration=duration, precision=0.01, summate=summate
            )
        )
        rel = np.abs(via_regressor - via_evaluate).max() / np.abs(via_evaluate).max()
        assert rel < 1e-3, (summate, rel)


# --------------------------------------------------------------------------
# #8 - summate is honoured through the regressor, not silently dropped
# --------------------------------------------------------------------------


def test_summate_false_is_duration_averaged_through_the_regressor():
    """summate=False divides by the block duration. Exactly.

    Pre-fix, ``regressor(..., summate=False)`` was a no-op: both settings
    returned 6.0478. The ratio below is the sharpest disqualifier available,
    because an unweighted sum gives duration/precision and a pointwise
    maximum gives neither.
    """
    hrf = _hrf()
    duration = 4.0
    peaks = {}
    for summate in (True, False):
        reg = regressor(
            onsets=[10.0, 30.0], hrf=hrf, duration=duration, summate=summate
        )
        peaks[summate] = float(
            np.abs(np.asarray(reg.evaluate(GRID, precision=0.1))).max()
        )

    assert peaks[True] != pytest.approx(peaks[False]), peaks
    assert peaks[True] / peaks[False] == pytest.approx(duration, rel=1e-6), peaks


@pytest.mark.parametrize("duration", [2.0, 5.0])
def test_hrf_evaluate_summate_ratio_is_exactly_duration(duration):
    hrf = _hrf()
    grid = np.arange(0, 40, 0.5)
    acc = np.abs(hrf.evaluate(grid - 10, duration=duration, precision=0.2)).max()
    avg = np.abs(
        hrf.evaluate(grid - 10, duration=duration, precision=0.2, summate=False)
    ).max()
    assert acc / avg == pytest.approx(duration, rel=1e-9)


# --------------------------------------------------------------------------
# #9 - the impulse/block branch is keyed to duration alone
# --------------------------------------------------------------------------


def test_evaluate_branch_does_not_depend_on_precision():
    """precision is an accuracy knob; it must not pick the model.

    Pre-fix: peak 0.17544 at precision 0.2 (treated as an impulse) versus
    0.03332 at precision 0.05 (treated as a block) -- a 5.3x jump.
    """
    hrf = _hrf()
    grid = np.arange(0, 40, 0.1)
    peaks = [
        float(np.abs(hrf.evaluate(grid - 10, duration=0.19, precision=p)).max())
        for p in (0.2, 0.1, 0.05)
    ]
    assert peaks[0] == pytest.approx(peaks[-1], rel=0.01), peaks


def test_sub_precision_duration_is_integrated_as_a_block():
    """A duration below precision carries mass = duration, not unit mass."""
    hrf = _hrf()
    grid = np.arange(0, 40, 0.1)
    duration = 0.01
    block = float(
        np.abs(hrf.evaluate(grid - 10, duration=duration, precision=0.1)).max()
    )
    impulse = float(np.abs(hrf.evaluate(grid - 10, duration=0)).max())
    assert block / (duration * impulse) == pytest.approx(1.0, rel=1e-3)


def test_zero_duration_is_still_the_impulse_response():
    hrf = _hrf()
    grid = np.arange(0, 40, 0.5)
    np.testing.assert_allclose(
        hrf.evaluate(grid, duration=0), np.asarray(hrf(grid)), rtol=0, atol=0
    )


# --------------------------------------------------------------------------
# #10 - per-event path does not truncate blocked events at span
# --------------------------------------------------------------------------


def test_per_event_blocked_error_shrinks_with_precision():
    """Truncating at `span` left an error floor that precision could not fix."""
    hrf = _hrf()
    onsets = np.array([10.0, 30.0])
    durs = np.array([4.0, 4.0])
    amps = np.array([1.0, 1.0])
    truth = _integral_truth(GRID, onsets, 4.0, hrf)

    errs = []
    for p in (0.33, 0.1, 0.02):
        got = convolve_hrf_per_event(
            GRID, onsets, durs, amps, [hrf, hrf], span=hrf.span, precision=p
        )
        errs.append(np.abs(got - truth).max() / np.abs(truth).max())

    assert errs[-1] < errs[0] / 5, errs
    # Pre-fix this stalled at ~2.7e-3 regardless of precision.
    assert errs[-1] < 5e-4, errs


def test_per_event_late_basis_columns_are_not_truncated():
    """A basis whose late column peaks at the right edge of span must survive."""
    hrf = get_hrf("bspline")
    onsets = np.array([10.0, 30.0])
    got = convolve_hrf_per_event(
        GRID,
        onsets,
        np.array([4.0, 4.0]),
        np.array([1.0, 1.0]),
        [hrf, hrf],
        span=hrf.span,
        precision=0.1,
    )
    peaks = np.abs(got).max(axis=0)
    assert np.all(peaks > 0), peaks
    # The last column must not be crushed relative to its neighbours.
    assert peaks[-1] > 0.5 * peaks.max(), peaks


# --------------------------------------------------------------------------
# #11 - scalar t
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,ncol",
    [(hrf_functions.hrf_sine, 5), (hrf_functions.fourier_hrf, 5), (hrf_functions.bspline_hrf, 5), (hrf_functions.daguerre_basis, 3)],
)
def test_basis_functions_accept_scalar_t(fn, ncol):
    """Decorators that shift time can hand a scalar to the shape function."""
    scalar = np.asarray(fn(-1.0))
    vector = np.asarray(fn(np.array([-1.0])))
    assert scalar.shape == (1, ncol)
    np.testing.assert_allclose(scalar, vector)


# --------------------------------------------------------------------------
# #12 - daguerre normalizes against a fixed reference grid
# --------------------------------------------------------------------------


def test_daguerre_value_is_independent_of_the_query_grid():
    """t = 0 must give the same value however the caller sampled the axis.

    Pre-fix a grid reaching negative lag inflated the divisor: 1.0 alone
    versus 0.7788 inside seq(-2, 32, 0.5).
    """
    alone = float(hrf_functions.daguerre_basis(np.array([0.0]), n_basis=3, scale=4).ravel()[0])

    wide = np.arange(-2, 32.5, 0.5)
    in_wide = float(
        hrf_functions.daguerre_basis(wide, n_basis=3, scale=4)[np.argmin(np.abs(wide)), 0]
    )

    fine = np.arange(0, 24.1, 0.1)
    in_fine = float(hrf_functions.daguerre_basis(fine, n_basis=3, scale=4)[0, 0])

    assert alone == pytest.approx(in_wide, rel=1e-12)
    assert alone == pytest.approx(in_fine, rel=1e-12)


def test_daguerre_is_causal():
    vals = hrf_functions.daguerre_basis(np.array([-5.0, -1.0, -0.1]), n_basis=3, scale=4)
    np.testing.assert_array_equal(vals, np.zeros_like(vals))


# --------------------------------------------------------------------------
# #13 - B-spline knots come from span, not from the caller's t
# --------------------------------------------------------------------------


def test_bspline_basis_is_independent_of_the_evaluation_grid():
    """The same experiment on two grids must get the same basis."""
    hrf = get_hrf("bspline")

    def at8(grid):
        return np.asarray(hrf.evaluate(grid))[np.argmin(np.abs(grid - 8.0))]

    coarse = at8(np.arange(0, 24.5, 0.5))
    ragged = at8(np.unique(np.append(np.arange(0, 23.76, 0.33), 8.0)))
    np.testing.assert_allclose(coarse, ragged, rtol=1e-10, atol=1e-12)


def test_bspline_does_not_collapse_on_a_single_point_grid():
    """Pre-fix, the knots collapsed onto the one supplied point."""
    hrf = get_hrf("bspline")
    single = np.asarray(hrf.evaluate(np.array([0.33]))).ravel()
    full = np.asarray(hrf.evaluate(np.arange(0, 24.5, 0.33)))[1]
    np.testing.assert_allclose(single, full, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# #14 - HRFs are causal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [hrf_functions.gaussian_hrf, hrf_functions.inv_logit_hrf, hrf_functions.mexhat_hrf, hrf_functions.lwu_hrf, hrf_functions.gamma_hrf],
)
def test_shape_functions_carry_no_pre_onset_response(fn):
    t = np.array([-10.0, -5.0, -1.0, -0.1])
    vals = np.asarray(fn(t))
    np.testing.assert_array_equal(vals, np.zeros_like(vals))


def test_causality_survives_the_block_decorator():
    """block_hrf samples the shape function at shifted times, so a decorated
    HRF would otherwise mix in pre-onset values."""
    blocked = block_hrf(get_hrf("gaussian"), width=4.0, precision=0.1)
    vals = np.asarray(blocked(np.array([-12.0, -8.0]))).ravel()
    np.testing.assert_array_equal(vals, np.zeros_like(vals))


def test_causality_survives_the_lag_decorator():
    lagged = get_hrf("gaussian").lag(3.0)
    vals = np.asarray(lagged(np.array([-10.0, -5.0]))).ravel()
    np.testing.assert_array_equal(vals, np.zeros_like(vals))


# --------------------------------------------------------------------------
# #17 - one convolution engine
# --------------------------------------------------------------------------


def test_convolution_aliases_agree_and_warn():
    hrf = _hrf()
    reg = regressor(onsets=[10.0, 30.0], hrf=hrf, duration=4.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        canonical = np.asarray(reg.evaluate(GRID, precision=0.1, method="conv"))

    for alias in ("direct", "fft"):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            got = np.asarray(reg.evaluate(GRID, precision=0.1, method=alias))
        np.testing.assert_array_equal(got, canonical)


def test_unknown_convolution_method_is_rejected():
    hrf = _hrf()
    reg = regressor(onsets=[10.0], hrf=hrf, duration=0.0)
    with pytest.raises(ValueError, match="must be one of"):
        reg.evaluate(GRID, method="wavelet")


def test_removed_fft_helpers_are_gone():
    """_convolve_fft / _next_power_of_2 were redundant with scipy's auto path."""
    from fmrimod.regressor import convolution

    assert not hasattr(convolution, "_convolve_fft")
    assert not hasattr(convolution, "_next_power_of_2")

    from fmrimod.regressor import neural_input

    assert not hasattr(neural_input, "neural_input_fast")
