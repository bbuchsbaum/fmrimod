"""Tests for new HRF functions: boxcar, weighted, tent."""

import numpy as np
import pytest

from fmrimod.hrf.functions import boxcar_hrf, weighted_hrf
from fmrimod.hrf.generators import (
    boxcar_generator,
    tent_generator,
    weighted_generator,
)
from fmrimod.hrf.registry import get_hrf

# ── boxcar_hrf ──────────────────────────────────────────────────────

class TestBoxcarHRF:
    def test_basic_shape(self):
        t = np.linspace(-1, 10, 200)
        y = boxcar_hrf(t, width=5.0)
        # Zero before 0
        assert np.all(y[t < 0] == 0)
        # Amplitude=1 inside window
        assert np.all(y[(t >= 0) & (t < 5)] == 1.0)
        # Zero at and after width
        assert np.all(y[t >= 5] == 0)

    def test_custom_amplitude(self):
        t = np.array([0, 1, 2, 5])
        y = boxcar_hrf(t, width=3.0, amplitude=2.5)
        np.testing.assert_array_equal(y, [2.5, 2.5, 2.5, 0.0])

    def test_normalize(self):
        """``normalize=True`` was retired (bd-01KRGCZ6QJME1JD8FD5D4PGC04).

        The replacement is to pass ``amplitude=1/width`` explicitly, which is
        what the retired flag did internally. A regression test makes sure the
        retired form fails loud rather than silently no-oping.
        """
        width = 4.0
        t = np.array([0, 1, 2, 3, 4])
        y = boxcar_hrf(t, width=width, amplitude=1.0 / width)
        assert y[0] == pytest.approx(1.0 / width)
        assert y[-1] == 0.0  # t == width is outside

        with pytest.raises(ValueError, match="normalize=True.*retired"):
            boxcar_hrf(t, width=width, normalize=True)

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="positive"):
            boxcar_hrf(np.array([0, 1]), width=0)
        with pytest.raises(ValueError, match="positive"):
            boxcar_hrf(np.array([0, 1]), width=-1)

    def test_generator(self):
        # ``normalize=True`` retired; the test now pins the explicit-amplitude
        # path that callers must use after bd-01KRGCZ6QJME1JD8FD5D4PGC04.
        with pytest.raises(ValueError, match="normalize=True.*retired"):
            boxcar_generator(width=3.0, normalize=True)

        hrf = boxcar_generator(width=3.0, amplitude=1.0 / 3.0)
        assert hrf.name == "boxcar[3]"
        assert hrf.span == 3.0
        assert hrf.nbasis == 1
        t = np.array([0, 1, 2, 3])
        y = hrf(t)
        assert y[0] == pytest.approx(1.0 / 3.0)

    def test_registry(self):
        hrf = get_hrf("boxcar", width=5.0)
        assert hrf.span == 5.0


# ── weighted_hrf ────────────────────────────────────────────────────

class TestWeightedHRF:
    def test_width_specification(self):
        t = np.linspace(0, 6, 100)
        y = weighted_hrf(t, weights=[0.2, 0.5, 0.8, 0.3], width=6.0)
        assert y.shape == (100,)
        assert np.max(y) == pytest.approx(0.8, abs=0.05)

    def test_times_specification(self):
        t = np.linspace(0, 6, 100)
        y = weighted_hrf(
            t,
            weights=[0.1, 0.5, 0.8, 0.5, 0.1],
            times=[0, 1, 3, 5, 6],
            method="linear",
        )
        assert y.shape == (100,)
        assert np.max(y) == pytest.approx(0.8, abs=0.02)

    def test_constant_method(self):
        """Step function: value at t=0 should be first weight."""
        t = np.array([0.0, 0.5, 1.0])
        y = weighted_hrf(t, weights=[2.0, 3.0], width=2.0, method="constant")
        assert y[0] == pytest.approx(2.0)

    def test_linear_method(self):
        """Linear interpolation: midpoint between two weights."""
        t = np.array([0.0, 1.0, 2.0])
        y = weighted_hrf(t, weights=[0.0, 1.0], width=2.0, method="linear")
        assert y[0] == pytest.approx(0.0)
        assert y[1] == pytest.approx(0.5)
        assert y[2] == pytest.approx(1.0)

    def test_normalize_constant(self):
        """``normalize=True`` retired; callers scale weights explicitly."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        # The retired flag divided weights so sum(weights[:-1]) == 1; do that
        # by hand.
        weights = np.array([1.0, 1.0, 1.0, 1.0]) / 3.0
        y = weighted_hrf(t, weights=weights, width=3.0, method="constant")
        assert y[0] == pytest.approx(1.0 / 3.0, abs=0.01)

        with pytest.raises(ValueError, match="normalize=True.*retired"):
            weighted_hrf(t, weights=[1.0, 1.0, 1.0, 1.0], width=3.0,
                         method="constant", normalize=True)

    def test_normalize_linear(self):
        t = np.array([0.0, 5.0])
        with pytest.raises(ValueError, match="normalize=True.*retired"):
            weighted_hrf(
                t,
                weights=[1.0, 1.0],
                width=5.0,
                method="linear",
                normalize=True,
            )

        # Retired flag divided by trapezoidal integral; for constant-1 over
        # [0, 5] that's 5, hence weights /= 5.
        weights = np.array([1.0, 1.0]) / 5.0
        y = weighted_hrf(t, weights=weights, width=5.0, method="linear")
        assert y[0] == pytest.approx(1.0 / 5.0)

    def test_zero_outside_domain(self):
        t = np.array([-1.0, 10.0])
        y = weighted_hrf(t, weights=[1, 1], width=5.0)
        np.testing.assert_array_equal(y, [0.0, 0.0])

    def test_too_few_weights(self):
        with pytest.raises(ValueError, match="at least 2"):
            weighted_hrf(np.array([0]), weights=[1.0])

    def test_mismatched_times_weights(self):
        with pytest.raises(ValueError, match="same length"):
            weighted_hrf(np.array([0]), weights=[1, 2], times=[0, 1, 2])

    def test_no_width_or_times(self):
        with pytest.raises(ValueError, match="Either"):
            weighted_hrf(np.array([0]), weights=[1, 2])

    def test_generator(self):
        hrf = weighted_generator(weights=[1, 2, 1], width=4.0)
        assert hrf.nbasis == 1
        assert hrf.span == 4.0

    def test_registry(self):
        hrf = get_hrf("weighted", weights=[1, 2, 1], width=4.0)
        assert hrf.span == 4.0


# ── tent_generator ──────────────────────────────────────────────────

class TestTentGenerator:
    def test_basic(self):
        hrf = tent_generator(nbasis=5, span=20.0)
        assert hrf.nbasis == 5
        assert hrf.span == 20.0
        assert "tent" in hrf.name.lower()

    def test_output_shape(self):
        hrf = tent_generator(nbasis=6, span=10.0)
        t = np.linspace(0, 10, 50)
        y = hrf(t)
        assert y.shape == (50, 6)

    def test_degree1_bspline_equivalence(self):
        """Tent basis should be equivalent to degree-1 B-splines."""
        from fmrimod.hrf.functions import bspline_hrf

        t = np.linspace(0, 20, 100)
        tent = tent_generator(nbasis=5, span=20.0)
        tent_vals = tent(t)
        bspline_vals = bspline_hrf(t, n_basis=5, degree=1, span=20.0)
        np.testing.assert_allclose(tent_vals, bspline_vals, atol=1e-10)

    def test_registry(self):
        hrf = get_hrf("tent", nbasis=4, span=12.0)
        assert hrf.nbasis == 4
        assert hrf.span == 12.0

    def test_partition_of_unity(self):
        """Tent functions should approximately sum to 1 in the interior.

        With R-compatible bs(intercept=FALSE), the first basis function is
        dropped so sums at the left boundary are < 1.  Check the true
        interior where all retained bases overlap.
        """
        hrf = tent_generator(nbasis=8, span=20.0)
        # Stay well inside the span so the dropped intercept basis has no effect
        t = np.linspace(3, 17, 80)
        y = hrf(t)
        row_sums = y.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.05)
