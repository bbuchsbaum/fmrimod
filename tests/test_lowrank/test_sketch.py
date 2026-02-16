"""Tests for sketch matrices."""

import numpy as np
import pytest

from fmrimod.lowrank.sketch import (
    SketchKind,
    make_sketch,
    sketch_data,
    _next_power_of_two,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestMakeSketch:
    def test_gaussian_shape(self, rng):
        S = make_sketch(100, 30, kind="gaussian", rng=rng)
        assert S.shape == (30, 100)

    def test_srht_shape(self, rng):
        S = make_sketch(100, 30, kind="srht", rng=rng)
        assert S.shape == (30, 100)

    def test_countsketch_shape(self, rng):
        S = make_sketch(100, 30, kind="countsketch", rng=rng)
        assert S.shape == (30, 100)

    def test_countsketch_sparsity(self, rng):
        """Each column should have exactly one nonzero entry."""
        S = make_sketch(50, 20, kind="countsketch", rng=rng)
        nnz_per_col = np.sum(S != 0, axis=0)
        assert np.all(nnz_per_col == 1)

    def test_enum_variant(self, rng):
        S = make_sketch(50, 10, kind=SketchKind.GAUSSIAN, rng=rng)
        assert S.shape == (10, 50)

    def test_invalid_kind_raises(self, rng):
        with pytest.raises(ValueError):
            make_sketch(10, 5, kind="bogus", rng=rng)


class TestSketchPreservation:
    """Sketching should approximately preserve matrix products."""

    def test_gaussian_preserves_norms(self, rng):
        n, d = 200, 5
        X = rng.standard_normal((n, d))
        S = make_sketch(n, 80, kind="gaussian", rng=rng)
        X_sk = S @ X

        # Frobenius norm should be roughly preserved
        ratio = np.linalg.norm(X_sk, "fro") / np.linalg.norm(X, "fro")
        assert 0.5 < ratio < 2.0

    def test_srht_preserves_norms(self, rng):
        n, d = 128, 5  # power of 2 for cleaner SRHT
        X = rng.standard_normal((n, d))
        S = make_sketch(n, 60, kind="srht", rng=rng)
        X_sk = S @ X

        ratio = np.linalg.norm(X_sk, "fro") / np.linalg.norm(X, "fro")
        assert 0.3 < ratio < 3.0


class TestSketchData:
    def test_multiple_arrays(self, rng):
        X = rng.standard_normal((100, 5))
        Y = rng.standard_normal((100, 20))
        S = make_sketch(100, 30, rng=rng)
        X_sk, Y_sk = sketch_data(S, X, Y)
        assert X_sk.shape == (30, 5)
        assert Y_sk.shape == (30, 20)


class TestNextPowerOfTwo:
    def test_exact_powers(self):
        assert _next_power_of_two(1) == 1
        assert _next_power_of_two(2) == 2
        assert _next_power_of_two(4) == 4

    def test_non_powers(self):
        assert _next_power_of_two(3) == 4
        assert _next_power_of_two(5) == 8
        assert _next_power_of_two(100) == 128
