"""Contract and metamorphic tests for robust weight estimators."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.robust.estimators import bisquare_weights, huber_weights, mad_scale


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


class TestMadScale:
    def test_translation_invariant(self, rng: np.random.Generator) -> None:
        residuals = rng.standard_normal((120, 7))
        shifted = residuals + 13.7
        np.testing.assert_allclose(
            mad_scale(residuals, axis=0),
            mad_scale(shifted, axis=0),
            atol=1e-12,
        )

    def test_zero_residuals_have_zero_scale(self) -> None:
        residuals = np.zeros((40, 5), dtype=np.float64)
        scale = mad_scale(residuals, axis=0)
        np.testing.assert_allclose(scale, 0.0, atol=0.0)


class TestWeightFunctions:
    @pytest.mark.parametrize(
        ("weight_fn", "kwargs"),
        [
            (huber_weights, {"k": 1.345}),
            (bisquare_weights, {"c": 4.685}),
        ],
    )
    def test_scale_invariant(
        self,
        rng: np.random.Generator,
        weight_fn,
        kwargs: dict[str, float],
    ) -> None:
        residuals = rng.standard_normal((150, 6))
        scale = mad_scale(residuals, axis=0)
        alpha = 9.0

        w_ref = weight_fn(residuals, scale, **kwargs)
        w_scaled = weight_fn(alpha * residuals, alpha * scale, **kwargs)
        np.testing.assert_allclose(w_ref, w_scaled, atol=1e-12)

    @pytest.mark.parametrize(
        ("weight_fn", "kwargs"),
        [
            (huber_weights, {"k": 1.345}),
            (bisquare_weights, {"c": 4.685}),
        ],
    )
    def test_sign_symmetric(
        self,
        rng: np.random.Generator,
        weight_fn,
        kwargs: dict[str, float],
    ) -> None:
        residuals = rng.standard_normal((90, 4))
        scale = mad_scale(residuals, axis=0)
        np.testing.assert_allclose(
            weight_fn(residuals, scale, **kwargs),
            weight_fn(-residuals, scale, **kwargs),
            atol=1e-12,
        )

    def test_huber_weights_monotone_with_absolute_residual(self) -> None:
        residuals = np.array([[0.0], [0.5], [1.0], [2.0], [5.0]])
        scale = np.array([1.0])
        w = huber_weights(residuals, scale, k=1.345).ravel()
        assert np.all(np.diff(w) <= 1e-12)
        assert 0.0 < w[-1] < 1.0

    def test_bisquare_weights_monotone_and_zero_beyond_cutoff(self) -> None:
        residuals = np.array([[0.0], [1.0], [2.0], [3.0], [6.0]])
        scale = np.array([1.0])
        w = bisquare_weights(residuals, scale, c=4.685).ravel()
        assert np.all(np.diff(w[:-1]) <= 1e-12)
        assert w[-1] == 0.0

    @pytest.mark.parametrize(
        ("weight_fn", "kwargs"),
        [
            (huber_weights, {"k": 1.345}),
            (bisquare_weights, {"c": 4.685}),
        ],
    )
    def test_zero_scale_guard_keeps_weights_finite(
        self,
        weight_fn,
        kwargs: dict[str, float],
    ) -> None:
        residuals = np.array([[0.0, 1e8], [0.0, -1e8], [1.0, 0.0]], dtype=np.float64)
        scale = np.array([0.0, 0.0], dtype=np.float64)
        w = weight_fn(residuals, scale, **kwargs)
        assert np.all(np.isfinite(w))
        assert np.all((w >= 0.0) & (w <= 1.0))
