"""API-level tests for fmri_lm result objects and contrast dispatch."""

import numpy as np
import pytest

from fmrimod.glm.fmri_lm import fmri_lm
from fmrimod.model.config import FmriLmConfig


class _DummyDataset:
    def __init__(self, y: np.ndarray):
        self._y = y

    def get_data(self, run: int) -> np.ndarray:
        if run != 0:
            raise IndexError("only one run is available in dummy dataset")
        return self._y

    def get_censor(self, run: int):
        if run != 0:
            raise IndexError("only one run is available in dummy dataset")
        return None


class _DummyModel:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._x = x
        self.dataset = _DummyDataset(y)
        self.n_runs = 1

    def design_matrix_array(self, run: int) -> np.ndarray:
        if run != 0:
            raise IndexError("only one run is available in dummy model")
        return self._x

    def contrast_weights(self):
        return {"slope": np.array([0.0, 1.0])}


@pytest.fixture
def rng():
    return np.random.default_rng(9)


@pytest.fixture
def fitted_result(rng):
    n, v = 80, 5
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
    beta = np.array([[1.0], [2.0]])
    y = x @ beta + rng.standard_normal((n, v)) * 0.5
    model = _DummyModel(x, y)
    return fmri_lm(model, FmriLmConfig())


def test_fmri_lm_accessors_have_expected_shapes(fitted_result):
    assert fitted_result.coef().shape == (2, 5)
    assert fitted_result.se().shape == (2, 5)
    assert fitted_result.tstat().shape == (2, 5)
    assert fitted_result.n_coefficients == 2
    assert fitted_result.n_voxels == 5


def test_contrast_dispatch_string_and_dict_are_equivalent(fitted_result):
    named = fitted_result.contrast("slope")
    by_dict = fitted_result.contrast(
        {"weights": np.array([0.0, 1.0]), "name": "slope_dict"}
    )

    np.testing.assert_allclose(named.stat, by_dict.stat, atol=1e-12)
    np.testing.assert_allclose(named.p_value, by_dict.p_value, atol=1e-12)
    assert "slope" in fitted_result.contrasts
    assert "slope_dict" in fitted_result.contrasts


def test_unknown_named_contrast_raises_keyerror(fitted_result):
    with pytest.raises(KeyError, match="Unknown contrast name"):
        fitted_result.contrast("does_not_exist")


def test_dict_contrast_requires_weights_key(fitted_result):
    with pytest.raises(ValueError, match="must contain 'weights'"):
        fitted_result.contrast({"name": "bad_spec"})
