"""Tests for fmrireg-style accessor compatibility helpers."""

import numpy as np
import pandas as pd

import fmrimod
from fmrimod.dataset import FmriDataset, group_data_from_csv
from fmrimod.dataset.adapters import NumpyAdapter
from fmrimod.glm.fmri_lm import fmri_lm
from fmrimod.model import FmriLmConfig, FmriModel


class _DummyDataset:
    def __init__(self, y):
        self._y = y

    def get_data(self, run):
        return self._y

    def get_censor(self, run):
        return None


class _DummyModel:
    def __init__(self, x, y):
        self._x = x
        self.dataset = _DummyDataset(y)
        self.n_runs = 1

    def design_matrix_array(self, run):
        return self._x

    def design_matrix(self):
        return pd.DataFrame(self._x, columns=["intercept", "slope"])

    def contrast_weights(self):
        return {"slope": np.array([0.0, 1.0])}


def _fitted_result():
    rng = np.random.default_rng(12)
    n, v = 60, 4
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([[1.0], [2.0]])
    y = x @ beta + rng.standard_normal((n, v)) * 0.4
    fit = fmri_lm(_DummyModel(x, y), FmriLmConfig())
    fit.ar_params = np.array([[0.2, 0.4, 0.6]])
    fit.contrast("slope")
    fit.contrast(np.eye(2), name="omnibus")
    return fit


def test_fmri_lm_result_accessors_have_expected_shapes():
    fit = _fitted_result()

    assert fmrimod.coef_names(fit) == ["intercept", "slope"]
    np.testing.assert_allclose(fmrimod.ar_parameters(fit), [0.4])
    assert fmrimod.standard_error(fit).shape == fit.betas.shape
    assert fmrimod.se(fit).shape == fit.betas.shape
    assert fmrimod.stats(fit).shape == fit.betas.shape
    assert fmrimod.p_values(fit).shape == fit.betas.shape
    np.testing.assert_allclose(fmrimod.pvalues(fit), fmrimod.p_values(fit))
    assert fmrimod.zscores(fit).shape == fit.betas.shape

    assert fmrimod.get_contrasts(fit) == ["slope", "omnibus"]
    assert set(fmrimod.standard_error(fit, type="contrasts")) == {"slope"}
    assert set(fmrimod.stats(fit, type="F")) == {"omnibus"}


def test_tidy_and_coef_image_accessors():
    fit = _fitted_result()

    tidy = fmrimod.tidy(fit)
    assert list(tidy.columns) == [
        "term",
        "voxel",
        "estimate",
        "std_error",
        "stat",
        "statistic",
        "p_value",
    ]
    assert len(tidy) == fit.n_coefficients * fit.n_voxels
    np.testing.assert_allclose(tidy["statistic"], tidy["stat"])

    vec = fmrimod.coef_image(fit, coef="slope")
    np.testing.assert_allclose(vec, fit.betas[1])

    mask = np.array([[True, False], [True, True], [False, True]])
    img = fmrimod.coef_image(fit, coef=1, mask=mask)
    assert img.shape == mask.shape
    np.testing.assert_allclose(img[mask], fit.betas[1])
    assert np.isnan(img[~mask]).all()


def test_dataset_and_group_accessors():
    sframe = fmrimod.SamplingFrame(blocklens=[3, 2], tr=2.0)
    data = [np.ones((3, 2)), np.full((2, 2), 2.0)]
    mask = np.array([[True, False], [True, False]])
    dataset = FmriDataset(NumpyAdapter(data, sframe, mask=mask))

    np.testing.assert_allclose(fmrimod.get_data(dataset, run=1), data[1])
    np.testing.assert_allclose(
        fmrimod.get_data_matrix(dataset),
        np.vstack(data),
    )
    np.testing.assert_array_equal(fmrimod.get_mask(dataset), mask)

    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s1", "s2"],
            "roi": ["V1", "V1", "V2", "V2"],
            "contrast": ["A", "A", "B", "B"],
            "beta": [1.0, 1.2, 0.5, 0.6],
            "se": [0.1, 0.2, 0.1, 0.2],
            "age": [20, 21, 20, 21],
        }
    )
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        roi_col="roi",
        contrast_col="contrast",
        covariate_cols=["age"],
    )

    assert fmrimod.n_subjects(gd) == 2
    assert fmrimod.get_subjects(gd) == ["s1", "s2"]
    assert list(fmrimod.get_covariates(gd)["age"]) == [20, 21]
    assert fmrimod.get_rois(gd) == ["V1", "V2"]
    assert fmrimod.get_contrasts(gd) == ["A", "B"]


def test_fitted_hrf_and_tidy_fitted_hrf_smoke():
    rng = np.random.default_rng(3)
    sframe = fmrimod.SamplingFrame(blocklens=40, tr=1.0)
    events = pd.DataFrame(
        {
            "onset": [2.0, 10.0, 18.0, 26.0],
            "condition": ["A", "B", "A", "B"],
        }
    )
    event = fmrimod.event_model(
        "onset ~ hrf(condition, basis='spmg1')",
        data=events,
        sampling_frame=sframe,
        durations=1.0,
    )
    baseline = fmrimod.baseline_model("constant", sframe=sframe)
    data = rng.standard_normal((40, 3))
    dataset = FmriDataset(NumpyAdapter(data, sframe), event_table=events)
    fit = fmrimod.fmri_lm(FmriModel(event, baseline, dataset), FmriLmConfig())

    out = fmrimod.fitted_hrf(fit, sample_at=[0.0, 1.0, 2.0])
    assert set(out) == {"condition"}
    assert out["condition"]["pred"].shape == (6, 3)
    assert list(out["condition"]["design"].columns) == ["condition", "time"]

    tidy = fmrimod.tidy_fitted_hrf(fit, sample_at=[0.0, 1.0])
    assert set(tidy.columns) == {
        "term",
        "condition",
        "time",
        "voxel",
        "estimate",
        "value",
    }
    np.testing.assert_allclose(tidy["estimate"], tidy["value"])
