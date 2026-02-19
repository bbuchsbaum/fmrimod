"""Model regression tests for fMRI model edge cases."""

import numpy as np
import pandas as pd
import pytest

import fmrimod
from fmrimod.model.fmri_model import FmriModel
from fmrimod.betas.extraction import estimate_betas_lss, estimate_betas_ols


class _DummyEvent:
    """Minimal event model exposing an ndarray design matrix."""

    def __init__(self, design_matrix):
        self.design_matrix = design_matrix


class _DummyBaseline:
    """Minimal baseline model placeholder."""


class _DummyBaselineWithDesign:
    """Minimal baseline model exposing a design matrix attribute."""

    def __init__(self, design_matrix):
        self.design_matrix = design_matrix


class _DummyDataset:
    """Minimal dataset protocol used by :class:`FmriModel`."""

    def __init__(self, n_timepoints):
        self._n_timepoints = n_timepoints
        self._sampling_frame = object()

    @property
    def n_timepoints(self):
        return self._n_timepoints

    @property
    def n_runs(self):
        return len(self._n_timepoints)

    def get_sampling_frame(self):
        return self._sampling_frame


class TestFmriModelRunwiseNdarray:
    def test_design_matrix_run_slice_ndarray_event_dm(self):
        """Run slicing must support ndarray event design matrices."""
        event_dm = np.arange(12).reshape(6, 2)
        model = FmriModel(
            _DummyEvent(event_dm),
            _DummyBaseline(),
            _DummyDataset([3, 3]),
        )

        dm = model._get_event_dm(run=0)
        assert dm.shape == (3, 2)
        np.testing.assert_allclose(dm.to_numpy(), event_dm[:3])

    def test_full_design_matrix_concatenates_event_and_baseline(self):
        event_dm = pd.DataFrame(
            {"e1": [1, 2, 3, 4, 5, 6], "e2": [6, 5, 4, 3, 2, 1]}
        )
        baseline_dm = pd.DataFrame({"b1": [10, 10, 10, 20, 20, 20]})
        model = FmriModel(
            _DummyEvent(event_dm),
            _DummyBaselineWithDesign(baseline_dm),
            _DummyDataset([3, 3]),
        )

        full = model.design_matrix()
        run1 = model.design_matrix(run=1)

        assert full.shape == (6, 3)
        assert run1.shape == (3, 3)
        np.testing.assert_allclose(
            run1.to_numpy(),
            np.column_stack([event_dm.iloc[3:6].to_numpy(), baseline_dm.iloc[3:6].to_numpy()]),
        )

    def test_design_matrix_array_is_float64(self):
        event_dm = np.arange(12).reshape(6, 2)
        baseline_dm = np.ones((6, 1))
        model = FmriModel(
            _DummyEvent(event_dm),
            _DummyBaselineWithDesign(baseline_dm),
            _DummyDataset([2, 4]),
        )

        arr = model.design_matrix_array(run=0)
        assert arr.dtype == np.float64
        assert arr.shape == (2, 3)

    def test_column_index_partition_is_consistent(self):
        event_dm = np.arange(18).reshape(6, 3)
        baseline_dm = np.arange(12).reshape(6, 2)
        model = FmriModel(
            _DummyEvent(event_dm),
            _DummyBaselineWithDesign(baseline_dm),
            _DummyDataset([3, 3]),
        )

        eidx = model.event_column_indices
        bidx = model.baseline_column_indices
        assert len(eidx) == model.n_event_columns
        assert len(bidx) == model.n_baseline_columns
        np.testing.assert_array_equal(
            np.concatenate([eidx, bidx]),
            np.arange(model.n_columns),
        )

    def test_missing_baseline_design_matrix_raises(self):
        event_dm = np.arange(12).reshape(6, 2)
        model = FmriModel(
            _DummyEvent(event_dm),
            _DummyBaseline(),
            _DummyDataset([3, 3]),
        )
        with pytest.raises(TypeError, match="baseline_model does not have a design_matrix"):
            model.baseline_design_matrix()

    def test_contrast_weights_delegates_to_event_model(self):
        class _EventWithContrasts(_DummyEvent):
            def contrast_weights(self, scale=1.0):
                return {"c1": np.array([scale, -scale])}

        model = FmriModel(
            _EventWithContrasts(np.arange(12).reshape(6, 2)),
            _DummyBaselineWithDesign(np.ones((6, 1))),
            _DummyDataset([3, 3]),
        )
        out = model.contrast_weights(scale=2.0)
        np.testing.assert_allclose(out["c1"], np.array([2.0, -2.0]))


class TestTopLevelGlmWrappers:
    @pytest.mark.parametrize("method", ["ols", "lss"])
    def test_top_level_exports_exist(self, method):
        assert hasattr(fmrimod, f"glm_{method}")

    @pytest.mark.parametrize(
        "call_fn,impl_fn",
        [
            (fmrimod.glm_ols, estimate_betas_ols),
            (fmrimod.glm_lss, estimate_betas_lss),
        ],
    )
    def test_top_level_glm_wrappers_aware(self, call_fn, impl_fn):
        trial_regressors = np.eye(4)
        y = np.arange(16).reshape(4, 4)
        left = call_fn(trial_regressors, y)
        right = impl_fn(trial_regressors, y)
        assert left.method == right.method
        assert left.betas.shape == right.betas.shape

    def test_glm_lss_accepts_deprecated_use_cpp_kwarg(self):
        trial_regressors = np.eye(4)
        y = np.arange(16).reshape(4, 4)

        baseline = fmrimod.glm_lss(trial_regressors, y)
        with pytest.warns(DeprecationWarning, match="use_cpp"):
            compat = fmrimod.glm_lss(trial_regressors, y, use_cpp=False)

        assert compat.method == baseline.method
        np.testing.assert_allclose(compat.betas, baseline.betas)

    @pytest.mark.parametrize("call_fn", [fmrimod.glm_ols, fmrimod.glm_lss])
    def test_glm_wrappers_accept_deprecated_progress_kwarg(self, call_fn):
        trial_regressors = np.eye(4)
        y = np.arange(16).reshape(4, 4)

        baseline = call_fn(trial_regressors, y)
        with pytest.warns(DeprecationWarning, match="progress"):
            compat = call_fn(trial_regressors, y, progress=True)

        assert compat.method == baseline.method
        np.testing.assert_allclose(compat.betas, baseline.betas)
