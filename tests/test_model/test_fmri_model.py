"""Model regression tests for fMRI model edge cases."""

import numpy as np
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


class _DummyDataset:
    """Minimal dataset protocol used by :class:`FmriModel`."""

    def __init__(self, n_timepoints):
        self._n_timepoints = n_timepoints

    @property
    def n_timepoints(self):
        return self._n_timepoints

    @property
    def n_runs(self):
        return len(self._n_timepoints)


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
