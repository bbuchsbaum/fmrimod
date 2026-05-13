"""Tests for ContrastResult spatial reverse converters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fmrimod.dataset.adapters import NumpyAdapter
from fmrimod.dataset.fmri_dataset import FmriDataset
from fmrimod.glm import combine_runs, fit_glm_from_suffstats
from fmrimod.glm.contrasts import ContrastResult
from fmrimod.sampling import SamplingFrame

neuroim = pytest.importorskip("neuroim")


def _mask() -> np.ndarray:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 1, 1] = True
    mask[1, 0, 1] = True
    return mask


def _fit(seed: int = 1):
    rng = np.random.default_rng(seed)
    mask = _mask()
    x = np.column_stack(
        [
            np.ones(9, dtype=np.float64),
            np.linspace(-1.0, 1.0, 9, dtype=np.float64),
        ]
    )
    y = rng.normal(size=(x.shape[0], int(mask.sum()))).astype(np.float64)
    dataset = FmriDataset(
        NumpyAdapter(
            y,
            SamplingFrame(blocklens=[x.shape[0]], tr=1.0),
            mask=mask,
        )
    )
    model = SimpleNamespace(dataset=dataset)
    return fit_glm_from_suffstats(
        model=model,
        XtX=x.T @ x,
        XtS=x.T @ y,
        StS=np.sum(y * y, axis=0),
        df=float(x.shape[0] - x.shape[1]),
    )


def test_contrast_to_neurovol_reconstructs_masked_statistic():
    fit = _fit()
    result = fit.contrast(np.array([0.0, 1.0]), name="slope")

    vol = result.to_neurovol("stat", fill=-1.0)
    assert isinstance(vol, neuroim.DenseNeuroVol)
    assert vol.label == "slope.stat"

    expected = np.full(_mask().shape, -1.0, dtype=np.float64)
    expected[_mask()] = result.stat
    np.testing.assert_allclose(vol.data, expected)


def test_contrast_to_neurovec_stacks_t_statistics():
    result = _fit().contrast(np.array([0.0, 1.0]), name="slope")

    vec = result.to_neurovec()
    assert isinstance(vec, neuroim.DenseNeuroVec)
    assert tuple(vec.data.shape) == _mask().shape + (3,)
    np.testing.assert_allclose(vec.data[..., 0][_mask()], result.estimate)
    np.testing.assert_allclose(vec.data[..., 1][_mask()], result.stat)
    np.testing.assert_allclose(vec.data[..., 2][_mask()], result.p_value)


def test_contrast_to_neurovec_supports_multirow_f_estimates():
    result = _fit().contrast(np.eye(2), name="omnibus")

    vec = result.to_neurovec(kinds=("estimate", "stat"))
    assert tuple(vec.data.shape) == _mask().shape + (3,)
    np.testing.assert_allclose(vec.data[..., 0][_mask()], result.estimate[0])
    np.testing.assert_allclose(vec.data[..., 1][_mask()], result.estimate[1])
    np.testing.assert_allclose(vec.data[..., 2][_mask()], result.stat)


def test_contrast_to_nifti_writes_masked_volume(tmp_path):
    result = _fit().contrast(np.array([0.0, 1.0]), name="slope")
    out = result.to_nifti(tmp_path / "slope_stat.nii.gz")
    assert out.exists()


def test_contrast_reverse_converters_require_spatial_context():
    result = ContrastResult(
        name="bare",
        estimate=np.array([1.0, 2.0]),
        stat=np.array([3.0, 4.0]),
        se=np.array([0.5, 0.5]),
        p_value=np.array([0.1, 0.2]),
        df=5.0,
        stat_type="t",
    )
    with pytest.raises(ValueError, match="no spatial context"):
        result.to_neurovol()


def test_combine_runs_preserves_spatial_context():
    fits = [_fit(2), _fit(3)]
    result = combine_runs(fits).contrast(np.array([0.0, 1.0]), name="slope")

    assert result.spatial is not None
    vol = result.to_neurovol("estimate", fill=0.0)
    assert tuple(vol.data.shape) == _mask().shape
