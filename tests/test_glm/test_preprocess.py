"""Parity-focused tests for GLM preprocessing helpers."""

import numpy as np
import pytest

from fmrimod.glm.preprocess import (
    compute_dvars,
    dvars_to_weights,
    dvars_weights,
    extract_nuisance_timeseries,
    volume_weights,
)
import fmrimod


def test_compute_dvars_matches_fmrireg_formula():
    # Two-voxel toy data with deterministic temporal differences.
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ]
    )
    got = compute_dvars(y, normalize=True)

    # fmrireg formula:
    # dvars_raw = sqrt(rowMeans(diff(Y)^2))
    # dvars = c(median(dvars_raw), dvars_raw)
    # dvars = dvars / median(dvars)
    dvars_raw = np.array([1.0, 2.0])
    expected = np.array([1.5, 1.0, 2.0]) / 1.5
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_compute_dvars_requires_at_least_two_timepoints():
    y = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="at least 2 timepoints"):
        compute_dvars(y)


def test_compute_dvars_normalize_false_uses_raw_scale_with_median_first():
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 6.0],
            [6.0, 12.0],
        ],
        dtype=np.float64,
    )
    got = compute_dvars(y, normalize=False)

    dvars_raw = np.sqrt(np.mean(np.diff(y, axis=0) ** 2, axis=1))
    expected = np.concatenate([[np.median(dvars_raw)], dvars_raw])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_inverse_squared_matches_fmrireg_formula():
    dvars = np.array([1.0, 2.0, 3.0])
    got = dvars_weights(dvars, method="inverse_squared")
    expected = 1.0 / (1.0 + dvars**2)
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_soft_threshold_matches_fmrireg_formula():
    dvars = np.array([0.5, 1.5, 3.0])
    threshold = 1.5
    steepness = 2.0
    got = dvars_weights(
        dvars,
        method="soft_threshold",
        threshold=threshold,
        steepness=steepness,
    )
    expected = 1.0 / (
        1.0 + ((np.maximum(dvars, threshold) - threshold) / threshold) ** steepness
    )
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_tukey_matches_fmrireg_formula():
    dvars = np.array([0.5, 2.0, 4.0])
    threshold = 1.5
    got = dvars_weights(dvars, method="tukey", threshold=threshold)
    c_tukey = threshold * 2.0
    u = dvars / c_tukey
    expected = np.where(np.abs(u) <= 1.0, (1.0 - u**2) ** 2, 0.0)
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_rejects_negative_dvars():
    with pytest.raises(ValueError, match="DVARS values must be non-negative"):
        dvars_weights(np.array([0.0, -0.1, 1.0]))


def test_dvars_to_weights_alias_matches_dvars_weights():
    dvars = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    expected = dvars_weights(dvars, method="tukey", threshold=1.25)
    got = dvars_to_weights(dvars, method="tukey", threshold=1.25)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_volume_weights_matches_compute_then_convert():
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 5.0],
            [6.0, 9.0],
        ],
        dtype=np.float64,
    )
    got = volume_weights(y, method="inverse_squared", threshold=1.5)
    expected = dvars_to_weights(
        compute_dvars(y, normalize=True),
        method="inverse_squared",
        threshold=1.5,
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_volume_weights_return_dvars_payload():
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 4.0],
            [3.0, 9.0],
        ],
        dtype=np.float64,
    )
    result = volume_weights(y, return_dvars=True)
    assert set(result.keys()) == {"weights", "dvars"}
    np.testing.assert_allclose(result["dvars"], compute_dvars(y, normalize=True), atol=1e-12)
    np.testing.assert_allclose(
        result["weights"],
        dvars_to_weights(result["dvars"]),
        atol=1e-12,
    )


def test_top_level_volume_quality_helpers_match_glm_preprocess():
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 4.0],
            [3.0, 9.0],
        ],
        dtype=np.float64,
    )
    dvars = fmrimod.compute_dvars(y)
    weights = fmrimod.dvars_to_weights(dvars)
    one_step = fmrimod.volume_weights(y)
    np.testing.assert_allclose(one_step, weights, atol=1e-12)


def test_package_exports_volume_quality_helpers():
    assert "volume_weights" in fmrimod.__all__
    assert "compute_dvars" in fmrimod.__all__
    assert "dvars_to_weights" in fmrimod.__all__


def test_package_volume_weights_matches_module_implementation():
    y = np.array(
        [
            [0.0, 1.0],
            [2.0, 2.0],
            [5.0, 8.0],
            [9.0, 17.0],
        ],
        dtype=np.float64,
    )

    mod = volume_weights(y, method="soft_threshold", threshold=1.2)
    top = fmrimod.volume_weights(y, method="soft_threshold", threshold=1.2)
    np.testing.assert_allclose(top, mod, atol=1e-12)

    mod_d = volume_weights(y, method="tukey", threshold=1.1, return_dvars=True)
    top_d = fmrimod.volume_weights(y, method="tukey", threshold=1.1, return_dvars=True)

    np.testing.assert_array_equal(top_d.keys(), mod_d.keys())
    np.testing.assert_allclose(top_d["weights"], mod_d["weights"], atol=1e-12)
    np.testing.assert_allclose(top_d["dvars"], mod_d["dvars"], atol=1e-12)


def test_extract_nuisance_timeseries_accepts_pathlike_mask_file(tmp_path):
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("neuroim")

    y = np.arange(20, dtype=np.float64).reshape(5, 4)
    mask = np.array([[[1]], [[0]], [[1]], [[0]]], dtype=np.uint8)
    mask_path = tmp_path / "nuisance_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))

    nuisance = extract_nuisance_timeseries(y, mask_path)
    np.testing.assert_array_equal(nuisance, y[:, [0, 2]])


def test_extract_nuisance_timeseries_path_mask_maps_via_dataset_mask(tmp_path):
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("neuroim")

    y = np.arange(15, dtype=np.float64).reshape(5, 3)
    # Full-space nuisance mask (4 voxels), then mapped through dataset_mask.
    nuisance_full = np.array([[[1]], [[0]], [[1]], [[0]]], dtype=np.uint8)
    dataset_mask = np.array([[[1]], [[0]], [[1]], [[1]]], dtype=bool)
    mask_path = tmp_path / "nuisance_full_mask.nii.gz"
    nib.save(nib.Nifti1Image(nuisance_full, affine=np.eye(4)), str(mask_path))

    nuisance = extract_nuisance_timeseries(y, mask_path, dataset_mask=dataset_mask)
    np.testing.assert_array_equal(nuisance, y[:, [0, 1]])


def _two_run_weighted_fit_inputs(n: int = 60):
    import pandas as pd

    rng = np.random.default_rng(8)
    onsets = np.arange(4.0, 100.0, 12.0)
    cond = np.array(["A", "B"] * 10)[: len(onsets)]
    events = pd.DataFrame(
        {
            "onset": np.concatenate([onsets, onsets]),
            "condition": np.concatenate([cond, cond]),
            "duration": 0.0,
            "run": np.repeat([1, 2], len(onsets)),
        }
    )
    Y = rng.standard_normal((2 * n, 4)) + 100.0
    ds = fmrimod.fmri_dataset(Y, tr=2.0, events=events, run_length=[n, n])
    w = np.random.default_rng(30).uniform(0.3, 1.0, 2 * n)
    w[[2, 3, 49, 60, 61, 100]] = 0.0
    return ds, w


def test_zero_volume_weights_carry_no_residual_df():
    # fmrireg 60f8430 (#226) / fb00ab1 (D3): as in stats::lm.wfit, a
    # zero-weight volume is not an observation.
    from fmrimod.model.config import FmriLmConfig, VolumeWeightOptions

    ds, w = _two_run_weighted_fit_inputs()
    fit = fmrimod.fmri_lm(
        "hrf(condition)",
        ds,
        config=FmriLmConfig(volume_weights=VolumeWeightOptions(enabled=True, weights=w)),
    )

    expected = 0
    for run, rows in enumerate((slice(0, 60), slice(60, 120))):
        X_r = fit.model.design_matrix_array(run=run)
        active = w[rows] > 0
        expected += int(active.sum()) - np.linalg.matrix_rank(X_r[active])
    assert fit.residual_df == expected
    for run_result in fit.run_results:
        np.testing.assert_allclose(run_result.sigma2, run_result.rss / run_result.dfres)


def test_volume_weight_vector_alone_enables_weighting():
    # fmrireg e30837e: a weight vector alone turns volume weighting on.
    from fmrimod.model.config import FmriLmConfig, VolumeWeightOptions

    ds, w = _two_run_weighted_fit_inputs()
    implied = fmrimod.fmri_lm(
        "hrf(condition)",
        ds,
        config=FmriLmConfig(volume_weights=VolumeWeightOptions(weights=w)),
    )
    explicit = fmrimod.fmri_lm(
        "hrf(condition)",
        ds,
        config=FmriLmConfig(volume_weights=VolumeWeightOptions(enabled=True, weights=w)),
    )
    unweighted = fmrimod.fmri_lm("hrf(condition)", ds)

    assert implied.config.volume_weights.enabled is True
    np.testing.assert_array_equal(implied.betas, explicit.betas)
    assert np.max(np.abs(implied.betas - unweighted.betas)) > 1e-3
