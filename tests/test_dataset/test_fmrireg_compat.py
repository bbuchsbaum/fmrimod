"""Compatibility tests for fmrireg dataset, IO, and benchmark helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fmrimod
from fmrimod.dataset import (
    group_data_from_csv,
    group_data_from_h5,
    group_data_from_nifti,
)


class _RawModel:
    def __init__(self, x):
        self._x = np.asarray(x, dtype=float)

    def design_matrix_array(self, run=0):
        return self._x

    def design_matrix(self):
        return pd.DataFrame(self._x, columns=["intercept", "slope"])

    def contrast_weights(self):
        return {}


def test_memory_latent_chunks_and_design_plot_helpers():
    data = np.arange(20, dtype=float).reshape(10, 2)
    ds = fmrimod.fmri_mem_dataset(data, tr=2.0)
    np.testing.assert_allclose(ds.get_data(0), data)

    chunks = fmrimod.voxel_index_chunks(ds, nchunks=2)
    assert [chunk.tolist() for chunk in chunks] == [[0], [1]]

    scores = np.column_stack([np.ones(10), np.linspace(-1, 1, 10)])
    loadings = np.array([[1.0, 0.5], [0.0, 2.0]])
    latent = fmrimod.latent_dataset(scores, loadings=loadings, tr=2.0)
    np.testing.assert_allclose(latent.get_data(0), scores @ loadings)

    fit = fmrimod.fmri_latent_lm(_RawModel(scores), latent)
    assert fit.betas.shape == (2, 2)

    plot_df = fmrimod.design_plot(_RawModel(scores))
    assert set(plot_df.columns) == {"time_index", "condition", "value"}


def test_csv_h5_nifti_and_config_readers(tmp_path):
    pytest.importorskip("neuroim")
    h5py = pytest.importorskip("h5py")
    nib = pytest.importorskip("nibabel")
    csv_df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s1", "s2"],
            "roi": ["V1", "V1", "V2", "V2"],
            "contrast": ["A", "A", "B", "B"],
            "beta": [1.0, 2.0, 3.0, 4.0],
            "se": [0.1, 0.2, 0.3, 0.4],
        }
    )
    gd_csv = group_data_from_csv(
        csv_df,
        effect_cols={"beta": "beta", "se": "se"},
        roi_col="roi",
        contrast_col="contrast",
    )
    extracted = fmrimod.extract_csv_data(gd_csv, roi="V1", contrast="A")
    np.testing.assert_allclose(extracted["beta"], [1.0, 2.0])
    np.testing.assert_allclose(extracted["var"], [0.01, 0.04])

    h5_paths = []
    for idx in range(2):
        path = tmp_path / f"s{idx}.h5"
        with h5py.File(path, "w") as handle:
            handle["beta"] = np.array([idx + 1.0, idx + 2.0])
            handle["se"] = np.array([0.1, 0.2])
        h5_paths.append(str(path))
    gd_h5 = group_data_from_h5(h5_paths, subjects=["s1", "s2"], validate=True)
    h5_full = fmrimod.read_h5_full(gd_h5, stat=["beta", "var"])
    assert h5_full.shape == (2, 2, 2)
    np.testing.assert_allclose(h5_full[:, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(h5_full[:, 0, 1], [0.01, 0.04])

    affine = np.eye(4)
    beta_paths = []
    se_paths = []
    for idx in range(2):
        beta_path = tmp_path / f"beta{idx}.nii.gz"
        se_path = tmp_path / f"se{idx}.nii.gz"
        nib.save(nib.Nifti1Image(np.full((2, 1, 1), idx + 1.0), affine), beta_path)
        nib.save(nib.Nifti1Image(np.full((2, 1, 1), 0.5), affine), se_path)
        beta_paths.append(str(beta_path))
        se_paths.append(str(se_path))
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.array([[[1]], [[0]]], dtype=float), affine), mask_path)
    gd_nifti = group_data_from_nifti(
        beta_paths=beta_paths,
        se_paths=se_paths,
        subjects=["s1", "s2"],
        mask=str(mask_path),
        validate=True,
    )
    nifti_full = fmrimod.read_nifti_full(gd_nifti)
    assert nifti_full["beta"].shape == (2, 1)
    np.testing.assert_allclose(nifti_full["var"], np.full((2, 1), 0.25))

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"tr": 2.0, "task": "demo"}))
    assert fmrimod.read_fmri_config(config_path)["tr"] == 2.0


def test_pyproject_declares_dataset_io_extras():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "h5py>=3.8.0" in extras["hdf5"]
    assert "h5py>=3.8.0" in extras["full"]
    assert "nibabel>=4.0.0" in extras["nibabel"]
    assert "nibabel>=4.0.0" in extras["nifti"]
    assert {"pybids>=0.16.0", "nibabel>=4.0.0"}.issubset(extras["bids"])
    # zarr/dask/cache extras were dropped: the consolidation PRD listed
    # them as `adopt` but no backend ever shipped, so declaring the
    # dependency promised a feature the package never imported. See
    # docs/contracts/fmridataset_consolidation_plan_v1.md (status
    # adopt -> deferred) and bd-01KRMR4B2Z4ZPTYHZC7J69CY77.
    assert "zarr" not in extras
    assert "dask" not in extras
    assert "cache" not in extras


def test_basis_registry_and_benchmark_helpers():
    fmrimod.register_basis("constant_basis", lambda scale=1.0: scale)
    assert fmrimod.resolve_basis("constant_basis", scale=3.0) == 3.0

    listing = fmrimod.list_benchmark_datasets()
    assert "BM_Canonical_HighSNR" in set(listing["Dataset"])

    data = fmrimod.load_benchmark_dataset("BM_Canonical_HighSNR")
    assert data["Y_noisy"].ndim == 2

    X = fmrimod.create_design_matrix_from_benchmark("BM_Canonical_HighSNR")
    assert X.shape[0] == data["Y_noisy"].shape[0]
    assert "Intercept" in X.columns

    summary = fmrimod.get_benchmark_summary("BM_Canonical_HighSNR")
    assert summary["dimensions"]["n_timepoints"] == data["Y_noisy"].shape[0]

    perf = fmrimod.evaluate_method_performance(
        "BM_Canonical_HighSNR",
        data["true_betas_condition"],
        method_name="oracle",
    )
    assert perf["overall_metrics"]["rmse"] == 0.0

    np.testing.assert_allclose(
        fmrimod.samples(fmrimod.SamplingFrame(3, tr=2.0)),
        [1.0, 3.0, 5.0],
    )
    evaluated = np.asarray(fmrimod.evaluate(lambda x: x + 1, np.array([1, 2])))
    assert evaluated.tolist() == [2, 3]
