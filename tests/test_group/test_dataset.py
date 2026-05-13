"""Tests for the native GroupDataset contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import (
    group_data_from_csv,
    group_data_from_fmrilm,
    group_data_from_h5,
    group_data_from_nifti,
)
from fmrimod.group import (
    AdapterContractError,
    GroupDataset,
    SampleLabelSpace,
    VoxelSpace,
    adapter_registry,
    group_dataset,
    group_dataset_from_group_data,
    register_core_adapters,
)


class _DummyFmriLm:
    def __init__(self, offset: float = 0.0) -> None:
        self.betas = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float64,
        ) + offset
        self.se = np.full((2, 3), 0.5 + offset, dtype=np.float64)
        self.tstat = self.betas / self.se
        self.n_voxels = 3
        self.coef_names = ("intercept", "task")


def test_group_dataset_validates_assay_axis_contract() -> None:
    ds = group_dataset(
        {"beta": np.ones((2, 3, 1), dtype=np.float32)},
        space=SampleLabelSpace(["r1", "r2"]),
        subjects=["s1", "s2", "s3"],
        contrasts=["c1"],
    )

    assert isinstance(ds, GroupDataset)
    assert ds.shape == (2, 3, 1)
    assert ds.n_samples == 2
    assert ds.n_subjects == 3
    assert ds.n_contrasts == 1
    assert ds.assay("beta").dtype == np.float64
    assert ds.assay_names() == ["beta"]


def test_group_dataset_rejects_shape_and_axis_mismatches() -> None:
    with pytest.raises(AdapterContractError, match="3-D"):
        group_dataset(
            {"beta": np.ones((2, 3))},
            space=SampleLabelSpace(["r1", "r2"]),
            subjects=["s1", "s2", "s3"],
            contrasts=["c1"],
        )
    with pytest.raises(AdapterContractError, match="space samples"):
        group_dataset(
            {"beta": np.ones((2, 3, 1))},
            space=SampleLabelSpace(["r1"]),
            subjects=["s1", "s2", "s3"],
            contrasts=["c1"],
        )
    with pytest.raises(AdapterContractError, match="subjects length"):
        group_dataset(
            {"beta": np.ones((2, 3, 1))},
            space=SampleLabelSpace(["r1", "r2"]),
            subjects=["s1", "s2"],
            contrasts=["c1"],
        )


def test_group_dataset_copies_arrays_and_exposes_readonly_assays() -> None:
    arr = np.ones((1, 1, 1))
    ds = group_dataset(
        {"beta": arr},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1"],
        contrasts=["c1"],
    )

    arr[0, 0, 0] = 99
    assert ds.assay("beta")[0, 0, 0] == 1
    with pytest.raises(ValueError, match="read-only"):
        ds.assay("beta")[0, 0, 0] = 2


def test_group_dataset_validates_axis_metadata_lengths() -> None:
    with pytest.raises(AdapterContractError, match="col_data rows"):
        group_dataset(
            {"beta": np.ones((1, 2, 1))},
            space=SampleLabelSpace(["r1"]),
            subjects=["s1", "s2"],
            contrasts=["c1"],
            col_data=pd.DataFrame({"age": [20]}),
        )


def test_group_dataset_from_group_data_csv_materializes_axis_cube() -> None:
    assert adapter_registry.is_registered("csv")
    frame = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s1", "s2"],
            "roi": ["r1", "r1", "r2", "r2"],
            "contrast": ["faces", "faces", "faces", "faces"],
            "beta": [1.0, 2.0, 3.0, 4.0],
            "se": [0.1, 0.2, 0.3, 0.4],
            "age": [20, 30, 20, 30],
        }
    )
    gd = group_data_from_csv(
        frame,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        contrast_col="contrast",
        covariate_cols=["age"],
    )

    ds = group_dataset_from_group_data(gd)

    assert ds.subjects == ("s1", "s2")
    assert ds.contrasts == ("faces",)
    assert ds.space.labels == ("r1", "r2")
    np.testing.assert_allclose(ds.assay("beta")[:, :, 0], np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(ds.assay("se")[:, :, 0], np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert ds.col_data is not None
    assert ds.col_data["age"].tolist() == [20, 30]


def test_register_core_adapters_restores_builtin_registration() -> None:
    assert adapter_registry.unregister("csv") is True
    try:
        register_core_adapters()
        assert adapter_registry.is_registered("csv")
    finally:
        register_core_adapters()


def test_group_dataset_from_group_data_csv_injects_missing_axes() -> None:
    frame = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "beta": [1.0, 2.0],
            "se": [0.1, 0.2],
        }
    )
    gd = group_data_from_csv(
        frame,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
    )

    ds = group_dataset_from_group_data(gd)

    assert ds.space.labels == ("sample1",)
    assert ds.contrasts == ("c1",)
    np.testing.assert_allclose(ds.assay("beta")[0, :, 0], np.array([1.0, 2.0]))


def test_group_dataset_from_group_data_h5_materializes_axis_cube(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    paths = []
    for subject_idx in range(2):
        path = tmp_path / f"sub-{subject_idx + 1:02d}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("beta", data=np.array([1.0, 2.0]) + subject_idx)
            handle.create_dataset("se", data=np.array([0.1, 0.2]))
        paths.append(path)

    covariates = pd.DataFrame({"age": [20, 30]})
    gd = group_data_from_h5(
        paths,
        subjects=["s1", "s2"],
        covariates=covariates,
        contrast="faces",
        stat=("beta", "var"),
    )

    ds = group_dataset_from_group_data(gd)

    assert ds.metadata["source_format"] == "h5"
    assert ds.subjects == ("s1", "s2")
    assert ds.contrasts == ("faces",)
    assert ds.space.labels == ("sample1", "sample2")
    assert ds.col_data is not None
    assert ds.col_data["age"].tolist() == [20, 30]
    np.testing.assert_allclose(ds.assay("beta")[:, :, 0], np.array([[1.0, 2.0], [2.0, 3.0]]))
    np.testing.assert_allclose(ds.assay("var")[:, :, 0], np.array([[0.01, 0.01], [0.04, 0.04]]))


def test_group_dataset_from_group_data_fmrilm_materializes_coefficients() -> None:
    gd = group_data_from_fmrilm(
        [_DummyFmriLm(), _DummyFmriLm(offset=1.0)],
        subjects=["s1", "s2"],
        stat=("beta", "se", "tstat"),
    )

    ds = group_dataset_from_group_data(gd)

    assert adapter_registry.is_registered("fmrilm")
    assert ds.metadata["source_format"] == "fmrilm"
    assert ds.subjects == ("s1", "s2")
    assert ds.contrasts == ("intercept", "task")
    assert ds.space.labels == ("sample1", "sample2", "sample3")
    np.testing.assert_allclose(ds.assay("beta")[:, 0, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ds.assay("beta")[:, 1, 1], np.array([5.0, 6.0, 7.0]))
    np.testing.assert_allclose(ds.assay("se")[:, 0, 0], np.full(3, 0.5))
    assert "t" in ds.assays


def test_group_dataset_from_group_data_nifti_materializes_voxel_space(tmp_path) -> None:
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("neuroim")

    affine = np.eye(4)
    beta_paths = []
    se_paths = []
    for subject_idx in range(2):
        beta_path = tmp_path / f"sub-{subject_idx + 1:02d}_beta.nii.gz"
        se_path = tmp_path / f"sub-{subject_idx + 1:02d}_se.nii.gz"
        nib.save(
            nib.Nifti1Image(np.array([[[subject_idx + 1.0]], [[10.0]]]), affine),
            beta_path,
        )
        nib.save(
            nib.Nifti1Image(np.full((2, 1, 1), 0.5), affine),
            se_path,
        )
        beta_paths.append(beta_path)
        se_paths.append(se_path)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.array([[[1.0]], [[0.0]]]), affine), mask_path)

    gd = group_data_from_nifti(
        beta_paths=beta_paths,
        se_paths=se_paths,
        subjects=["s1", "s2"],
        mask=mask_path,
        target_space="MNI152",
        validate=False,
    )

    ds = group_dataset_from_group_data(gd)

    assert ds.metadata["source_format"] == "nifti"
    assert ds.subjects == ("s1", "s2")
    assert ds.contrasts == ("c1",)
    assert isinstance(ds.space, VoxelSpace)
    assert ds.space.template_id == "MNI152"
    np.testing.assert_array_equal(ds.space.mask_idx, np.array([0]))
    np.testing.assert_allclose(ds.assay("beta")[:, :, 0], np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(ds.assay("var")[:, :, 0], np.array([[0.25, 0.25]]))
