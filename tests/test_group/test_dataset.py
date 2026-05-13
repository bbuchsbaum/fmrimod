"""Tests for the native GroupDataset contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv, group_data_from_h5
from fmrimod.group import (
    AdapterContractError,
    GroupDataset,
    SampleLabelSpace,
    UnsupportedGroupFeatureError,
    group_dataset,
    group_dataset_from_group_data,
)


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


def test_group_dataset_from_group_data_rejects_unported_formats(tmp_path) -> None:
    path = tmp_path / "sub-01.h5"
    path.touch()
    gd = group_data_from_h5(path, subjects=["s1"], validate=False)

    with pytest.raises(UnsupportedGroupFeatureError, match="only csv"):
        group_dataset_from_group_data(gd)

