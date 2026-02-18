"""Tests for group-level dataset construction helpers."""

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset.group_data import (
    GroupData,
    detect_group_data_format,
    group_data,
    group_data_from_csv,
    group_data_from_fmrilm,
    group_data_from_h5,
    group_data_from_nifti,
)


class DummyFmriLm:
    """Minimal fmri_lm-like object for constructor validation tests."""

    def __init__(self):
        self.betas = np.zeros((1, 2))
        self.tstat = np.zeros((1, 2))
        self.se = np.ones((1, 2))
        self.n_voxels = 2


def test_detect_group_data_format_h5(tmp_path):
    h5_file = tmp_path / "sub-01_task-demo.h5"
    h5_file.touch()

    assert detect_group_data_format(str(h5_file)) == "h5"


def test_detect_group_data_format_nifti_and_fmrilm():
    assert detect_group_data_format("sub-01_beta.nii.gz") == "nifti"
    assert detect_group_data_format([DummyFmriLm(), DummyFmriLm()]) == "fmrilm"


def test_detect_group_data_format_reject_multi_csv(tmp_path):
    csv1 = tmp_path / "sub-01.csv"
    csv2 = tmp_path / "sub-02.csv"
    csv1.touch()
    csv2.touch()

    with pytest.raises(ValueError, match="Could not auto-detect group_data format"):
        detect_group_data_format([csv1, csv2])


def test_group_data_auto_dispatch_h5(tmp_path):
    h5_file = tmp_path / "sub-01.h5"
    h5_file.touch()

    gd = group_data([h5_file], subjects=["sub-01"])
    assert isinstance(gd, GroupData)
    assert gd.format == "h5"
    assert gd.subjects == ["sub-01"]


def test_group_data_from_h5_infers_subjects_when_missing(tmp_path):
    h5_a = tmp_path / "sub-01_run-01.h5"
    h5_b = tmp_path / "sub-02_run-01.h5"
    h5_a.touch(); h5_b.touch()

    gd = group_data_from_h5([h5_a, h5_b], validate=False)
    assert gd.subjects == ["sub-01", "sub-02"]


def test_group_data_from_nifti_requires_metadata():
    with pytest.raises(ValueError, match="Must provide either beta_paths or t_paths"):
        group_data_from_nifti()


def test_group_data_from_nifti_requires_df_with_t(tmp_path):
    t1 = tmp_path / "sub-01_t.nii.gz"
    t2 = tmp_path / "sub-02_t.nii.gz"
    t1.touch(); t2.touch()

    with pytest.raises(ValueError, match="df is required"):
        group_data_from_nifti(t_paths=[t1, t2], validate=False)


def test_group_data_from_nifti_accepts_numpy_int_df(tmp_path):
    import numpy as np

    beta = np.zeros((3, 3, 3), dtype=np.float64)
    beta_path = tmp_path / "sub-01_beta.nii.gz"
    nib = pytest.importorskip("nibabel")

    nib.save(nib.Nifti1Image(beta, affine=np.eye(4)), str(beta_path))

    # This should work for NumPy scalar integers passed as df.
    gd = group_data_from_nifti(
        beta_paths=beta_path,
        se_paths=beta_path,
        validate=False,
        df=np.int64(10),
    )
    assert gd.data["df"] == [10]


def test_group_data_from_nifti_rejects_beta_and_t_inputs(tmp_path):
    beta = tmp_path / "sub-01_beta.nii.gz"
    t = tmp_path / "sub-01_t.nii.gz"
    beta.touch()
    t.touch()

    with pytest.raises(ValueError, match="either beta_paths or t_paths"):
        group_data_from_nifti(beta_paths=beta, t_paths=t, validate=False)


def test_group_data_from_nifti_t_paths_validate_matching_shapes(tmp_path):
    nib = pytest.importorskip("nibabel")

    t1 = np.zeros((3, 3, 3), dtype=np.float64)
    t2 = np.zeros((4, 3, 3), dtype=np.float64)

    p1 = tmp_path / "sub-01_t.nii.gz"
    p2 = tmp_path / "sub-02_t.nii.gz"

    nib.save(nib.Nifti1Image(t1, affine=np.eye(4)), str(p1))
    nib.save(nib.Nifti1Image(t2, affine=np.eye(4)), str(p2))

    with pytest.raises(
        ValueError, match="NIfTI dimensions mismatch for .+sub-02_t.nii.gz"
    ):
        group_data_from_nifti(
            t_paths=[p1, p2],
            df=[12, 12],
            validate=True,
        )


def test_group_data_from_nifti_validates_matching_nifti_shapes(tmp_path):
    nib = pytest.importorskip("nibabel")

    beta1 = np.zeros((3, 3, 3), dtype=np.float64)
    beta2 = np.zeros((4, 3, 3), dtype=np.float64)
    se1 = np.ones((3, 3, 3), dtype=np.float64)
    se2 = np.ones((4, 3, 3), dtype=np.float64)

    p1 = tmp_path / "sub-01_beta.nii.gz"
    p2 = tmp_path / "sub-02_beta.nii.gz"
    s1 = tmp_path / "sub-01_se.nii.gz"
    s2 = tmp_path / "sub-02_se.nii.gz"

    nib.save(nib.Nifti1Image(beta1, affine=np.eye(4)), str(p1))
    nib.save(nib.Nifti1Image(beta2, affine=np.eye(4)), str(p2))
    nib.save(nib.Nifti1Image(se1, affine=np.eye(4)), str(s1))
    nib.save(nib.Nifti1Image(se2, affine=np.eye(4)), str(s2))

    with pytest.raises(
        ValueError, match="NIfTI dimensions mismatch for .+sub-02_beta.nii.gz"
    ):
        group_data_from_nifti(
            beta_paths=[p1, p2],
            se_paths=[s1, s2],
            validate=True,
        )


def test_group_data_from_nifti_validates_mask_shape(tmp_path):
    nib = pytest.importorskip("nibabel")

    beta = np.zeros((3, 3, 3), dtype=np.float64)
    se = np.ones((3, 3, 3), dtype=np.float64)
    mask = np.zeros((4, 3, 3), dtype=np.uint8)

    beta_path = tmp_path / "sub-01_beta.nii.gz"
    se_path = tmp_path / "sub-01_se.nii.gz"
    mask_path = tmp_path / "bad_mask.nii.gz"

    nib.save(nib.Nifti1Image(beta, affine=np.eye(4)), str(beta_path))
    nib.save(nib.Nifti1Image(se, affine=np.eye(4)), str(se_path))
    nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))

    with pytest.raises(ValueError, match="Mask dimensions"):
        group_data_from_nifti(
            beta_paths=beta_path,
            se_paths=se_path,
            mask=mask_path,
            validate=True,
        )


def test_group_data_from_nifti_t_mode_validates_mask_shape(tmp_path):
    nib = pytest.importorskip("nibabel")

    t = np.zeros((3, 3, 3), dtype=np.float64)
    mask = np.zeros((4, 3, 3), dtype=np.uint8)

    t_path = tmp_path / "sub-01_t.nii.gz"
    mask_path = tmp_path / "bad_mask.nii.gz"

    nib.save(nib.Nifti1Image(t, affine=np.eye(4)), str(t_path))
    nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))

    with pytest.raises(ValueError, match="Mask dimensions"):
        group_data_from_nifti(
            t_paths=t_path,
            df=10,
            mask=mask_path,
            validate=True,
        )


def test_group_data_from_csv_builds_metadata():
    df = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-02"],
            "beta": [0.1, 0.2],
            "se": [0.01, 0.02],
            "roi": ["roi-1", "roi-1"],
        }
    )

    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
    )

    assert gd.format == "csv"
    assert gd.subjects == ["sub-01", "sub-02"]
    assert gd.data["roi_col"] == "roi"


def test_group_data_from_csv_requires_se_or_var_with_beta():
    df = pd.DataFrame({"subject": ["sub-01"], "beta": [1.23]})

    with pytest.raises(
        ValueError, match="Effect columns must include se or var when beta is provided"
    ):
        group_data_from_csv(df, effect_cols={"beta": "beta"}, subject_col="subject")


def test_group_data_from_fmrilm_dispatch_and_validation():
    models = [DummyFmriLm(), DummyFmriLm()]
    gd = group_data(models, subjects=["s1", "s2"], covariates=None)

    assert gd.format == "fmrilm"
    assert gd.data["contrast"] is None
    assert len(gd.data["lm_list"]) == 2


def test_group_data_from_fmrilm_rejects_non_fmri_lm():
    with pytest.raises(TypeError, match="All elements of lm_list must behave like fmri_lm objects"):
        group_data_from_fmrilm([DummyFmriLm(), object()])
