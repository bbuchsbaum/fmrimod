"""Top-level wrapper tests for group-data constructors."""

import numpy as np
import pandas as pd

import fmrimod


class DummyFmriLm:
    """Minimal fmri_lm-like object for constructor validation tests."""

    def __init__(self):
        self.betas = np.zeros((1, 2))
        self.tstat = np.zeros((1, 2))
        self.se = np.ones((1, 2))
        self.n_voxels = 2


def test_top_level_detect_group_data_format_nifti():
    assert fmrimod.detect_group_data_format("sub-01_beta.nii.gz") == "nifti"


def test_top_level_group_data_from_h5_and_auto_dispatch(tmp_path):
    h5_path = tmp_path / "sub-01_task-demo.h5"
    h5_path.touch()

    gd_direct = fmrimod.group_data_from_h5([h5_path], subjects=["sub-01"], validate=True)
    gd_auto = fmrimod.group_data([h5_path], subjects=["sub-01"])

    assert gd_direct.format == "h5"
    assert gd_auto.format == "h5"
    assert gd_direct.subjects == ["sub-01"]
    assert gd_auto.subjects == ["sub-01"]


def test_top_level_group_data_from_csv():
    df = pd.DataFrame(
        {
            "subject": ["sub-01", "sub-02"],
            "beta": [0.1, 0.2],
            "se": [0.01, 0.02],
        }
    )
    gd = fmrimod.group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
    )
    assert gd.format == "csv"
    assert gd.subjects == ["sub-01", "sub-02"]


def test_top_level_group_data_from_fmrilm():
    gd = fmrimod.group_data_from_fmrilm([DummyFmriLm(), DummyFmriLm()], subjects=["s1", "s2"])
    assert gd.format == "fmrilm"
    assert gd.subjects == ["s1", "s2"]

