"""Unit tests for fmrigds bridge payload construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fmrimod.dataset import group_data_from_csv, group_data_from_h5, group_data_from_nifti
from fmrimod.stats.backends.fmrigds_backend import _build_bridge_payload
from fmrimod.stats.interfaces import GroupFitRequest


def test_build_bridge_payload_csv_writes_paths(tmp_path):
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "beta": [0.1, 0.2],
            "se": [0.1, 0.1],
            "roi": ["r1", "r1"],
        }
    )
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
    )
    req = GroupFitRequest(data=gd, model="meta", method="fe")

    payload = _build_bridge_payload(req, tmp_path, backend_options={})
    inp = payload["input"]
    assert inp["format"] == "csv"
    assert Path(inp["path"]).exists()
    assert inp["effect_cols"] == {"beta": "beta", "se": "se"}
    assert inp["subject_col"] == "subject"
    assert inp["sample_col"] == "roi"
    assert inp["contrast_col"] == "contrast"


def test_build_bridge_payload_h5_maps_fields_without_serializing_data(tmp_path):
    p1 = tmp_path / "sub-01.h5"
    p2 = tmp_path / "sub-02.h5"
    p1.touch()
    p2.touch()
    gd = group_data_from_h5(
        [p1, p2],
        validate=True,
        contrast="c1",
        stat=("beta", "se"),
    )
    req = GroupFitRequest(data=gd, model="meta", method="fe")

    payload = _build_bridge_payload(req, tmp_path, backend_options={})
    inp = payload["input"]
    assert inp["format"] == "h5"
    assert inp["paths"] == [str(p1), str(p2)]
    assert inp["contrast"] == "c1"
    assert inp["stat"] == ["beta", "se"]
    assert inp["subjects"] == gd.subjects


def test_build_bridge_payload_nifti_maps_fields_without_execution(tmp_path):
    beta_paths = [tmp_path / "sub-01_beta.nii.gz", tmp_path / "sub-02_beta.nii.gz"]
    se_paths = [tmp_path / "sub-01_se.nii.gz", tmp_path / "sub-02_se.nii.gz"]
    gd = group_data_from_nifti(
        beta_paths=beta_paths,
        se_paths=se_paths,
        validate=False,
        subjects=["sub-01", "sub-02"],
    )
    req = GroupFitRequest(data=gd, model="meta", method="fe")

    payload = _build_bridge_payload(req, tmp_path, backend_options={})
    inp = payload["input"]
    assert inp["format"] == "nifti"
    assert inp["beta_paths"] == [str(beta_paths[0]), str(beta_paths[1])]
    assert inp["se_paths"] == [str(se_paths[0]), str(se_paths[1])]
    assert inp["subjects"] == ["sub-01", "sub-02"]
