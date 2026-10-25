"""Unit tests for fmrigds backend helpers (no R process required)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats.backends import fmrigds_backend
from fmrimod.stats.interfaces import GroupFitRequest


def test_fmrigds_backend_available_false_when_rscript_missing(monkeypatch):
    monkeypatch.setattr(fmrigds_backend.shutil, "which", lambda _: None)
    ok, reason = fmrigds_backend.fmrigds_backend_available()
    assert ok is False
    assert "Rscript not found" in reason


def test_fmrigds_backend_available_parses_ok(monkeypatch):
    monkeypatch.setattr(fmrigds_backend.shutil, "which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr(
        fmrigds_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    ok, reason = fmrigds_backend.fmrigds_backend_available()
    assert ok is True
    assert reason == "ok"


def test_fmrigds_backend_available_parses_missing_pkgload(monkeypatch):
    monkeypatch.setattr(fmrigds_backend.shutil, "which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr(
        fmrigds_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="missing_pkgload", stderr=""),
    )
    ok, reason = fmrigds_backend.fmrigds_backend_available(fmrigds_source="/tmp/fmrigds")
    assert ok is False
    assert "pkgload" in reason


def test_fmrigds_backend_available_reports_subprocess_failure(monkeypatch):
    monkeypatch.setattr(fmrigds_backend.shutil, "which", lambda _: "/usr/bin/Rscript")
    monkeypatch.setattr(
        fmrigds_backend.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    ok, reason = fmrigds_backend.fmrigds_backend_available()
    assert ok is False
    assert reason == "boom"


def test_build_bridge_payload_csv_injects_axis_columns_and_covariates(tmp_path):
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "beta": [0.2, 0.3],
            "se": [0.1, 0.1],
        }
    )
    cov = pd.DataFrame({"age": [20, 30]})
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        subjects=["s1", "s2"],
        covariates=cov,
    )
    req = GroupFitRequest(data=gd, model="meta", method="fe")

    payload = fmrigds_backend._build_bridge_payload(req, tmp_path, backend_options={})
    inp = payload["input"]

    csv_df = pd.read_csv(inp["path"])
    assert "sample" in csv_df.columns
    assert "contrast" in csv_df.columns
    assert set(csv_df["sample"]) == {"sample1"}
    assert set(csv_df["contrast"]) == {"c1"}

    cov_df = pd.read_csv(inp["covariates_path"])
    assert list(cov_df.columns)[0] == "subject"
    assert cov_df["subject"].tolist() == ["s1", "s2"]


@pytest.mark.parametrize("weights", ["equal", "custom"])
def test_fmrigds_backend_refuses_non_ivw_weights_before_rscript(monkeypatch, weights):
    # fmrigds e359af6, R/verb-reduce.R:239-262: reduce() now errors for weight
    # schemes an inverse-variance meta:* reducer would silently ignore.
    def no_rscript(*args, **kwargs):
        raise AssertionError("Rscript must not be invoked")

    monkeypatch.setattr(fmrigds_backend.subprocess, "run", no_rscript)
    gd = group_data_from_csv(
        pd.DataFrame({"subject": ["s1", "s2"], "beta": [0.2, 0.3], "se": [0.1, 0.1]}),
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
    )
    req = GroupFitRequest(data=gd, model="meta", method="fe", weights=weights)

    with pytest.raises(NotImplementedError, match="weights"):
        fmrigds_backend.FmrigdsBackend().fit(req)
