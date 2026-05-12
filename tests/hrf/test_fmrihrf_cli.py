"""Ports of the R fmrihrf ``test-cli.R`` command-line contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from fmrimod.cli import fmrihrf_cli, install_cli


def test_fmrihrf_cli_help_paths_return_success(capsys):
    status = fmrihrf_cli(["--help"])
    out = capsys.readouterr().out
    assert status == 0
    assert "Usage: fmrihrf" in out

    status = fmrihrf_cli(["help", "eval"])
    out = capsys.readouterr().out
    assert status == 0
    assert "Usage: fmrihrf eval" in out

    status = fmrihrf_cli(["eval", "--help"])
    out = capsys.readouterr().out
    assert status == 0
    assert "Usage: fmrihrf eval" in out


def test_parser_rejects_unknown_options_and_missing_values():
    assert fmrihrf_cli(["list", "--bogus"]) == 2
    assert fmrihrf_cli(["eval", "--hrf"]) == 2


def test_parser_supports_equals_values_and_negative_numeric_values(capsys):
    status = fmrihrf_cli(["eval", "--hrf=spmg1", "--times", "-0.5,0,0.5"])
    out = capsys.readouterr().out.splitlines()

    assert status == 0
    assert out[0] == '"time","value"'
    assert out[1].startswith("-0.5,")


def test_list_command_emits_parseable_json(capsys):
    status = fmrihrf_cli(["list", "--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)

    assert status == 0
    assert {"name", "type", "nbasis_default"}.issubset(parsed[0])
    assert "spmg1" in {row["name"] for row in parsed}


def test_eval_command_emits_expected_columns(tmp_path):
    path = tmp_path / "eval.csv"
    status = fmrihrf_cli(
        [
            "eval",
            "--hrf",
            "spmg3",
            "--from",
            "0",
            "--to",
            "1",
            "--by",
            "1",
            "--output",
            str(path),
        ]
    )
    out = pd.read_csv(path)

    assert status == 0
    assert list(out.columns) == ["time", "basis1", "basis2", "basis3"]
    assert out["time"].tolist() == [0, 1]


def test_regressor_command_evaluates_onsets_over_sampling_frame(tmp_path):
    path = tmp_path / "regressor.csv"
    status = fmrihrf_cli(
        [
            "regressor",
            "--onsets",
            "0,4",
            "--blocklens",
            "4",
            "--tr",
            "1",
            "--hrf",
            "spmg1",
            "--output",
            str(path),
        ]
    )
    out = pd.read_csv(path)

    assert status == 0
    assert list(out.columns) == ["time", "value"]
    assert len(out) == 4


def test_design_command_creates_condition_columns_from_events_table(tmp_path):
    events = tmp_path / "events.csv"
    events.write_text(
        "onset,condition,block,duration,amplitude\n"
        "0,A,1,0,1\n"
        "4,B,1,0,1\n",
        encoding="utf-8",
    )
    path = tmp_path / "design.csv"
    status = fmrihrf_cli(
        [
            "design",
            "--events",
            str(events),
            "--blocklens",
            "6",
            "--tr",
            "1",
            "--output",
            str(path),
        ]
    )
    out = pd.read_csv(path)

    assert status == 0
    assert list(out.columns) == ["time", "A", "B"]
    assert len(out) == 6


def test_install_cli_copies_wrapper_and_refuses_accidental_overwrite(tmp_path):
    installed = install_cli(tmp_path)
    wrapper = installed["fmrihrf"]

    assert os.path.exists(wrapper)
    assert os.access(wrapper, os.X_OK)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        install_cli(tmp_path)


def test_module_wrapper_has_valid_help_path():
    result = subprocess.run(
        [sys.executable, "-m", "fmrimod.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage: fmrihrf" in result.stdout
