"""Ports of the R fmrihrf ``test-cli.R`` command-line contract."""

from __future__ import annotations

import json
import os
import re
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


def _write_events(tmp_path, text):
    events = tmp_path / "events.csv"
    events.write_text(text, encoding="utf-8")
    return events


def test_design_response_does_not_spill_into_next_run(tmp_path, capsys):
    """Inventory v2 HRF-06 (fmrihrf ``a8556ff``, #62)."""
    events = _write_events(tmp_path, "onset,condition,block\n230,a,1\n")
    status = fmrihrf_cli(
        ["design", "--events", str(events), "--blocklens", "120,120", "--tr", "2", "--json"]
    )
    rows = json.loads(capsys.readouterr().out)
    assert status == 0
    assert len(rows) == 240
    values = [row["a"] for row in rows]
    assert any(v != 0 for v in values[:120])
    assert all(v == 0 for v in values[120:])


def test_design_equals_per_run_regressor_sets(tmp_path, capsys):
    """Each run is evaluated on its own run-local grid and stacked."""
    import numpy as np

    from fmrimod.hrf import get_hrf
    from fmrimod.regressor.core import Regressor, RegressorSet
    from fmrimod.sampling import SamplingFrame

    table = pd.DataFrame(
        {
            "onset": [5.0, 20.0, 55.0, 3.0, 17.3, 40.0],
            "condition": ["a", "b", "a", "a", "a", "a"],
            "block": [1, 1, 1, 2, 2, 2],
            "duration": [0.0, 2.0, 0.0, 1.0, 0.0, 3.0],
        }
    )
    events = tmp_path / "events.csv"
    table.to_csv(events, index=False)
    status = fmrihrf_cli(
        [
            "design", "--events", str(events), "--blocklens", "30,40",
            "--tr", "2,1.5", "--hrf", "spmg3", "--json",
        ]
    )
    rows = json.loads(capsys.readouterr().out)
    assert status == 0
    got = pd.DataFrame(rows).drop(columns="time")

    sf = SamplingFrame(blocklens=[30, 40], tr=[2.0, 1.5])
    hrf = get_hrf("spmg3")
    expected = []
    for b in (1, 2):
        run = table[table["block"] == b]
        regs = [
            Regressor(
                onsets=run.loc[run["condition"] == lev, "onset"].to_numpy(float),
                hrf=hrf,
                duration=run.loc[run["condition"] == lev, "duration"].to_numpy(float),
                amplitude=np.ones(int((run["condition"] == lev).sum())),
                span=24.0,
            )
            for lev in ("a", "b")
        ]
        rset = RegressorSet(regressors=regs, levels=["a", "b"])
        expected.append(
            rset.evaluate(sf.sample_times(global_time=False, blockids=b - 1))
        )
    expected = np.vstack(expected)

    assert list(got.columns) == RegressorSet(regressors=regs, levels=["a", "b"]).column_names
    # Run 2 has no "b" events: its b columns are zero, not missing.
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(got.to_numpy()[30:, 3:], 0.0)


def test_design_rejects_out_of_range_block(tmp_path, capsys):
    events = _write_events(tmp_path, "onset,condition,block\n5,a,3\n")
    status = fmrihrf_cli(
        ["design", "--events", str(events), "--blocklens", "10,10", "--tr", "2"]
    )
    assert status == 1
    assert "block" in capsys.readouterr().err


def test_design_renamed_duration_or_amplitude_column_must_exist(tmp_path, capsys):
    """Inventory v2 HRF-15 (fmrihrf ``742b1f1``)."""
    events = _write_events(tmp_path, "onset,condition,block\n0,A,1\n4,B,1\n")
    base = ["design", "--events", str(events), "--blocklens", "6", "--tr", "1"]

    # The default duration/amplitude columns stay optional.
    assert fmrihrf_cli([*base, "--output", str(tmp_path / "ok.csv")]) == 0
    capsys.readouterr()

    for option in ("--duration", "--amplitude"):
        status = fmrihrf_cli([*base, option, "dur_s"])
        err = capsys.readouterr().err
        assert status == 1
        assert "Missing required column" in err
        assert "dur_s" in err


_HELP_OPTION = re.compile(r"^\s+--([a-z][a-z-]*)", re.M)


@pytest.mark.parametrize("command", ["list", "eval", "regressor", "design"])
def test_help_lists_exactly_the_accepted_options(command, capsys):
    """Inventory v2 HRF-15 (fmrihrf ``8af0c2d``): help matches the parser."""
    from fmrimod.cli import CliUsageError, _accepted_options

    assert fmrihrf_cli(["help", command]) == 0
    listed = set(_HELP_OPTION.findall(capsys.readouterr().out))
    accepted = _accepted_options(command)
    assert listed == accepted

    # Every listed option is really accepted by the parser, and an option
    # outside the set is rejected.
    from fmrimod.cli import _parse_command

    for name in listed:
        try:
            _parse_command(command, [f"--{name}", "1"])
        except CliUsageError as exc:
            assert "Unknown option" not in str(exc), name
    with pytest.raises(CliUsageError, match="Unknown option"):
        _parse_command(command, ["--definitely-not-an-option"])


def test_help_shows_defaults(capsys):
    fmrihrf_cli(["help", "eval"])
    out = capsys.readouterr().out
    assert "--nbasis" in out and "[5]" in out
    assert "--no-summate" in out
    fmrihrf_cli(["help", "design"])
    out = capsys.readouterr().out
    assert "--duration COL" in out and "[duration]" in out


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
