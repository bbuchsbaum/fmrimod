"""Small fmrihrf-compatible command line interface.

The Python package is named ``fmrimod``, but the HRF surface is a port of the
R ``fmrihrf`` package.  This module keeps the R CLI test contract available for
the Python port without making command-line behavior depend on R.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .hrf.registry import get_hrf, list_available_hrfs
from .regressor.core import regressor, regressor_set
from .regressor.design import regressor_design
from .sampling import SamplingFrame


class CliUsageError(ValueError):
    """Usage error that maps to exit status 2."""


class CliDomainError(RuntimeError):
    """Domain error that maps to exit status 1."""


def fmrihrf_cli(args: Sequence[str] | None = None) -> int:
    """Run the fmrihrf-compatible CLI and return an integer status."""
    if args is None:
        args = sys.argv[1:]
    try:
        return _cli_main(list(args))
    except CliDomainError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except CliUsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"fmrihrf: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    """Console-script entry point."""
    status = fmrihrf_cli()
    raise SystemExit(status)


def install_cli(
    dest_dir: str | os.PathLike[str] = "~/.local/bin",
    *,
    overwrite: bool = False,
    commands: Sequence[str] | None = None,
) -> dict[str, str]:
    """Install lightweight command wrappers.

    Parameters mirror the R ``install_cli()`` helper closely enough for ported
    tests and local use.  Installed wrappers execute ``python -m fmrimod.cli``.
    """
    command_map = {"fmrihrf": "fmrihrf"}
    if commands is None:
        commands = tuple(command_map)

    unknown = sorted(set(commands).difference(command_map))
    if unknown:
        raise ValueError(f"Unknown command(s): {', '.join(unknown)}")

    dest = Path(dest_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    installed: dict[str, str] = {}
    for command in commands:
        path = dest / command
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing command: {path}\n"
                "Use overwrite=True to replace it."
            )
        path.write_text(
            "#!/bin/sh\n"
            f"exec {sys.executable!r} -m fmrimod.cli \"$@\"\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        installed[command] = str(path)
    return installed


def _cli_main(args: list[str]) -> int:
    if not args or args[0] in {"-h", "--help", "help"}:
        if len(args) >= 2 and args[0] == "help":
            return _cli_help(args[1])
        return _cli_help()

    command = args[0]
    command_args = args[1:]
    if "--help" in command_args or "-h" in command_args:
        return _cli_help(command)

    if command == "list":
        return _cli_list(command_args)
    if command == "eval":
        return _cli_eval(command_args)
    if command == "regressor":
        return _cli_regressor(command_args)
    if command == "design":
        return _cli_design(command_args)

    raise CliUsageError(
        f"Unknown command: {command}\nRun `fmrihrf --help` for available commands."
    )


def _cli_help(command: str | None = None) -> int:
    topic = command or "main"
    help_text = {
        "main": [
            "Usage: fmrihrf <command> [options]",
            "",
            "Commands:",
            "  list        List available HRFs and basis generators",
            "  eval        Evaluate an HRF or basis over a time grid",
            "  regressor   Build and evaluate one event regressor",
            "  design      Build a condition design matrix from an events table",
            "",
            "Run `fmrihrf help <command>` for command-specific options.",
        ],
        "list": [
            "Usage: fmrihrf list [--details] [--json] [--output FILE]",
        ],
        "eval": [
            "Usage: fmrihrf eval [options]",
            "  --hrf NAME          HRF name from `fmrihrf list` [spmg1]",
            "  --from SEC          Grid start in seconds [0]",
            "  --to SEC            Grid end in seconds [32]",
            "  --by SEC            Grid step in seconds [0.5]",
            "  --times LIST        Comma-separated grid values",
            "  --output FILE       Write output to FILE instead of stdout",
        ],
        "regressor": [
            "Usage: fmrihrf regressor [options]",
            "  --onsets LIST       Comma-separated onset times in seconds",
            "  --blocklens LIST    Scans per block for acquisition grid",
            "  --tr LIST           Repetition time(s) in seconds",
            "  --output FILE       Write output to FILE instead of stdout",
        ],
        "design": [
            "Usage: fmrihrf design --events FILE --blocklens LIST --tr LIST [options]",
        ],
    }
    if topic not in help_text:
        raise CliUsageError(f"Unknown help topic: {topic}")
    print("\n".join(help_text[topic]))
    return 0


def _cli_list(args: list[str]) -> int:
    opts = _parse_cli_args(args, flags={"details", "json"}, values={"output"})
    rows = list_available_hrfs(details=True)
    _write_cli_table(rows, json_output=bool(opts.get("json")), output=opts.get("output"))
    return 0


def _cli_eval(args: list[str]) -> int:
    opts = _parse_cli_args(
        args,
        flags={"normalize", "json", "summate"},
        false_flags={"summate"},
        values={
            "hrf",
            "from",
            "to",
            "by",
            "times",
            "nbasis",
            "span",
            "lag",
            "width",
            "amplitude",
            "duration",
            "precision",
            "output",
        },
    )
    opts = _defaults(
        opts,
        {
            "hrf": "spmg1",
            "from": "0",
            "to": "32",
            "by": "0.5",
            "nbasis": "5",
            "span": "24",
            "lag": "0",
            "width": "0",
            "amplitude": "1",
            "duration": "0",
            "precision": "0.2",
            "summate": True,
        },
    )

    grid = _grid_from_options(opts)
    hrf = _get_cli_hrf(opts)
    values = hrf.evaluate(
        grid,
        duration=_as_scalar_numeric(opts["duration"], "duration"),
        precision=_as_scalar_numeric(opts["precision"], "precision"),
        summate=bool(opts["summate"]),
        normalize=bool(opts.get("normalize")),
    )
    values = values * _as_scalar_numeric(opts["amplitude"], "amplitude")
    _write_cli_table(
        _values_to_rows(grid, values),
        json_output=bool(opts.get("json")),
        output=opts.get("output"),
    )
    return 0


def _cli_regressor(args: list[str]) -> int:
    opts = _parse_cli_args(
        args,
        flags={"normalize", "json"},
        values={
            "onsets",
            "events",
            "hrf",
            "blocklens",
            "tr",
            "start-time",
            "from",
            "to",
            "by",
            "duration",
            "amplitude",
            "nbasis",
            "span",
            "precision",
            "method",
            "output",
        },
    )
    opts = _defaults(
        opts,
        {
            "hrf": "spmg1",
            "by": "1",
            "duration": "0",
            "amplitude": "1",
            "nbasis": "5",
            "span": "24",
            "precision": "0.33",
            "method": "conv",
        },
    )

    events = _events_for_regressor(opts)
    hrf = _get_cli_hrf(opts, allow_decorators=False)
    reg = regressor(
        onsets=events["onset"].to_numpy(float),
        hrf=hrf,
        duration=events["duration"].to_numpy(float),
        amplitude=events["amplitude"].to_numpy(float),
        span=_as_scalar_numeric(opts["span"], "span"),
    )
    grid = _regressor_grid(opts, events["onset"].to_numpy(float))
    values = reg.evaluate(
        grid,
        precision=_as_scalar_numeric(opts["precision"], "precision"),
        method=_match_choice(opts["method"], {"conv", "fft", "direct", "loop", "Rconv"}, "method"),
        normalize=bool(opts.get("normalize")),
    )
    _write_cli_table(
        _values_to_rows(grid, values),
        json_output=bool(opts.get("json")),
        output=opts.get("output"),
    )
    return 0


def _cli_design(args: list[str]) -> int:
    opts = _parse_cli_args(
        args,
        flags={"sparse", "json"},
        values={
            "events",
            "condition",
            "onset",
            "block",
            "duration",
            "amplitude",
            "blocklens",
            "tr",
            "start-time",
            "hrf",
            "nbasis",
            "span",
            "precision",
            "method",
            "output",
        },
    )
    opts = _defaults(
        opts,
        {
            "condition": "condition",
            "onset": "onset",
            "block": "block",
            "duration": "duration",
            "amplitude": "amplitude",
            "hrf": "spmg1",
            "nbasis": "5",
            "span": "24",
            "precision": "0.33",
            "method": "conv",
        },
    )
    _require_options(opts, ("events", "blocklens", "tr"))

    events = _read_events_table(str(opts["events"]))
    _require_columns(events, (str(opts["onset"]), str(opts["condition"]), str(opts["block"])))
    sf = _sampling_frame_from_options(opts)
    hrf = _get_cli_hrf(opts, allow_decorators=False)

    blockids = events[str(opts["block"])].to_numpy(int)
    if blockids.size and np.min(blockids) >= 1:
        blockids = blockids - 1
    global_event_onsets = sf.global_onsets(
        events[str(opts["onset"])].to_numpy(float),
        blockids,
    )

    duration = (
        events[str(opts["duration"])].to_numpy(float)
        if str(opts["duration"]) in events
        else 0.0
    )
    amplitude = (
        events[str(opts["amplitude"])].to_numpy(float)
        if str(opts["amplitude"]) in events
        else 1.0
    )
    rset = regressor_set(
        onsets=global_event_onsets,
        fac=events[str(opts["condition"])].to_numpy(),
        hrf=hrf,
        duration=duration,
        amplitude=amplitude,
        span=_as_scalar_numeric(opts["span"], "span"),
    )
    grid = sf.sample_times(global_time=True)
    design = regressor_design(
        rset,
        grid,
        precision=_as_scalar_numeric(opts["precision"], "precision"),
        method=_match_choice(opts["method"], {"conv", "fft", "direct", "loop", "Rconv"}, "method"),
        sparse=bool(opts.get("sparse")),
    )
    if hasattr(design, "toarray"):
        design = design.toarray()
    rows = _values_to_rows(grid, np.asarray(design), column_names=_design_column_names(rset))
    _write_cli_table(rows, json_output=bool(opts.get("json")), output=opts.get("output"))
    return 0


def _parse_cli_args(
    args: Sequence[str],
    *,
    flags: set[str] | None = None,
    values: set[str] | None = None,
    false_flags: set[str] | None = None,
) -> dict[str, Any]:
    flags = flags or set()
    values = values or set()
    false_flags = false_flags or set()
    out: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            raise CliUsageError(f"Unexpected positional argument: {token}")
        token = token[2:]
        value: str | None = None
        if "=" in token:
            key, value = token.split("=", 1)
        else:
            key = token

        if key.startswith("no-"):
            flag = key[3:]
            if flag not in false_flags:
                raise CliUsageError(f"Unknown option: --{key}")
            if value is not None:
                raise CliUsageError(f"Boolean option does not take a value: --{key}")
            out[flag] = False
            i += 1
            continue

        if key in flags:
            if value is not None:
                raise CliUsageError(f"Boolean option does not take a value: --{key}")
            out[key] = True
            i += 1
            continue

        if key in values:
            if value is None:
                if i == len(args) - 1:
                    raise CliUsageError(f"Missing value for option: --{key}")
                value = args[i + 1]
                if value.startswith("--"):
                    raise CliUsageError(f"Missing value for option: --{key}")
                i += 1
            out[key] = value
            i += 1
            continue

        raise CliUsageError(f"Unknown option: --{key}")
    return out


def _defaults(opts: dict[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in defaults.items():
        opts.setdefault(key, value)
    return opts


def _require_options(opts: Mapping[str, Any], names: Iterable[str]) -> None:
    missing = [name for name in names if opts.get(name) is None]
    if missing:
        raise CliUsageError(
            "Missing required option(s): "
            + ", ".join(f"--{name}" for name in missing)
        )


def _require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise CliDomainError("Missing required column(s): " + ", ".join(missing))


def _grid_from_options(opts: Mapping[str, Any]) -> np.ndarray:
    if opts.get("times") is not None:
        values = _parse_numeric_list(opts["times"], "times")
        if len(values) == 0:
            raise CliUsageError("--times must contain at least one number")
        return values

    start = _as_scalar_numeric(opts["from"], "from")
    stop = _as_scalar_numeric(opts["to"], "to")
    step = _as_scalar_numeric(opts["by"], "by")
    if step <= 0:
        raise CliUsageError("--by must be positive")
    if stop < start:
        raise CliUsageError("--to must be greater than or equal to --from")
    return np.arange(start, stop + step / 2.0, step, dtype=np.float64)


def _regressor_grid(opts: dict[str, Any], onsets: np.ndarray) -> np.ndarray:
    if opts.get("blocklens") is not None or opts.get("tr") is not None:
        _require_options(opts, ("blocklens", "tr"))
        return _sampling_frame_from_options(opts).sample_times(global_time=True)
    opts.setdefault("from", "0")
    if opts.get("to") is None:
        max_onset = float(np.max(onsets)) if onsets.size else 0.0
        opts["to"] = str(max_onset + _as_scalar_numeric(opts["span"], "span"))
    return _grid_from_options(opts)


def _sampling_frame_from_options(opts: Mapping[str, Any]) -> SamplingFrame:
    blocklens = _parse_numeric_list(opts["blocklens"], "blocklens")
    tr = _parse_numeric_list(opts["tr"], "tr")
    start_time = (
        _parse_numeric_list(opts["start-time"], "start-time")
        if opts.get("start-time") is not None
        else tr / 2.0
    )
    return SamplingFrame(
        blocklens=blocklens.astype(int),
        tr=tr,
        start_time=start_time,
        precision=_as_scalar_numeric(opts["precision"], "precision"),
    )


def _events_for_regressor(opts: Mapping[str, Any]) -> pd.DataFrame:
    if opts.get("events") is not None:
        events = _read_events_table(str(opts["events"]))
        _require_columns(events, ("onset",))
        return pd.DataFrame(
            {
                "onset": events["onset"].astype(float),
                "duration": (
                    events["duration"].astype(float)
                    if "duration" in events
                    else np.zeros(len(events))
                ),
                "amplitude": (
                    events["amplitude"].astype(float)
                    if "amplitude" in events
                    else np.ones(len(events))
                ),
            }
        )
    if opts.get("onsets") is None:
        raise CliUsageError("Provide --onsets or --events")
    onsets = _parse_numeric_list(opts["onsets"], "onsets")
    duration = _recycle_or_error(
        _parse_numeric_list(opts["duration"], "duration"),
        len(onsets),
        "duration",
    )
    amplitude = _recycle_or_error(
        _parse_numeric_list(opts["amplitude"], "amplitude"),
        len(onsets),
        "amplitude",
    )
    return pd.DataFrame({"onset": onsets, "duration": duration, "amplitude": amplitude})


def _read_events_table(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise CliDomainError(f"File does not exist: {path}")
    sep = "\t" if Path(path).suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep)


def _get_cli_hrf(opts: Mapping[str, Any], *, allow_decorators: bool = True):
    kwargs = {
        "nbasis": _as_scalar_integer(opts["nbasis"], "nbasis"),
        "n_basis": _as_scalar_integer(opts["nbasis"], "nbasis"),
        "span": _as_scalar_numeric(opts["span"], "span"),
    }
    lag = _as_scalar_numeric(opts.get("lag", "0"), "lag") if allow_decorators else 0.0
    width = _as_scalar_numeric(opts.get("width", "0"), "width") if allow_decorators else 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Parameters .* ignored for pre-defined HRF",
            category=UserWarning,
        )
        return get_hrf(
            str(opts["hrf"]),
            lag=lag,
            width=width,
            summate=bool(opts.get("summate", True)),
            normalize=bool(opts.get("normalize")),
            **kwargs,
        )


def _parse_numeric_list(value: Any, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([value], dtype=np.float64)
    if not isinstance(value, str) or value == "":
        raise CliUsageError(f"--{name} must be a scalar or comma-separated list")
    pieces = [piece.strip() for piece in value.split(",")]
    if any(piece == "" for piece in pieces):
        raise CliUsageError(f"--{name} contains an empty value")
    try:
        return np.asarray([float(piece) for piece in pieces], dtype=np.float64)
    except ValueError as exc:
        raise CliUsageError(f"--{name} must contain only numeric values") from exc


def _as_scalar_numeric(value: Any, name: str) -> float:
    values = _parse_numeric_list(value, name)
    if len(values) != 1:
        raise CliUsageError(f"--{name} must be a single number")
    return float(values[0])


def _as_scalar_integer(value: Any, name: str) -> int:
    number = _as_scalar_numeric(value, name)
    if not np.isfinite(number) or number % 1 != 0:
        raise CliUsageError(f"--{name} must be a whole number")
    return int(number)


def _match_choice(value: str, choices: set[str], name: str) -> str:
    if value not in choices:
        raise CliUsageError(f"--{name} must be one of: {', '.join(sorted(choices))}")
    if value in {"loop", "Rconv"}:
        return "conv"
    return value


def _recycle_or_error(values: np.ndarray, n: int, name: str) -> np.ndarray:
    if len(values) == 1:
        return np.repeat(values, n)
    if len(values) == n:
        return values
    raise CliUsageError(f"`{name}` must have length 1 or {n}, not {len(values)}")


def _values_to_rows(
    grid: np.ndarray,
    values: np.ndarray,
    *,
    column_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    values = np.asarray(values)
    if values.ndim == 1:
        names = ["value"]
        matrix = values[:, np.newaxis]
    else:
        names = list(column_names) if column_names is not None else [
            f"basis{i}" for i in range(1, values.shape[1] + 1)
        ]
        matrix = values

    rows: list[dict[str, Any]] = []
    for i, time in enumerate(grid):
        row = {"time": _json_scalar(time)}
        for name, value in zip(names, matrix[i]):
            row[name] = _json_scalar(value)
        rows.append(row)
    return rows


def _design_column_names(rset) -> list[str]:
    names: list[str] = []
    for level, reg in zip(rset.levels, rset.regressors):
        if reg.nbasis == 1:
            names.append(level)
        else:
            names.extend(f"{level}_basis{i}" for i in range(1, reg.nbasis + 1))
    return names


def _write_cli_table(
    data: Sequence[Mapping[str, Any]],
    *,
    json_output: bool,
    output: str | None,
) -> None:
    if json_output:
        text = json.dumps(list(data), indent=2)
    else:
        rows = list(data)
        fieldnames = list(rows[0].keys()) if rows else []
        from io import StringIO

        handle = StringIO()
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_NONNUMERIC,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        text = handle.getvalue().rstrip("\n")

    if output is None:
        print(text)
    else:
        Path(output).write_text(text + "\n", encoding="utf-8")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    main()
