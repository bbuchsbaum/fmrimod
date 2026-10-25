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
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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


class _CliOption(NamedTuple):
    """One command-line option: ``metavar=None`` marks a boolean flag."""

    name: str
    metavar: str | None
    help: str
    default: object = None
    negatable: bool = False


class _CliCommand(NamedTuple):
    usage: str
    options: tuple[_CliOption, ...]
    notes: tuple[str, ...] = ()


_O = _CliOption
_JSON = _O("json", None, "Write JSON instead of CSV")
_OUTPUT = _O("output", "FILE", "Write output to FILE instead of stdout")
_HRF = _O("hrf", "NAME", "HRF name from `fmrihrf list`", "spmg1")
_NBASIS = _O("nbasis", "N", "Basis count for generated HRFs", "5")
_METHOD = _O("method", "METHOD", "conv, fft, direct, Rconv, or loop", "conv")
_START_TIME = _O("start-time", "LIST", "First acquisition offset(s); default TR / 2")

# The single option table: the parser, the defaults and the help text are
# all generated from it, so help cannot drift from what is accepted
# (fmrihrf 8af0c2d).
_CLI_COMMANDS: dict[str, _CliCommand] = {
    "list": _CliCommand(
        "Usage: fmrihrf list [--details] [--json] [--output FILE]",
        (
            _O("details", None, "Include descriptions from the HRF registry"),
            _JSON,
            _OUTPUT,
        ),
    ),
    "eval": _CliCommand(
        "Usage: fmrihrf eval [options]",
        (
            _HRF,
            _O("from", "SEC", "Grid start in seconds", "0"),
            _O("to", "SEC", "Grid end in seconds", "32"),
            _O("by", "SEC", "Grid step in seconds", "0.5"),
            _O("times", "LIST", "Comma-separated grid values; overrides from/to/by"),
            _NBASIS,
            _O("span", "SEC", "HRF span in seconds", "24"),
            _O("lag", "SEC", "Shift HRF by this many seconds", "0"),
            _O("width", "SEC", "Block HRF width in seconds", "0"),
            _O("amplitude", "VALUE", "Event amplitude for HRF evaluation", "1"),
            _O("duration", "SEC", "Event duration for HRF evaluation", "0"),
            _O("precision", "SEC", "Internal block precision", "0.2"),
            _O("normalize", None, "Normalize columns to unit peak"),
            _O(
                "summate",
                None,
                "Summate the block response (the default; --no-summate averages)",
                True,
                negatable=True,
            ),
            _JSON,
            _OUTPUT,
        ),
    ),
    "regressor": _CliCommand(
        "Usage: fmrihrf regressor [options]",
        (
            _O("onsets", "LIST", "Comma-separated onset times in seconds"),
            _O("events", "FILE", "CSV/TSV with onset[,duration,amplitude] columns"),
            _HRF,
            _O("blocklens", "LIST", "Scans per block for acquisition grid"),
            _O("tr", "LIST", "Repetition time(s) in seconds"),
            _START_TIME,
            _O("from", "SEC", "Grid start when no sampling frame is supplied", "0"),
            _O(
                "to",
                "SEC",
                "Grid end when no sampling frame is supplied; "
                "default last onset + span",
            ),
            _O("by", "SEC", "Grid step when no sampling frame is supplied", "1"),
            _O("duration", "LIST", "Event duration(s) when --events is not used", "0"),
            _O("amplitude", "LIST", "Event amplitude(s) when --events is not used", "1"),
            _NBASIS,
            _O("span", "SEC", "HRF/regressor span in seconds", "24"),
            _O("precision", "SEC", "Internal convolution precision", "0.33"),
            _METHOD,
            _O("normalize", None, "Normalize output columns to unit peak"),
            _JSON,
            _OUTPUT,
        ),
    ),
    "design": _CliCommand(
        "Usage: fmrihrf design --events FILE --blocklens LIST --tr LIST [options]",
        (
            _O("events", "FILE", "CSV/TSV events table"),
            _O("condition", "COL", "Condition column name", "condition"),
            _O("onset", "COL", "Onset column name", "onset"),
            _O("block", "COL", "Block column name", "block"),
            _O("duration", "COL", "Duration column name", "duration"),
            _O("amplitude", "COL", "Amplitude column name", "amplitude"),
            _O("blocklens", "LIST", "Scans per block"),
            _O("tr", "LIST", "Repetition time(s) in seconds"),
            _START_TIME,
            _HRF,
            _NBASIS,
            _O("span", "SEC", "HRF/regressor span in seconds", "24"),
            _O("precision", "SEC", "Internal convolution precision", "0.33"),
            _METHOD,
            _O(
                "sparse",
                None,
                "Return sparse design internally before writing dense output",
            ),
            _JSON,
            _OUTPUT,
        ),
        (
            "The events file must include onset, condition, and block columns.",
            "Duration and amplitude columns are used when present; a column named",
            "with --duration or --amplitude must exist.",
        ),
    ),
}
del _O


def _accepted_options(command: str) -> set[str]:
    """Every ``--name`` the parser accepts for ``command``."""
    names: set[str] = set()
    for opt in _CLI_COMMANDS[command].options:
        names.add(opt.name)
        if opt.negatable:
            names.add(f"no-{opt.name}")
    return names


def _parse_command(command: str, args: Sequence[str]) -> dict[str, object]:
    """Parse ``args`` for ``command`` and fill in the table's defaults."""
    options = _CLI_COMMANDS[command].options
    opts = _parse_cli_args(
        args,
        flags={o.name for o in options if o.metavar is None},
        values={o.name for o in options if o.metavar is not None},
        false_flags={o.name for o in options if o.negatable},
    )
    return _defaults(
        opts, {o.name: o.default for o in options if o.default is not None}
    )


def _command_help(spec: _CliCommand) -> list[str]:
    lines = [spec.usage, ""]
    if spec.notes:
        lines.extend([*spec.notes, ""])
    lines.append("Options:")
    entries: list[tuple[str, str]] = []
    for opt in spec.options:
        label = f"--{opt.name}" + (f" {opt.metavar}" if opt.metavar else "")
        text = opt.help
        if opt.metavar is not None and opt.default is not None:
            text = f"{text} [{opt.default}]"
        entries.append((label, text))
        if opt.negatable:
            entries.append((f"--no-{opt.name}", f"Turn off --{opt.name}"))
    width = max(len(label) for label, _ in entries) + 2
    lines.extend(f"  {label:<{width}}{text}" for label, text in entries)
    return lines


_MAIN_HELP = (
    "Usage: fmrihrf <command> [options]",
    "",
    "Commands:",
    "  list        List available HRFs and basis generators",
    "  eval        Evaluate an HRF or basis over a time grid",
    "  regressor   Build and evaluate one event regressor",
    "  design      Build a condition design matrix from an events table",
    "",
    "Run `fmrihrf help <command>` for command-specific options.",
)


def _cli_help(command: str | None = None) -> int:
    if command is None or command == "main":
        lines = list(_MAIN_HELP)
    elif command in _CLI_COMMANDS:
        lines = _command_help(_CLI_COMMANDS[command])
    else:
        raise CliUsageError(f"Unknown help topic: {command}")
    print("\n".join(lines))
    return 0


def _cli_list(args: list[str]) -> int:
    opts = _parse_command("list", args)
    rows = list_available_hrfs(details=True)
    _write_cli_table(cast(Any, rows), json_output=bool(opts.get("json")), output=opts.get("output"))
    return 0


def _cli_eval(args: list[str]) -> int:
    opts = _parse_command("eval", args)

    grid = _grid_from_options(opts)
    hrf = cast(Any, _get_cli_hrf(opts))
    values = hrf.evaluate(
        grid,
        duration=_as_scalar_numeric(opts["duration"], "duration"),
        precision=_as_scalar_numeric(opts["precision"], "precision"),
        summate=bool(opts["summate"]),
    )
    values = values * _as_scalar_numeric(opts["amplitude"], "amplitude")
    _write_cli_table(
        _values_to_rows(grid, values),
        json_output=bool(opts.get("json")),
        output=opts.get("output"),
    )
    return 0


def _cli_regressor(args: list[str]) -> int:
    opts = _parse_command("regressor", args)

    events = _events_for_regressor(opts)
    hrf = _get_cli_hrf(opts, allow_decorators=False)
    reg = regressor(
        onsets=events["onset"].to_numpy(float),
        hrf=cast(Any, hrf),
        duration=events["duration"].to_numpy(float),
        amplitude=events["amplitude"].to_numpy(float),
        span=_as_scalar_numeric(opts["span"], "span"),
    )
    grid = _regressor_grid(opts, events["onset"].to_numpy(float))
    values = cast(Any, reg).evaluate(
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
    opts = _parse_command("design", args)
    _require_options(opts, ("events", "blocklens", "tr"))

    events = _read_events_table(str(opts["events"]))
    # The default duration/amplitude columns are optional; renamed ones are
    # not (fmrihrf 742b1f1).
    renamed = [
        str(opts[name]) for name in ("duration", "amplitude") if str(opts[name]) != name
    ]
    _require_columns(
        events,
        dict.fromkeys(
            [str(opts["onset"]), str(opts["condition"]), str(opts["block"]), *renamed]
        ),
    )
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
        hrf=cast(Any, hrf),
        duration=duration,
        amplitude=amplitude,
        span=_as_scalar_numeric(opts["span"], "span"),
    )
    grid = sf.sample_times(global_time=True)
    design = cast(Any, regressor_design)(
        rset,
        grid,
        precision=_as_scalar_numeric(opts["precision"], "precision"),
        method=_match_choice(opts["method"], {"conv", "fft", "direct", "loop", "Rconv"}, "method"),
        sparse=bool(opts.get("sparse")),
    )
    if hasattr(design, "toarray"):
        design = design.toarray()
    rows = _values_to_rows(grid, np.asarray(design), column_names=rset.column_names)
    _write_cli_table(rows, json_output=bool(opts.get("json")), output=opts.get("output"))
    return 0


def _parse_cli_args(
    args: Sequence[str],
    *,
    flags: set[str] | None = None,
    values: set[str] | None = None,
    false_flags: set[str] | None = None,
) -> dict[str, object]:
    flags = flags or set()
    values = values or set()
    false_flags = false_flags or set()
    out: dict[str, object] = {}
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


def _defaults(opts: dict[str, object], defaults: Mapping[str, object]) -> dict[str, object]:
    for key, value in defaults.items():
        opts.setdefault(key, value)
    return opts


def _require_options(opts: Mapping[str, object], names: Iterable[str]) -> None:
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


def _grid_from_options(opts: Mapping[str, object]) -> NDArray[np.float64]:
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


def _regressor_grid(
    opts: dict[str, object], onsets: NDArray[np.float64]
) -> NDArray[np.float64]:
    if opts.get("blocklens") is not None or opts.get("tr") is not None:
        _require_options(opts, ("blocklens", "tr"))
        return _sampling_frame_from_options(opts).sample_times(global_time=True)
    opts.setdefault("from", "0")
    if opts.get("to") is None:
        max_onset = float(np.max(onsets)) if onsets.size else 0.0
        opts["to"] = str(max_onset + _as_scalar_numeric(opts["span"], "span"))
    return _grid_from_options(opts)


def _sampling_frame_from_options(opts: Mapping[str, object]) -> SamplingFrame:
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


def _events_for_regressor(opts: Mapping[str, object]) -> pd.DataFrame:
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


def _get_cli_hrf(opts: Mapping[str, object], *, allow_decorators: bool = True) -> object:
    kwargs = _cli_hrf_constructor_kwargs(opts)
    lag = _as_scalar_numeric(opts.get("lag", "0"), "lag") if allow_decorators else 0.0
    width = _as_scalar_numeric(opts.get("width", "0"), "width") if allow_decorators else 0.0
    return get_hrf(
        str(opts["hrf"]),
        lag=lag,
        width=width,
        summate=bool(opts.get("summate", True)),
        normalize=bool(opts.get("normalize")),
        **cast("dict[str, Any]", kwargs),
    )


def _cli_hrf_constructor_kwargs(opts: Mapping[str, object]) -> dict[str, object]:
    name = str(opts["hrf"]).strip().lower()
    nbasis = _as_scalar_integer(opts["nbasis"], "nbasis")
    span = _as_scalar_numeric(opts["span"], "span")

    if name in {"bspline", "hrf_bspline", "fir", "hrf_fir", "fourier", "hrf_fourier", "daguerre", "hrf_daguerre"}:
        return {"n_basis": nbasis, "span": span}
    if name in {"tent", "hrf_tent", "sine"}:
        return {"nbasis": nbasis, "span": span}
    return {}


def _parse_numeric_list(value: object, name: str) -> NDArray[np.float64]:
    if isinstance(value, np.ndarray):
        return cast(NDArray[np.float64], value.astype(np.float64))
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


def _as_scalar_numeric(value: object, name: str) -> float:
    values = _parse_numeric_list(value, name)
    if len(values) != 1:
        raise CliUsageError(f"--{name} must be a single number")
    return float(values[0])


def _as_scalar_integer(value: object, name: str) -> int:
    number = _as_scalar_numeric(value, name)
    if not np.isfinite(number) or number % 1 != 0:
        raise CliUsageError(f"--{name} must be a whole number")
    return int(number)


def _match_choice(value: object, choices: set[str], name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise CliUsageError(f"--{name} must be one of: {', '.join(sorted(choices))}")
    if value in {"loop", "Rconv"}:
        return "conv"
    return value


def _recycle_or_error(
    values: NDArray[np.float64], n: int, name: str
) -> NDArray[np.float64]:
    if len(values) == 1:
        return cast(NDArray[np.float64], np.repeat(values, n))
    if len(values) == n:
        return values
    raise CliUsageError(f"`{name}` must have length 1 or {n}, not {len(values)}")


def _values_to_rows(
    grid: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    column_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    values = np.asarray(values)
    if values.ndim == 1:
        names = ["value"]
        matrix = values[:, np.newaxis]
    else:
        names = list(column_names) if column_names is not None else [
            f"basis{i}" for i in range(1, values.shape[1] + 1)
        ]
        matrix = values

    rows: list[dict[str, object]] = []
    for i, time in enumerate(grid):
        row = {"time": _json_scalar(time)}
        for name, value in zip(names, matrix[i]):
            row[name] = _json_scalar(value)
        rows.append(row)
    return rows


def _write_cli_table(
    data: Sequence[Mapping[str, object]],
    *,
    json_output: bool,
    output: object,
) -> None:
    output_path: str | None
    if output is None:
        output_path = None
    elif isinstance(output, str):
        output_path = output
    else:
        raise CliUsageError("--output must be a string path")
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

    if output_path is None:
        print(text)
    else:
        Path(output_path).write_text(text + "\n", encoding="utf-8")


def _json_scalar(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    main()
