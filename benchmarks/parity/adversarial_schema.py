"""Validation helpers for adversarial parity benchmark reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "adversarial-gauntlet/v1"

TOP_LEVEL_STATUSES = {"pass", "fail"}
CASE_STATUSES = {"pass", "fail", "changed", "boundary_observed"}
ENGINE_STATUSES = {"ok", "exception"}

TOP_LEVEL_KEYS = {"schema_version", "name", "status", "summary", "cases"}
CASE_KEYS = {
    "case_id",
    "purpose",
    "status",
    "expected_boundary",
    "design_shape",
    "design_rank",
    "design_condition",
    "fmrimod",
    "nilearn",
    "comparisons",
    "verdict",
}
ENGINE_KEYS = {
    "status",
    "rank",
    "df_residual",
    "is_full_rank",
    "ill_conditioned",
    "aliased_columns",
    "finite_effect_fraction",
    "finite_stat_fraction",
    "nan_se_fraction",
    "nan_p_fraction",
    "warning_messages",
    "undefined_t_policy",
    "exception_type",
    "exception_message",
}


def _require_mapping(value: object, path: str, errors: list[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    errors.append(f"{path} must be an object")
    return {}


def _require_keys(
    value: Mapping[str, Any],
    required: set[str],
    path: str,
    errors: list[str],
) -> None:
    missing = sorted(required - set(value))
    if missing:
        errors.append(f"{path} missing required keys: {', '.join(missing)}")


def _require_status(
    value: Mapping[str, Any],
    allowed: set[str],
    path: str,
    errors: list[str],
) -> None:
    status = value.get("status")
    if status not in allowed:
        errors.append(
            f"{path}.status must be one of {sorted(allowed)}, got {status!r}"
        )


def _is_number_or_none(value: object) -> bool:
    return value is None or isinstance(value, (int, float))


def _require_fraction(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[str],
) -> None:
    fraction = value.get(key)
    if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) <= 1.0:
        errors.append(f"{path}.{key} must be a number in [0, 1]")


def _validate_engine(value: object, path: str, errors: list[str]) -> None:
    engine = _require_mapping(value, path, errors)
    _require_keys(engine, ENGINE_KEYS, path, errors)
    _require_status(engine, ENGINE_STATUSES, path, errors)
    for key in ("finite_effect_fraction", "finite_stat_fraction", "nan_se_fraction", "nan_p_fraction"):
        _require_fraction(engine, key, path, errors)
    if not isinstance(engine.get("aliased_columns"), Sequence) or isinstance(
        engine.get("aliased_columns"),
        (str, bytes),
    ):
        errors.append(f"{path}.aliased_columns must be a sequence of strings")
    if not isinstance(engine.get("warning_messages"), Sequence) or isinstance(
        engine.get("warning_messages"),
        (str, bytes),
    ):
        errors.append(f"{path}.warning_messages must be a sequence of strings")
    for key in ("rank", "df_residual"):
        if not _is_number_or_none(engine.get(key)):
            errors.append(f"{path}.{key} must be numeric or null")


def _validate_case(value: object, index: int, errors: list[str]) -> None:
    path = f"cases[{index}]"
    case = _require_mapping(value, path, errors)
    _require_keys(case, CASE_KEYS, path, errors)
    _require_status(case, CASE_STATUSES, path, errors)
    shape = case.get("design_shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in shape)
    ):
        errors.append(f"{path}.design_shape must be two positive integers")
    if not isinstance(case.get("design_rank"), int) or case.get("design_rank") < 0:
        errors.append(f"{path}.design_rank must be a nonnegative integer")
    if not isinstance(case.get("comparisons"), Mapping):
        errors.append(f"{path}.comparisons must be an object")
    _validate_engine(case.get("fmrimod"), f"{path}.fmrimod", errors)
    _validate_engine(case.get("nilearn"), f"{path}.nilearn", errors)


def validate_adversarial_report(report: Mapping[str, Any]) -> None:
    """Validate an adversarial parity report.

    Raises
    ------
    ValueError
        If the report does not satisfy the shared adversarial benchmark
        schema.
    """

    errors: list[str] = []
    _require_keys(report, TOP_LEVEL_KEYS, "report", errors)
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"report.schema_version must be {SCHEMA_VERSION!r}, "
            f"got {report.get('schema_version')!r}"
        )
    _require_status(report, TOP_LEVEL_STATUSES, "report", errors)
    cases = report.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        errors.append("report.cases must be a nonempty sequence")
    else:
        for index, case in enumerate(cases):
            _validate_case(case, index, errors)
    if errors:
        raise ValueError("; ".join(errors))
