"""Thin BIDS Stats Model translator for fmrimod first-level workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fmrimod.spec import Spec


@dataclass(frozen=True)
class StatsModelContrast:
    """Typed BIDS contrast request plus its realised weight array."""

    name: str
    test: Literal["t", "F"]
    conditions: tuple[str, ...]
    weights: NDArray[np.float64]


@dataclass(frozen=True)
class StatsModelTranslation:
    """Translated first-level model components."""

    event_model: object
    baseline_model: object
    column_names: list[str]
    contrast_vectors: dict[str, NDArray[np.float64]]
    node: Mapping[str, Any]
    caveats: tuple[str, ...] = ()
    model_spec: Spec | None = None
    contrast_specs: dict[str, StatsModelContrast] = field(default_factory=dict)
    contrast_matrices: dict[str, NDArray[np.float64]] = field(default_factory=dict)


def load_stats_model(path: str | Path) -> dict[str, Any]:
    """Load a BIDS Stats Model JSON document."""

    with open(path) as f:
        return json.load(f)


def _run_node(model: Mapping[str, Any], level: str = "run") -> Mapping[str, Any]:
    for node in model.get("Nodes", []):
        if str(node.get("Level", "")).lower() == level.lower():
            return node
    raise ValueError(f"No BIDS Stats Model node with Level={level!r}")


def _convolved_factor(node: Mapping[str, Any]) -> tuple[str, str]:
    """Return the factor column used by a supported Convolve transformation."""

    transforms = node.get("Transformations", [])
    if isinstance(transforms, Mapping):
        transforms = transforms.get("Instructions", [])
    for transform in transforms:
        if str(transform.get("Name", "")).lower() != "convolve":
            continue
        inputs = transform.get("Input", [])
        if not inputs:
            raise ValueError("Convolve transformation must declare Input")
        bases = {str(item).split(".")[0] for item in inputs}
        if len(bases) != 1:
            raise NotImplementedError(
                "Only single-factor Convolve transformations are supported"
            )
        return bases.pop(), str(transform.get("Model", "spm")).lower()
    raise NotImplementedError("Stats model node does not contain Convolve")


def _baseline_terms(node: Mapping[str, Any]) -> tuple[bool, list[str]]:
    model = node.get("Model", {})
    x_terms = model.get("X", [])
    intercept = any(item == 1 or str(item).lower() == "intercept" for item in x_terms)
    nuisance = [str(item) for item in x_terms if isinstance(item, str)]
    return intercept, nuisance


def _event_column_for_condition(column_names: Sequence[str], condition: str) -> str:
    condition = str(condition)
    candidates = [
        condition,
        condition.replace(".", "_"),
        condition.replace(".", "_trial_type."),
    ]
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    suffix = "." + condition.split(".")[-1]
    matches = [name for name in column_names if name.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Could not align BIDS condition {condition!r} to fmrimod columns")


def _realised_contrast(
    column_names: Sequence[str],
    names: Sequence[str],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    if weights.ndim == 1:
        vector = np.zeros(len(column_names), dtype=np.float64)
        for condition, weight in zip(names, weights):
            col = _event_column_for_condition(column_names, str(condition))
            vector[column_names.index(col)] = float(weight)
        return vector
    matrix = np.zeros((weights.shape[0], len(column_names)), dtype=np.float64)
    for row_ix, row in enumerate(weights):
        for condition, weight in zip(names, row):
            col = _event_column_for_condition(column_names, str(condition))
            matrix[row_ix, column_names.index(col)] = float(weight)
    return matrix


def _contrast_weight_array(
    contrast: Mapping[str, object],
    *,
    test: Literal["t", "F"],
    n_conditions: int,
) -> NDArray[np.float64]:
    raw = contrast.get("Weights", None)
    if raw is None:
        if test == "F":
            return np.eye(n_conditions, dtype=np.float64)
        raise ValueError("t contrasts must declare Weights")
    weights = np.asarray(raw, dtype=np.float64)
    if test == "t":
        if weights.ndim != 1 or weights.shape[0] != n_conditions:
            raise ValueError(
                "t contrast Weights must be a 1-D array matching ConditionList"
            )
        return weights
    if weights.ndim == 2 and weights.shape[1] == n_conditions:
        return weights
    raise ValueError(
        "F contrast Weights must be a 2-D matrix with one column per condition"
    )


def _contrast_specs(
    node: Mapping[str, object],
    column_names: Sequence[str],
) -> dict[str, StatsModelContrast]:
    specs: dict[str, StatsModelContrast] = {}
    for contrast in node.get("Contrasts", []):
        raw_test = str(contrast.get("Test", "t")).lower()
        if raw_test == "t":
            test: Literal["t", "F"] = "t"
        elif raw_test == "f":
            test = "F"
        else:
            raise NotImplementedError(f"Unsupported contrast test: {raw_test!r}")
        names = list(contrast.get("ConditionList", []))
        weights = _contrast_weight_array(
            contrast,
            test=test,
            n_conditions=len(names),
        )
        realised = _realised_contrast(column_names, names, weights)
        name = str(contrast.get("Name", "contrast"))
        specs[name] = StatsModelContrast(
            name=name,
            test=test,
            conditions=tuple(str(item) for item in names),
            weights=realised,
        )
    return specs


def _model_spec(
    *,
    factor: str,
    hrf_model: str,
    duration_col: str,
    nuisance_cols: Sequence[str],
    intercept: bool,
    confounds: pd.DataFrame | None,
) -> Spec:
    from fmrimod.spec import confounds as confounds_term
    from fmrimod.spec import hrf
    from fmrimod.spec import intercept as intercept_term

    spec = hrf(
        factor,
        basis=hrf_model,
        durations=duration_col,
    ) + intercept_term(per="global" if intercept else "none")
    if nuisance_cols:
        spec = spec + confounds_term(*nuisance_cols, source=confounds)
    return spec


def translate_run_node(
    stats_model: Mapping[str, Any],
    *,
    events: pd.DataFrame,
    sampling_frame: object,
    confounds: pd.DataFrame | None = None,
    level: str = "run",
    block: str = "run",
    duration_col: str = "duration",
) -> StatsModelTranslation:
    """Translate a supported run-level BIDS Stats Model node.

    Supported v1 surface:
    - one run-level node;
    - Factor + Convolve over one categorical event column;
    - Model.X strings that are either event terms or confound columns;
    - integer ``1`` for a global intercept;
    - t and F contrasts over event conditions.

    The returned object keeps the legacy ``event_model``/``baseline_model`` and
    ``contrast_vectors`` fields for existing parity workflows, while also
    exposing ``model_spec`` and ``contrast_specs`` as typed public artifacts.
    """

    import fmrimod as fm

    node = _run_node(stats_model, level=level)
    factor, hrf_model = _convolved_factor(node)
    intercept, model_terms = _baseline_terms(node)
    event_levels = set(events[factor].astype(str))
    nuisance_cols = [
        term for term in model_terms if "." not in term and term not in event_levels
    ]

    missing = [col for col in nuisance_cols if confounds is None or col not in confounds]
    if missing:
        raise KeyError(f"Stats model requested missing confound columns: {missing}")

    event_model = fm.event_model(
        f"hrf({factor})",
        data=events,
        sampling_frame=sampling_frame,
        block=block if block in events.columns else None,
        durations=duration_col,
    )
    nuisance_list = None
    if nuisance_cols:
        assert confounds is not None
        nuisance_list = [confounds[nuisance_cols].to_numpy(dtype=np.float64)]
    baseline_model = fm.baseline_model(
        basis="constant",
        sframe=sampling_frame,
        intercept="global" if intercept else "none",
        nuisance_list=nuisance_list,
    )
    column_names = list(event_model.column_names) + list(baseline_model.column_names)
    contrast_specs = _contrast_specs(node, column_names)
    return StatsModelTranslation(
        event_model=event_model,
        baseline_model=baseline_model,
        column_names=column_names,
        contrast_vectors={
            name: spec.weights
            for name, spec in contrast_specs.items()
            if spec.test == "t"
        },
        node=node,
        caveats=(
            "stats-model-v1 supports a constrained run-level Factor+Convolve subset",
        ),
        model_spec=_model_spec(
            factor=factor,
            hrf_model=hrf_model,
            duration_col=duration_col,
            nuisance_cols=nuisance_cols,
            intercept=intercept,
            confounds=confounds,
        ),
        contrast_specs=contrast_specs,
        contrast_matrices={
            name: spec.weights
            for name, spec in contrast_specs.items()
            if spec.test == "F"
        },
    )
