"""Thin BIDS Stats Model translator for fmrimod first-level workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class StatsModelTranslation:
    """Translated first-level model components."""

    event_model: object
    baseline_model: object
    column_names: list[str]
    contrast_vectors: dict[str, NDArray[np.float64]]
    node: Mapping[str, Any]
    caveats: tuple[str, ...] = ()


def load_stats_model(path: str | Path) -> dict[str, Any]:
    """Load a BIDS Stats Model JSON document."""

    with open(path) as f:
        return json.load(f)


def _run_node(model: Mapping[str, Any], level: str = "run") -> Mapping[str, Any]:
    for node in model.get("Nodes", []):
        if str(node.get("Level", "")).lower() == level.lower():
            return node
    raise ValueError(f"No BIDS Stats Model node with Level={level!r}")


def _convolved_factor(node: Mapping[str, Any]) -> str:
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
        return bases.pop()
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


def _contrast_vectors(
    node: Mapping[str, Any],
    column_names: Sequence[str],
) -> dict[str, NDArray[np.float64]]:
    vectors: dict[str, NDArray[np.float64]] = {}
    for contrast in node.get("Contrasts", []):
        if str(contrast.get("Test", "t")).lower() != "t":
            raise NotImplementedError("Only t contrasts are supported in v1")
        names = list(contrast.get("ConditionList", []))
        weights = list(contrast.get("Weights", []))
        if len(names) != len(weights):
            raise ValueError("Contrast ConditionList and Weights lengths differ")
        vector = np.zeros(len(column_names), dtype=np.float64)
        for condition, weight in zip(names, weights):
            col = _event_column_for_condition(column_names, str(condition))
            vector[column_names.index(col)] = float(weight)
        vectors[str(contrast.get("Name", "contrast"))] = vector
    return vectors


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
    - t contrasts over event conditions.
    """

    import fmrimod as fm

    node = _run_node(stats_model, level=level)
    factor = _convolved_factor(node)
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
    return StatsModelTranslation(
        event_model=event_model,
        baseline_model=baseline_model,
        column_names=column_names,
        contrast_vectors=_contrast_vectors(node, column_names),
        node=node,
        caveats=(
            "stats-model-v1 supports a constrained run-level Factor+Convolve subset",
        ),
    )
