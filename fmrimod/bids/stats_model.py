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
    condition_weights: NDArray[np.float64] | None = None

    def resolve(self, column_names: Sequence[str]) -> NDArray[np.float64]:
        """Resolve this contrast against a realised design-column order."""
        weights = self.condition_weights
        if weights is None:
            full = np.asarray(self.weights, dtype=np.float64)
            n_columns = len(column_names)
            if full.ndim == 1 and full.shape[0] == n_columns:
                return full
            if full.ndim == 2 and full.shape[1] == n_columns:
                return full
            raise ValueError(
                "StatsModelContrast cannot resolve without condition_weights "
                "when stored weights do not match the target column count"
            )
        return _realised_contrast(column_names, self.conditions, weights)

    def apply(self, fit: object) -> object:
        """Compute this contrast on an ``fmri_lm`` result."""
        from fmrimod.glm.contrasts import (
            ContrastIntent,
            basis_label,
            design_id,
            provenance_id,
            weights_payload,
        )

        columns = fit.design_columns()  # type: ignore[attr-defined]
        weights = self.resolve(columns.names)
        result = fit.contrast(weights, name=self.name)  # type: ignore[attr-defined]
        term, levels = _condition_term_and_levels(self.conditions)
        result.intent = ContrastIntent(
            kind="bids_stats_model",
            name=self.name,
            term=term,
            levels=levels,
            rows=int(np.atleast_2d(weights).shape[0]),
            basis_label=basis_label(fit),
            weights=weights_payload(weights),
            design_id=design_id(fit),
            provenance_id=provenance_id(fit),
        )
        return result


@dataclass(frozen=True)
class StatsModelTranslation:
    """Translated first-level model components."""

    event_model: object
    baseline_model: object
    event_table: pd.DataFrame
    column_names: list[str]
    contrast_vectors: dict[str, NDArray[np.float64]]
    node: Mapping[str, Any]
    caveats: tuple[str, ...] = ()
    model_spec: Spec | None = None
    contrast_specs: dict[str, StatsModelContrast] = field(default_factory=dict)
    contrast_matrices: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def fit(self, dataset: object) -> object:
        """Fit the translated typed model through the public fmrimod seam."""
        if self.model_spec is None:
            raise ValueError("StatsModelTranslation has no typed model_spec")
        import fmrimod as fm

        replace_events = getattr(dataset, "with_event_table", None)
        if callable(replace_events):
            translated_dataset = replace_events(self.event_table)
        else:
            data_source = getattr(dataset, "_source", dataset)
            translated_dataset = fm.fmri_dataset(
                data_source=data_source,
                events=self.event_table,
                censor=getattr(dataset, "censor", None),
            )
        return fm.fmri_lm(self.model_spec, translated_dataset)

    def contrast(self, fit: object, name: str) -> object:
        """Compute one translated contrast on a fitted public-seam model."""
        return self.contrast_specs[name].apply(fit)

    def compute_contrasts(self, fit: object) -> dict[str, object]:
        """Compute every translated contrast on a fitted public-seam model."""
        return {
            name: spec.apply(fit)
            for name, spec in self.contrast_specs.items()
        }


@dataclass(frozen=True)
class _ConvolvedModel:
    factor: str
    hrf_model: str
    modulators: tuple[str, ...] = ()


def load_stats_model(path: str | Path) -> dict[str, Any]:
    """Load a BIDS Stats Model JSON document."""

    with open(path) as f:
        return json.load(f)


def _run_node(model: Mapping[str, Any], level: str = "run") -> Mapping[str, Any]:
    for node in model.get("Nodes", []):
        if str(node.get("Level", "")).lower() == level.lower():
            return node
    raise ValueError(f"No BIDS Stats Model node with Level={level!r}")


def _transform_instructions(node: Mapping[str, object]) -> list[Mapping[str, object]]:
    transforms = node.get("Transformations", [])
    if isinstance(transforms, Mapping):
        transforms = transforms.get("Instructions", [])
    if not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
        raise TypeError("Transformations must be a sequence or Instructions mapping")
    out: list[Mapping[str, object]] = []
    for transform in transforms:
        if not isinstance(transform, Mapping):
            raise TypeError("Transformation instructions must be mappings")
        out.append(transform)
    return out


def _input_list(transform: Mapping[str, object]) -> list[str]:
    raw = transform.get("Input", [])
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw]
    raise TypeError("Transformation Input must be a string or sequence")


def _output_name(transform: Mapping[str, object]) -> str | None:
    raw = transform.get("Output")
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Sequence) and len(raw) == 1:
        return str(raw[0])
    raise TypeError("Only single-output transformations are supported")


def _convolved_model(node: Mapping[str, object]) -> _ConvolvedModel:
    """Return the supported run-level convolved event model."""

    transforms = _transform_instructions(node)
    factor: str | None = None
    hrf_model = "spm"
    convolved_inputs: set[str] = set()
    for transform in transforms:
        if str(transform.get("Name", "")).lower() != "convolve":
            continue
        inputs = _input_list(transform)
        if not inputs:
            raise ValueError("Convolve transformation must declare Input")
        bases = {str(item).split(".")[0] for item in inputs}
        if len(bases) != 1:
            raise NotImplementedError(
                "Only single-factor Convolve transformations are supported"
            )
        factor = bases.pop()
        hrf_model = str(transform.get("Model", "spm")).lower()
        convolved_inputs = set(inputs)
        break
    if factor is None:
        raise NotImplementedError("Stats model node does not contain Convolve")

    modulators: list[str] = []
    for transform in transforms:
        if str(transform.get("Name", "")).lower() != "product":
            continue
        inputs = _input_list(transform)
        factor_inputs = {
            item
            for item in inputs
            if item == factor or item.startswith(f"{factor}.")
        }
        if not factor_inputs:
            continue
        if convolved_inputs and not factor_inputs.issubset(convolved_inputs):
            continue
        continuous = [
            item
            for item in inputs
            if item not in factor_inputs and "." not in item
        ]
        if len(continuous) != 1:
            raise NotImplementedError(
                "Product parametric modulators must combine one continuous "
                "event column with the convolved factor levels"
            )
        modulator = continuous[0]
        if modulator not in modulators:
            modulators.append(modulator)
    return _ConvolvedModel(
        factor=factor,
        hrf_model=hrf_model,
        modulators=tuple(modulators),
    )


def _flag(transform: Mapping[str, object], *names: str, default: bool) -> bool:
    for name in names:
        if name in transform:
            return bool(transform[name])
    return default


def _numeric_param(
    transform: Mapping[str, object],
    *names: str,
    default: float,
) -> float:
    for name in names:
        if name in transform:
            return float(transform[name])
    return default


def _apply_scale_transform(
    out: pd.DataFrame,
    transform: Mapping[str, object],
) -> None:
    inputs = _input_list(transform)
    if len(inputs) != 1:
        raise NotImplementedError("Scale currently supports one input column")
    source = inputs[0]
    if source not in out.columns:
        raise KeyError(f"Scale requested missing event column: {source!r}")
    output = _output_name(transform) or source
    values = out[source].to_numpy(dtype=np.float64)
    scaled = values.copy()
    if _flag(transform, "Demean", "demean", "Center", "center", default=True):
        scaled = scaled - float(np.nanmean(scaled))
    if _flag(transform, "Rescale", "rescale", "Scale", "scale", default=True):
        sd = float(np.nanstd(scaled, ddof=0))
        if sd == 0.0 or not np.isfinite(sd):
            raise ValueError(f"Cannot scale constant event column {source!r}")
        scaled = scaled / sd
    out[output] = scaled


def _apply_threshold_transform(
    out: pd.DataFrame,
    transform: Mapping[str, object],
) -> None:
    inputs = _input_list(transform)
    if len(inputs) != 1:
        raise NotImplementedError("Threshold currently supports one input column")
    source = inputs[0]
    if source not in out.columns:
        raise KeyError(f"Threshold requested missing event column: {source!r}")
    output = _output_name(transform) or source
    threshold = _numeric_param(transform, "Threshold", "threshold", default=0.0)
    binarize = _flag(transform, "Binarize", "binarize", default=False)
    above = _flag(transform, "Above", "above", default=True)
    signed = _flag(transform, "Signed", "signed", default=True)

    data = pd.to_numeric(out[source], errors="raise").astype(float)
    if not signed:
        threshold = abs(threshold)
        data = data.abs()
    keep = data >= threshold if above else data <= threshold
    result = data.copy()
    result.loc[~keep] = 0.0
    if binarize:
        result.loc[keep] = 1.0
    out[output] = result


def _apply_or_transform(
    out: pd.DataFrame,
    transform: Mapping[str, object],
) -> None:
    inputs = _input_list(transform)
    if len(inputs) < 2:
        raise NotImplementedError("Or requires at least two input columns")
    missing = [source for source in inputs if source not in out.columns]
    if missing:
        raise KeyError(f"Or requested missing event columns: {missing}")
    output = _output_name(transform)
    if output is None:
        raise ValueError("Or transformations must declare Output")
    out[output] = out[inputs].astype(bool).any(axis=1).astype(int)


def _apply_event_table_transforms(
    node: Mapping[str, object],
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Realise supported event-table transforms in BIDS instruction order."""

    out = events.copy()
    for transform in _transform_instructions(node):
        name = str(transform.get("Name", "")).lower()
        if name == "scale":
            _apply_scale_transform(out, transform)
        elif name == "threshold":
            _apply_threshold_transform(out, transform)
        elif name in {"or", "or_"}:
            _apply_or_transform(out, transform)
    return out


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


def _condition_term_and_levels(
    conditions: Sequence[str],
) -> tuple[str | None, tuple[str, ...]]:
    parts = [str(condition).split(".", maxsplit=1) for condition in conditions]
    if not parts:
        return None, ()
    if any(len(part) != 2 for part in parts):
        return None, tuple(str(condition) for condition in conditions)
    terms = {part[0] for part in parts}
    if len(terms) != 1:
        return None, tuple(str(condition) for condition in conditions)
    return terms.pop(), tuple(part[1] for part in parts)


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
            condition_weights=weights,
        )
    return specs


def _model_spec(
    *,
    factor: str,
    hrf_model: str,
    modulators: Sequence[str],
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
        modulators=modulators,
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
    - Scale, Threshold, Or, and Product for event-table parametric modulators;
    - Model.X strings that are either event terms or confound columns;
    - integer ``1`` for a global intercept;
    - t and F contrasts over event conditions.

    The returned object keeps the legacy ``event_model``/``baseline_model`` and
    ``contrast_vectors`` fields for existing parity workflows, while also
    exposing ``model_spec`` and ``contrast_specs`` as typed public artifacts.
    """

    import fmrimod as fm
    from fmrimod.spec import compile_events

    node = _run_node(stats_model, level=level)
    convolved = _convolved_model(node)
    transformed_events = _apply_event_table_transforms(node, events)
    intercept, model_terms = _baseline_terms(node)
    event_levels = set(transformed_events[convolved.factor].astype(str))
    nuisance_cols = [
        term
        for term in model_terms
        if "." not in term
        and term not in event_levels
        and term not in convolved.modulators
    ]
    missing_modulators = [
        col for col in convolved.modulators if col not in transformed_events.columns
    ]
    if missing_modulators:
        raise KeyError(
            f"Stats model requested missing modulator columns: {missing_modulators}"
        )

    missing = [
        col for col in nuisance_cols if confounds is None or col not in confounds
    ]
    if missing:
        raise KeyError(f"Stats model requested missing confound columns: {missing}")

    model_spec = _model_spec(
        factor=convolved.factor,
        hrf_model=convolved.hrf_model,
        modulators=convolved.modulators,
        duration_col=duration_col,
        nuisance_cols=nuisance_cols,
        intercept=intercept,
        confounds=confounds,
    )
    event_model = compile_events(
        model_spec,
        transformed_events,
        sampling_frame=sampling_frame,
        block=block if block in transformed_events.columns else None,
        durations=duration_col,
        precision=None,
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
        event_table=transformed_events,
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
        model_spec=model_spec,
        contrast_specs=contrast_specs,
        contrast_matrices={
            name: spec.weights
            for name, spec in contrast_specs.items()
            if spec.test == "F"
        },
    )
