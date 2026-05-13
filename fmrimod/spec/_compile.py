"""Lower a typed :class:`Spec` to ``(EventModel, BaselineModel)`` artefacts.

The string formula path (``event_model("hrf(trial_type)")``) remains legacy
syntax sugar. Typed :class:`Spec` terms lower directly into explicit
``EventModel`` terms here so ``fmri_lm(spec, dataset)`` does not depend on the
string formula parser.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from ..hrf.core import HRF
from ..hrf.registry import get_hrf
from .terms import Confounds, Drift, HrfTerm, Intercept, Spec, Term

BaselineBasis = Literal["constant", "poly", "bs", "ns"]
BaselineIntercept = Literal["runwise", "global", "none"]


def _resolve_hrf(term: HrfTerm) -> HRF | str:
    """Resolve the HRF value carried by a typed :class:`HrfTerm`."""
    if term.norm is None:
        return term.hrf

    from ..hrf.normalization import normalize

    base = term.hrf
    if isinstance(base, str):
        base = get_hrf(base)
    return normalize(base, term.norm)


def _hrf_term_to_event_model_term(term: HrfTerm) -> Any:
    """Translate one typed HRF term to EventModel's explicit Term object.

    This builds only the main-effect term; ``term.modulators`` is expanded
    into separate parametric-modulator terms by :func:`_lower_hrf_term`,
    which is what :func:`compile_events` calls.
    """
    from ..formula.base import Term as EventModelTerm

    if not term.variables:
        raise ValueError("HrfTerm has no variables to lower")

    events: str | list[str]
    if len(term.variables) == 1:
        events = term.variables[0]
    else:
        events = list(term.variables)

    lowered = EventModelTerm(
        events,
        hrf=_resolve_hrf(term),
        name=term.id,
        normalize=term.normalize,
        summate=term.summate,
    )
    if term.contrasts:
        lowered._kwargs["contrasts"] = term.contrasts
    if term.durations is not None:
        lowered._kwargs["durations"] = term.durations
    if term.subset is not None:
        lowered._kwargs["subset"] = term.subset
    if term.prefix is not None:
        lowered._kwargs["prefix"] = term.prefix
    if term.lag:
        lowered._kwargs["lag"] = term.lag
    return lowered


def _lower_hrf_term(term: HrfTerm) -> list[Any]:
    """Lower one typed HrfTerm into one or more legacy EventModelTerms.

    With no modulators, this is a single main-effect term. With
    ``modulators=("rt", ...)`` it expands to one main-effect term plus one
    interaction term per modulator, mirroring the legacy
    ``trial_type:hrf(...) + trial_type:rt:hrf(...)`` formula path.

    The main term keeps any declared ``contrasts``; parametric terms inherit
    the main term's HRF, durations, lag, subset, prefix, normalize and
    summate but never the contrasts (those name level-combinations of the
    categorical variables, not the modulator regressors).
    """
    from ..formula.base import Term as EventModelTerm

    main = _hrf_term_to_event_model_term(term)
    if not term.modulators:
        return [main]

    base_id = term.id or main.name
    lowered: list[Any] = [main]
    for modulator in term.modulators:
        param_events = list(term.variables) + [modulator]
        param_name = f"{base_id}:{modulator}" if base_id else None
        param = EventModelTerm(
            param_events,
            hrf=_resolve_hrf(term),
            name=param_name,
            normalize=term.normalize,
            summate=term.summate,
        )
        if term.durations is not None:
            param._kwargs["durations"] = term.durations
        if term.subset is not None:
            param._kwargs["subset"] = term.subset
        if term.prefix is not None:
            param._kwargs["prefix"] = term.prefix
        if term.lag:
            param._kwargs["lag"] = term.lag
        lowered.append(param)
    return lowered


def _event_model_term_to_hrf_term(term: Any) -> HrfTerm:
    """Convert a legacy formula Term to the typed HrfTerm representation."""
    events = getattr(term, "events", None)
    hrf_value = getattr(term, "hrf", None)
    if not events or hrf_value is None:
        raise TypeError("Only HRF-bearing event terms can lower to HrfTerm")

    extra = getattr(term, "_kwargs", {}) or {}
    contrasts = extra.get("contrasts", ())
    modulators = extra.get("modulators", ())
    if contrasts is None:
        contrasts = ()
    if modulators is None:
        modulators = ()

    return HrfTerm(
        variables=tuple(str(event) for event in events),
        hrf=hrf_value,
        contrasts=tuple(contrasts),
        modulators=tuple(modulators),
        durations=extra.get("durations"),
        lag=float(extra.get("lag", 0.0)),
        subset=extra.get("subset"),
        prefix=extra.get("prefix"),
        id=getattr(term, "name", None),
        normalize=bool(extra.get("normalize", getattr(term, "normalize", False))),
        summate=bool(extra.get("summate", getattr(term, "summate", True))),
    )


def legacy_formula_to_spec(formula: str | Sequence[object]) -> Spec:
    """Convert supported legacy formula inputs into typed :class:`Spec`.

    This adapter intentionally supports the HRF-bearing formula subset first.
    Plain unconvolved event terms, basis transforms, trialwise terms, and custom
    builders continue through the legacy event-model path until they have a
    typed representation with equivalent semantics.
    """
    from ..formula.base import Term as EventModelTerm

    terms: list[object]
    if isinstance(formula, str):
        from ..formula.parser import parse_formula

        terms = list(parse_formula(formula, for_event_model=True))
    elif isinstance(formula, Sequence):
        terms = list(formula)
    else:
        raise TypeError(
            "legacy_formula_to_spec expects a formula string or sequence of terms"
        )

    out = Spec()
    for term in terms:
        if isinstance(term, str):
            raise TypeError("String items in formula sequences are not typed HRF terms")
        if not isinstance(term, EventModelTerm):
            raise TypeError(
                f"Unsupported legacy formula item: {type(term).__name__}"
            )
        out = out + _event_model_term_to_hrf_term(term)
    return out


def compile_events(
    spec: Spec,
    data: pd.DataFrame,
    sampling_frame: Any,
    *,
    block: Any,
    durations: Any,
    precision: float | None,
) -> Any:
    """Build an EventModel from the spec's event terms.

    Returns ``None`` if the spec has no event terms (baseline-only design).
    """
    from ..design.event_model import event_model as build_event_model

    event_terms = spec.events
    if not event_terms:
        return None

    lowered_terms: list[Any] = []
    for term in event_terms:
        if not isinstance(term, HrfTerm):
            raise TypeError(
                f"Spec.events may only contain HrfTerm objects; got "
                f"{type(term).__name__}"
            )
        lowered_terms.extend(_lower_hrf_term(term))

    kwargs: dict[str, Any] = dict(
        data=data,
        sampling_frame=sampling_frame,
        block=block,
        durations=durations,
    )
    if precision is not None:
        kwargs["precision"] = precision

    return build_event_model(lowered_terms, **kwargs)


def compile_baseline(
    spec: Spec,
    sampling_frame: Any,
) -> Any:
    """Build a BaselineModel from the spec's baseline terms.

    If no baseline terms were declared, defaults to a runwise constant
    intercept — the same default the standalone :func:`fmri_lm` path uses.
    """
    from ..baseline.baseline_model import baseline_model as build_baseline

    drifts = [t for t in spec.baseline if isinstance(t, Drift)]
    intercepts = [t for t in spec.baseline if isinstance(t, Intercept)]
    confounds = [t for t in spec.baseline if isinstance(t, Confounds)]

    if len(drifts) > 1:
        raise ValueError("Spec.baseline may contain at most one Drift term")
    if len(intercepts) > 1:
        raise ValueError("Spec.baseline may contain at most one Intercept term")

    if not drifts and not intercepts and not confounds:
        return build_baseline(basis="constant", sframe=sampling_frame, intercept="runwise")

    drift_term = drifts[0] if drifts else None
    intercept_term = intercepts[0] if intercepts else None

    basis = cast(BaselineBasis, drift_term.basis if drift_term else "constant")
    degree: int = drift_term.degree if drift_term else 1
    intercept_kind: str = intercept_term.per if intercept_term else "runwise"
    # Map our "run" convention to the legacy "runwise" label.
    if intercept_kind == "run":
        intercept_kind = "runwise"
    intercept = cast(BaselineIntercept, intercept_kind)

    nuisance_list = None
    if confounds:
        # Pull confound matrices per run from each Confounds.source.
        per_run = []
        for c in confounds:
            if c.source is None:
                raise NotImplementedError(
                    "Confounds without an explicit `source` DataFrame are not yet "
                    "supported; supply source= or pre-join into the event table."
                )
            per_run.append(np.asarray(c.source[list(c.columns)].to_numpy(), dtype=np.float64))
        nuisance_list = per_run

    return build_baseline(
        basis=basis,
        degree=degree,
        sframe=sampling_frame,
        intercept=intercept,
        nuisance_list=nuisance_list,
    )


def compile(
    spec: Spec | Term,
    data: pd.DataFrame,
    sampling_frame: Any,
    *,
    block: Any = None,
    durations: Any = None,
    precision: float | None = None,
) -> Tuple[Any, Any]:
    """Lower a Spec or single Term to ``(EventModel, BaselineModel)``."""
    if isinstance(spec, Term):
        from .terms import as_spec

        spec = as_spec(spec)
    em = compile_events(
        spec,
        data,
        sampling_frame,
        block=block,
        durations=durations,
        precision=precision,
    )
    bm = compile_baseline(spec, sampling_frame)
    return em, bm
