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


def _cosine_degree_from_cutoff(drift_term: Drift, sampling_frame: Any) -> int:
    """Compute the SPM DCT cosine-basis count from a high-pass cutoff.

    Convention (matching SPM and Nilearn's ``create_cosine_drift``):
    ``high_pass_hz = 1 / cutoff_seconds`` and
    ``n_basis = floor(2 * T * high_pass_hz)`` where ``T`` is the
    block duration in seconds. When run lengths differ across blocks
    we use the maximum so every per-block stripe carries the same
    number of drift columns; shorter blocks then carry the same
    DCT-II frequencies sampled over their own length.
    """
    if drift_term.cutoff is None:
        raise ValueError(
            "drift(basis='cosine') requires cutoff= (high-pass period "
            "in seconds, e.g. cutoff=128.0); none was supplied."
        )
    cutoff = float(drift_term.cutoff)
    if cutoff <= 0:
        raise ValueError(
            f"drift(basis='cosine'): cutoff must be positive, got {cutoff!r}"
        )
    blocklens_attr = getattr(sampling_frame, "blocklens", None)
    blocklens = (
        list(np.asarray(blocklens_attr).ravel())
        if blocklens_attr is not None
        else []
    )
    if not blocklens:
        raise ValueError(
            "drift(basis='cosine'): sampling frame has no block lengths"
        )
    tr_attr = getattr(sampling_frame, "tr", None)
    if tr_attr is None:
        tr_attr = getattr(sampling_frame, "TR", None)
    if tr_attr is None:
        raise ValueError(
            "drift(basis='cosine'): sampling frame has no TR; cannot "
            "translate cutoff (seconds) to a DCT basis count."
        )
    # SamplingFrame may carry per-run TRs as a list/array.
    tr_arr = np.atleast_1d(np.asarray(tr_attr, dtype=np.float64))
    tr_max = float(np.max(tr_arr))
    block_len_max = int(max(blocklens))
    T = float(block_len_max) * tr_max
    high_pass_hz = 1.0 / cutoff
    n_basis = int(np.floor(2.0 * T * high_pass_hz))
    if n_basis < 1:
        raise ValueError(
            f"drift(basis='cosine', cutoff={cutoff!r}): high-pass period "
            f"is longer than the run duration ({T:.1f} s); no cosine "
            f"basis functions would be generated. Reduce cutoff or use "
            f"a different drift basis."
        )
    return n_basis


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
    if drift_term is not None and drift_term.basis == "cosine":
        degree = _cosine_degree_from_cutoff(drift_term, sampling_frame)
    elif drift_term is not None and drift_term.cutoff is not None:
        import warnings as _warnings
        _warnings.warn(
            f"drift(basis={drift_term.basis!r}, cutoff={drift_term.cutoff!r}): "
            "cutoff= is only meaningful with basis='cosine'; ignoring.",
            UserWarning,
            stacklevel=3,
        )
    intercept_kind: str = intercept_term.per if intercept_term else "runwise"
    # Map our "run" convention to the legacy "runwise" label.
    if intercept_kind == "run":
        intercept_kind = "runwise"
    intercept = cast(BaselineIntercept, intercept_kind)

    nuisance_list = None
    if confounds:
        # Pull confound matrices per run from each Confounds.source. Keep the
        # column names as a DataFrame so downstream rank/aliasing diagnostics
        # surface the user-visible labels ("motion_x") instead of the
        # auto-numbered "V1"/"V2"/"V3" placeholders.
        per_run = []
        for c in confounds:
            if c.source is None:
                raise NotImplementedError(
                    "Confounds without an explicit `source` DataFrame are not yet "
                    "supported; supply source= or pre-join into the event table."
                )
            per_run.append(
                c.source[list(c.columns)].astype(np.float64).copy()
            )
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
