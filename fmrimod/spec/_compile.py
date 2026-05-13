"""Lower a typed :class:`Spec` to ``(EventModel, BaselineModel)`` artefacts.

The string formula path (``event_model("hrf(trial_type)")``) and the typed
:class:`Spec` path converge here.  This module renders a Spec back into the
string DSL the existing parser understands, which keeps a single source of
truth for design-matrix construction while letting users write typed terms at
the public API surface.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Tuple, cast

import numpy as np
import pandas as pd

from ..hrf.core import HRF
from ..hrf.registry import get_hrf
from .terms import Confounds, Drift, HrfTerm, Intercept, Spec, Term

BaselineBasis = Literal["constant", "poly", "bs", "ns"]
BaselineIntercept = Literal["runwise", "global", "none"]


def _hrf_label(hrf_obj: HRF | str) -> str:
    """Return a string label suitable for the formula parser's ``hrf(..., basis=...)``."""
    if isinstance(hrf_obj, str):
        return hrf_obj
    # Map known objects back to short registry labels.
    name = getattr(hrf_obj, "name", "")
    # Most pre-built objects retain their original names; the formula parser
    # accepts short labels like "spm", "spmg1", "gamma", etc.
    short = {
        "SPMG1_HRF": "spmg1",
        "SPMG2_HRF": "spmg2",
        "SPMG3_HRF": "spmg3",
        "GAMMA_HRF": "gamma",
        "GAUSSIAN_HRF": "gaussian",
        "BSPLINE_HRF": "bspline",
        "FIR_HRF": "fir",
    }
    return short.get(name, name or "spm")


def _resolve_normalized_hrf_label(term: HrfTerm) -> str:
    """If ``term.norm`` is set, register a normalized HRF and return its label.

    Otherwise return ``_hrf_label(term.hrf)``.
    """
    if term.norm is None:
        return _hrf_label(term.hrf)

    from ..hrf.normalization import normalize
    from ..hrf.registry import _HRF_REGISTRY, register_hrf

    base = term.hrf
    if isinstance(base, str):
        base = get_hrf(base)
    normalized = normalize(base, term.norm)
    label = f"__norm_{term.norm}_{id(base)}__"
    if label.lower() not in _HRF_REGISTRY:
        register_hrf(label, normalized)
    return label


def _format_subset(subset: Any) -> str | None:
    """Render a subset predicate as a string the formula parser can carry."""
    if subset is None:
        return None
    if isinstance(subset, str):
        return subset
    if isinstance(subset, Mapping):
        parts = [f"{k} == {v!r}" if isinstance(v, str) else f"{k} == {v}"
                 for k, v in subset.items()]
        return " & ".join(parts)
    if callable(subset):
        # Callable predicates are not representable in the string DSL; flag for
        # the compile pass so it can apply them after term realisation.
        return None
    raise TypeError(f"Unsupported subset predicate type: {type(subset).__name__}")


def _hrf_term_to_formula(term: HrfTerm) -> str:
    """Render one :class:`HrfTerm` as an ``hrf(...)`` formula clause."""
    if not term.variables:
        raise ValueError("HrfTerm has no variables to render")
    var_part = ", ".join(term.variables)
    extras = [f"basis={_resolve_normalized_hrf_label(term)!r}"]
    subset_str = _format_subset(term.subset)
    if subset_str is not None:
        extras.append(f"subset={subset_str!r}")
    if term.id is not None:
        extras.append(f"id={term.id!r}")
    if term.prefix is not None:
        extras.append(f"prefix={term.prefix!r}")
    if term.lag:
        extras.append(f"lag={term.lag}")
    return f"hrf({var_part}, {', '.join(extras)})"


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

    formula_parts: list[str] = []
    for term in event_terms:
        if not isinstance(term, HrfTerm):
            raise TypeError(
                f"Spec.events may only contain HrfTerm objects; got "
                f"{type(term).__name__}"
            )
        formula_parts.append(_hrf_term_to_formula(term))

    formula = " + ".join(formula_parts)

    kwargs: dict[str, Any] = dict(
        data=data,
        sampling_frame=sampling_frame,
        block=block,
        durations=durations,
    )
    if precision is not None:
        kwargs["precision"] = precision

    em = build_event_model(formula, **kwargs)

    # Apply callable subset filters that couldn't be represented in the DSL.
    # (No-op when none are callable; reserved for follow-up enhancement.)
    return em


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
