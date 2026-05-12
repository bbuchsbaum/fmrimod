"""Normalization helpers for second-level requests."""

from __future__ import annotations

from dataclasses import replace

from .interfaces import GroupFitRequest


_METHOD_ALIASES = {
    "fixed": "fe",
    "random": "dl",
    "meta:fe": "fe",
    "meta:re": "dl",
    "fe": "fe",
    "dl": "dl",
    "pm": "pm",
    "reml": "reml",
}

_WEIGHT_ALIASES = {
    "ivw": "ivw",
    "1/var": "ivw",
    "inverse_variance": "ivw",
    "equal": "equal",
    "custom": "custom",
}

_CORRECTION_ALIASES = {
    "bh": "bh",
    "by": "by",
    "spatial": "spatial",
    "fdr:bh": "bh",
    "fdr:by": "by",
    "fdr:spatial": "spatial",
}

_BACKEND_ALIASES = {
    "auto": "auto",
    "python": "python",
    "fmrigds": "fmrigds",
    "r": "fmrigds",
}


def _normalize_method(request: GroupFitRequest) -> str:
    if request.method is not None:
        key = str(request.method).strip().lower()
        out = _METHOD_ALIASES.get(key)
        if out is None:
            raise ValueError(
                "method must be one of: fe, dl, pm, reml (or aliases fixed/random/meta:fe/meta:re)"
            )
        return out

    if request.effects == "fixed":
        return "fe"
    return request.tau2


def normalize_group_fit_request(request: GroupFitRequest) -> GroupFitRequest:
    """Normalize aliases and derive canonical method defaults."""
    weights_key = str(request.weights).strip().lower()
    weights = _WEIGHT_ALIASES.get(weights_key)
    if weights is None:
        raise ValueError("weights must be one of: ivw, equal, custom (or alias 1/var)")

    corr = request.correction
    if corr is not None:
        corr_key = str(corr).strip().lower()
        corr_out = _CORRECTION_ALIASES.get(corr_key)
        if corr_out is None:
            raise ValueError("correction must be one of: bh, by, spatial")
        corr = corr_out

    backend_key = str(request.backend).strip().lower()
    backend = _BACKEND_ALIASES.get(backend_key)
    if backend is None:
        raise ValueError("backend must be one of: auto, python, fmrigds")

    method = _normalize_method(request)

    return replace(
        request,
        method=method,
        weights=weights,
        correction=corr,
        backend=backend,
    )
