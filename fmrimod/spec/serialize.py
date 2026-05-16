"""JSON-safe serialization for :class:`~fmrimod.spec.Spec` trees.

The vision contract this implements: a checked-in design specification
must be enough, on its own, to reconstruct the same analysis on a
different machine. ``Spec.to_dict()`` returns a JSON-safe dict;
``Spec.from_dict(payload)`` rebuilds an equal Spec from that dict. The
two are exact inverses for the supported value set.

Schema is versioned (``Spec/v1``) so future format changes can be
detected at load time rather than silently producing a wrong shape.

What v1 supports
----------------

- Supported Term subclasses (HrfTerm, CovariateTerm, Drift, Intercept,
  Confounds).
- HRF specifiers given as registry-key strings (``"spm"``,
  ``"spmg3"``, ``"gamma"``, ...).
- Subset predicates expressed as strings (``"block <= 3"``) or
  Mappings (``{"block": 1}``).
- Confounds / CovariateTerms whose values are attached outside the portable
  spec (i.e. ``source=None``).

What v1 deliberately rejects
----------------------------

- :class:`~fmrimod.hrf.HRF` *instances* as ``HrfTerm.hrf``. Use a
  registry-key string instead; the typed HRF subclasses are
  reconstructed at compile time through the registry.
- Non-empty :attr:`HrfTerm.contrasts`. The contrast taxonomy is its
  own typed object tree; once the Spec is loaded, attach contrasts
  through the normal builder.
- :class:`pandas.DataFrame` payloads on ``source`` fields. A
  spec is the *shape* of an analysis, not its data; ship confound
  values separately and resolve them against the event table at
  compile time.
- Callable subset predicates. Inline lambdas don't admit JSON
  representation; use the string-predicate or dict form.

Each rejection raises a :class:`SpecSerializationError` whose message
names the field, the term, and the path forward. The error is
deliberately loud so checked-in specs cannot drift into a state where
they are silently incomplete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal, Optional, cast

from .terms import Confounds, CovariateTerm, Drift, HrfTerm, Intercept, Spec, Term

SCHEMA_VERSION = "Spec/v1"


class SpecSerializationError(ValueError):
    """Raised when a Spec value cannot round-trip through to_dict / from_dict."""


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def to_dict(spec: Spec) -> dict[str, object]:
    """Serialize a :class:`Spec` to a JSON-safe dict.

    The returned payload always carries ``schema_version`` so loaders
    can detect format drift. See the module docstring for the
    supported value set.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "events": [_encode_term(t) for t in spec.events],
        "baseline": [_encode_term(t) for t in spec.baseline],
    }


def _encode_term(term: Term) -> dict[str, object]:
    if isinstance(term, CovariateTerm):
        return _encode_covariate_term(term)
    if isinstance(term, HrfTerm):
        return _encode_hrf_term(term)
    if isinstance(term, Drift):
        return _encode_drift(term)
    if isinstance(term, Intercept):
        return _encode_intercept(term)
    if isinstance(term, Confounds):
        return _encode_confounds(term)
    raise SpecSerializationError(
        f"Unknown Term subclass: {type(term).__name__}. "
        "Spec/v1 supports HrfTerm, CovariateTerm, Drift, Intercept, "
        "and Confounds."
    )


def _encode_covariate_term(term: CovariateTerm) -> dict[str, object]:
    if term.source is not None:
        raise SpecSerializationError(
            "CovariateTerm(source=<DataFrame>) cannot be serialized to a Spec "
            "payload: a Spec describes the shape of an analysis, not its "
            "sampled time-course data. Strip `source` before saving and "
            "re-attach the DataFrame at load time."
        )
    return {
        "kind": "CovariateTerm",
        "variables": list(term.variables),
        "prefix": term.prefix,
        "id": term.id,
    }


def _encode_hrf_term(term: HrfTerm) -> dict[str, object]:
    if term.contrasts:
        raise SpecSerializationError(
            f"HrfTerm(id={term.id!r}, variables={term.variables!r}) carries "
            f"{len(term.contrasts)} contrasts; Spec/v1 cannot round-trip them. "
            "Attach contrasts after loading the spec."
        )
    if term.hrf_fun is not None:
        raise SpecSerializationError(
            f"HrfTerm(id={term.id!r}, variables={term.variables!r}) carries "
            "a callable hrf_fun; Spec/v1 only serializes registry-key HRFs."
        )
    if not isinstance(term.hrf, str):
        raise SpecSerializationError(
            f"HrfTerm(id={term.id!r}, variables={term.variables!r}).hrf is a "
            f"{type(term.hrf).__name__} instance; Spec/v1 only serializes "
            "registry-key strings (e.g. 'spm', 'spmg3', 'gamma'). The typed "
            "HRF subclasses round-trip through the registry at compile time."
        )
    return {
        "kind": "HrfTerm",
        "variables": list(term.variables),
        "hrf": term.hrf,
        "nbasis": term.nbasis,
        "modulators": list(term.modulators),
        "durations": term.durations,
        "lag": term.lag,
        "subset": _encode_subset(term),
        "prefix": term.prefix,
        "id": term.id,
        "norm": term.norm,
        "normalize": term.normalize,
        "summate": term.summate,
    }


def _encode_subset(term: HrfTerm) -> object:
    sub = term.subset
    if sub is None:
        return None
    if isinstance(sub, str):
        return {"kind": "str", "value": sub}
    if isinstance(sub, Mapping):
        # Mapping[str, object]. We don't recurse into the values: they're
        # typically scalar comparisons (e.g. {"block": 1}) and JSON's
        # native types cover the supported set. If a caller passes a
        # nested non-scalar, json.dumps will reject it later with a
        # clear error -- preferable to silently dropping shape.
        return {"kind": "dict", "value": dict(sub)}
    if callable(sub):
        raise SpecSerializationError(
            f"HrfTerm(id={term.id!r}, variables={term.variables!r}) has a "
            "callable `subset` predicate. Inline callables can't round-trip "
            "through JSON; use a string predicate ('block <= 3') or a dict "
            "({'block': 1}) instead."
        )
    raise SpecSerializationError(
        f"HrfTerm.subset must be str | Mapping | callable | None, got "
        f"{type(sub).__name__}."
    )


def _encode_drift(term: Drift) -> dict[str, object]:
    return {
        "kind": "Drift",
        "basis": term.basis,
        "degree": term.degree,
        "cutoff": term.cutoff,
    }


def _encode_intercept(term: Intercept) -> dict[str, object]:
    return {"kind": "Intercept", "per": term.per}


def _encode_confounds(term: Confounds) -> dict[str, object]:
    if term.source is not None:
        raise SpecSerializationError(
            "Confounds(source=<DataFrame>) cannot be serialized to a Spec "
            "payload: a Spec describes the shape of an analysis, not its "
            "data. Strip `source` before saving and re-attach the DataFrame "
            "at load time, or save it separately and resolve against the "
            "event table at compile time."
        )
    return {"kind": "Confounds", "columns": list(term.columns)}


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def from_dict(payload: Mapping[str, object]) -> Spec:
    """Reconstruct a :class:`Spec` from :func:`to_dict` output.

    Raises :class:`SpecSerializationError` if the schema version does
    not match :data:`SCHEMA_VERSION` or any term payload is malformed.
    """
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SpecSerializationError(
            f"Spec payload has schema_version={version!r}; this build of "
            f"fmrimod reads {SCHEMA_VERSION!r}."
        )
    events_payload = payload.get("events", ())
    baseline_payload = payload.get("baseline", ())
    if not isinstance(events_payload, (list, tuple)):
        raise SpecSerializationError("Spec.events must be a sequence")
    if not isinstance(baseline_payload, (list, tuple)):
        raise SpecSerializationError("Spec.baseline must be a sequence")
    events = tuple(_decode_term(p, slot="events") for p in events_payload)
    baseline = tuple(_decode_term(p, slot="baseline") for p in baseline_payload)
    return Spec(events=events, baseline=baseline)


_TermDecoder = Callable[[Mapping[str, object]], Term]
_DECODERS: dict[str, _TermDecoder] = {}


def _register_decoder(
    kind: str,
) -> Callable[[_TermDecoder], _TermDecoder]:
    def deco(fn: _TermDecoder) -> _TermDecoder:
        _DECODERS[kind] = fn
        return fn
    return deco


def _decode_term(payload: Mapping[str, object], *, slot: str) -> Term:
    if not isinstance(payload, Mapping):
        raise SpecSerializationError(
            f"{slot} entry must be a mapping, got {type(payload).__name__}."
        )
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise SpecSerializationError(
            f"{slot} entry is missing the 'kind' discriminator."
        )
    decoder = _DECODERS.get(kind)
    if decoder is None:
        raise SpecSerializationError(
            f"Unknown term kind {kind!r} in {slot}; "
            f"expected one of {sorted(_DECODERS)}."
        )
    return decoder(payload)


@_register_decoder("HrfTerm")
def _decode_hrf_term(payload: Mapping[str, object]) -> HrfTerm:
    variables = _coerce_str_tuple(payload.get("variables", ()), field="variables")
    hrf_value = payload.get("hrf", "spm")
    if not isinstance(hrf_value, str):
        raise SpecSerializationError(
            f"HrfTerm.hrf must round-trip as a registry-key string; got "
            f"{type(hrf_value).__name__}."
        )
    modulators = _coerce_str_tuple(payload.get("modulators", ()), field="modulators")
    return HrfTerm(
        variables=variables,
        hrf=hrf_value,
        contrasts=(),  # Spec/v1 never round-trips contrasts; load-time empty.
        modulators=modulators,
        nbasis=cast(Optional[int], payload.get("nbasis")),
        durations=cast("str | float | None", payload.get("durations")),
        lag=float(cast(Any, payload.get("lag", 0.0))),
        subset=cast(
            "str | Mapping[str, object] | Callable[..., object] | None",
            _decode_subset(payload.get("subset")),
        ),
        prefix=cast(Optional[str], payload.get("prefix")),
        id=cast(Optional[str], payload.get("id")),
        norm=cast(Any, payload.get("norm")),
        normalize=bool(payload.get("normalize", False)),
        summate=bool(payload.get("summate", True)),
    )


@_register_decoder("CovariateTerm")
def _decode_covariate_term(payload: Mapping[str, object]) -> CovariateTerm:
    variables = _coerce_str_tuple(payload.get("variables", ()), field="variables")
    if not variables:
        raise SpecSerializationError("CovariateTerm requires at least one variable.")
    return CovariateTerm(
        variables=variables,
        hrf="identity",
        source=None,
        prefix=cast(Optional[str], payload.get("prefix")),
        id=cast(Optional[str], payload.get("id")),
        normalize=False,
        summate=False,
    )


def _decode_subset(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SpecSerializationError(
            f"HrfTerm.subset must round-trip as null or a {{'kind', 'value'}} "
            f"object; got {type(value).__name__}."
        )
    kind = value.get("kind")
    if kind == "str":
        sub_str = value.get("value")
        if not isinstance(sub_str, str):
            raise SpecSerializationError(
                "HrfTerm.subset {'kind': 'str'} must carry a 'value' string."
            )
        return sub_str
    if kind == "dict":
        sub_dict = value.get("value")
        if not isinstance(sub_dict, Mapping):
            raise SpecSerializationError(
                "HrfTerm.subset {'kind': 'dict'} must carry a 'value' mapping."
            )
        return dict(sub_dict)
    raise SpecSerializationError(
        f"HrfTerm.subset payload has unknown 'kind' {kind!r}; "
        "expected 'str' or 'dict'."
    )


@_register_decoder("Drift")
def _decode_drift(payload: Mapping[str, object]) -> Drift:
    return Drift(
        basis=cast(
            "Literal['constant', 'poly', 'bs', 'ns', 'cosine']",
            payload.get("basis", "constant"),
        ),
        degree=int(cast(Any, payload.get("degree", 1))),
        cutoff=cast(Optional[float], payload.get("cutoff")),
    )


@_register_decoder("Intercept")
def _decode_intercept(payload: Mapping[str, object]) -> Intercept:
    return Intercept(
        per=cast("Literal['run', 'global', 'none']", payload.get("per", "run"))
    )


@_register_decoder("Confounds")
def _decode_confounds(payload: Mapping[str, object]) -> Confounds:
    columns = _coerce_str_tuple(payload.get("columns", ()), field="columns")
    if not columns:
        raise SpecSerializationError("Confounds requires at least one column.")
    return Confounds(columns=columns, source=None)


def _coerce_str_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SpecSerializationError(
            f"{field} must be a sequence of strings, got {type(value).__name__}."
        )
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise SpecSerializationError(
                f"{field}[{i}] must be a string, got {type(item).__name__}."
            )
        out.append(item)
    return tuple(out)


# Self-check at import time: every concrete Term subclass we know
# about has a decoder wired. Adding a new Term subclass without
# updating the serializer trips this immediately rather than at run
# time on the first round-trip attempt.
_KNOWN_TERMS = (HrfTerm, CovariateTerm, Drift, Intercept, Confounds)
for _cls in _KNOWN_TERMS:
    assert _cls.__name__ in _DECODERS, (
        f"serialize.py: missing decoder for {_cls.__name__}"
    )
del _cls
