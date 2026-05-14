"""Payload-equality invariant for :class:`ContrastIntent`.

Source: ``beat-nilearn-10/post-01KRK82AEF16DAPYV56YWRFSJF`` item 13. The
board converged on payload equality (not Python object identity) as the
invariant for typed contrast survival across the first-level -> group
seam. Until ``ContrastIntent`` carries the four extra payload fields
(``basis_label``, ``weights``, ``design_id``, ``provenance_id``), the
"intent survives the seam" claim is satisfied on an empty subset of the
agreed payload. These tests pin the schema and the round-trip.
"""

from __future__ import annotations

import json

from fmrimod.glm.contrasts import ContrastIntent


def _full_intent() -> ContrastIntent:
    return ContrastIntent(
        kind="omnibus",
        name="task_effect",
        term="trial_type",
        levels=("word", "pseudoword"),
        rows=2,
        basis_label="hrf:canonical",
        weights=((1.0, -1.0, 0.0), (0.0, 1.0, -1.0)),
        design_id="design:sha256:abc123",
        provenance_id="fitprov:sha256:def456",
    )


def test_contrast_intent_carries_payload_equality_fields() -> None:
    intent = _full_intent()

    assert intent.basis_label == "hrf:canonical"
    assert intent.weights == ((1.0, -1.0, 0.0), (0.0, 1.0, -1.0))
    assert intent.design_id == "design:sha256:abc123"
    assert intent.provenance_id == "fitprov:sha256:def456"


def test_contrast_intent_to_dict_round_trip() -> None:
    intent = _full_intent()
    payload = intent.to_dict()

    for key in ("basis_label", "weights", "design_id", "provenance_id"):
        assert key in payload, f"missing key in to_dict(): {key}"

    rehydrated = ContrastIntent.from_dict(payload)
    assert rehydrated == intent
    assert rehydrated.to_dict() == payload

    json_payload = json.dumps(payload, sort_keys=True)
    via_json = ContrastIntent.from_dict(json.loads(json_payload))
    assert via_json == intent


def test_contrast_intent_payload_equality_independent_of_object_identity() -> None:
    intent_a = _full_intent()
    intent_b = ContrastIntent.from_dict(intent_a.to_dict())

    assert intent_a is not intent_b
    assert intent_a == intent_b
    assert intent_a.to_dict() == intent_b.to_dict()


def test_contrast_intent_defaults_keep_existing_callers_working() -> None:
    """Existing call sites construct ContrastIntent with only legacy fields;
    the new fields must default in such a way that those sites continue to
    work and their resulting payloads round-trip.
    """

    legacy = ContrastIntent(
        kind="omnibus",
        term="trial_type",
        levels=("a", "b"),
        rows=2,
    )

    payload = legacy.to_dict()
    rehydrated = ContrastIntent.from_dict(payload)
    assert rehydrated == legacy


def test_contrast_intent_weights_accept_one_dimensional_t_contrast() -> None:
    """t-contrasts are a single-row F-matrix. The schema must accept that
    shape uniformly via a tuple-of-tuples weights payload so the same
    invariant covers both t- and F-contrasts at the seam.
    """

    t_intent = ContrastIntent(
        kind="contrast_spec",
        name="slope",
        weights=((1.0, 0.0, -1.0),),
        provenance_id="fitprov:sha256:xyz",
    )

    payload = t_intent.to_dict()
    rehydrated = ContrastIntent.from_dict(payload)
    assert rehydrated == t_intent
    assert rehydrated.weights == ((1.0, 0.0, -1.0),)
