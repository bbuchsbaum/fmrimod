"""JSON-safe Spec serialization round-trip.

The reproducibility contract under test, from VISION.md::

    Two people on two machines should be able to reconstruct the
    same result from a checked-in design specification alone.

That clause has two halves: the spec round-trips byte-for-byte
through JSON, and the rebuilt spec is structurally equal to the
original (so any downstream compile / fit step produces the same
artifacts). This module covers the first half end-to-end and the
second half through structural equality plus the `spec_diff`
helper -- which is the canonical "are these the same analysis"
operator on the typed Spec tree.

Negative-path coverage is intentionally explicit. Spec/v1 deliberately
refuses to silently drop information when a value can't round-trip
(inline callables, DataFrame confound payloads, HRF instances rather
than registry strings, attached ContrastSpec lists). Each rejection
must raise :class:`SpecSerializationError` with a message that names
the offending field and points the user at the supported alternative.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fmrimod.hrf.library import GammaHRF
from fmrimod.spec import (
    SCHEMA_VERSION,
    Confounds,
    Drift,
    HrfTerm,
    Intercept,
    Spec,
    SpecSerializationError,
    confounds,
    drift,
    hrf,
    intercept,
    spec_diff,
)


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------


def _round_trip(spec: Spec) -> Spec:
    """Round-trip via dict and via JSON text -- both must reproduce."""
    via_dict = Spec.from_dict(spec.to_dict())
    via_text = Spec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert via_dict == via_text, (
        "JSON-text round-trip diverged from dict round-trip; the payload "
        "must already be JSON-safe."
    )
    return via_dict


def test_empty_spec_round_trips():
    s = Spec()
    rebuilt = _round_trip(s)
    assert rebuilt == s
    assert spec_diff(s, rebuilt).is_empty


def test_minimal_hrf_round_trips():
    s = hrf("trial_type")
    # `hrf(...)` returns a bare term; wrap to ensure we go through Spec.
    spec = Spec() + s
    rebuilt = _round_trip(spec)
    assert spec_diff(spec, rebuilt).is_empty


def test_full_event_and_baseline_spec_round_trips():
    spec = (
        hrf("trial_type", basis="spm", durations=0.5, lag=0.5, summate=False)
        + hrf("block", id="block_term", normalize=True, norm="unit_peak")
        + drift("cosine", cutoff=128)
        + intercept(per="run")
        + confounds("trans_x", "trans_y", "rot_z")
    )
    rebuilt = _round_trip(spec)
    assert spec_diff(spec, rebuilt).is_empty
    # Spot-check the rehydrated values reach their typed fields.
    assert rebuilt.events[0].lag == 0.5
    assert rebuilt.events[0].summate is False
    assert rebuilt.events[1].id == "block_term"
    assert rebuilt.events[1].norm == "unit_peak"
    assert rebuilt.events[1].normalize is True
    assert rebuilt.baseline[0] == Drift(basis="cosine", cutoff=128)
    assert rebuilt.baseline[1] == Intercept(per="run")
    assert rebuilt.baseline[2] == Confounds(columns=("trans_x", "trans_y", "rot_z"))


def test_subset_string_predicate_round_trips():
    spec = Spec() + hrf("trial_type", subset="block <= 3")
    rebuilt = _round_trip(spec)
    assert rebuilt.events[0].subset == "block <= 3"


def test_subset_dict_predicate_round_trips():
    spec = Spec() + hrf("trial_type", subset={"task": "memory", "block": 1})
    rebuilt = _round_trip(spec)
    assert rebuilt.events[0].subset == {"task": "memory", "block": 1}


def test_payload_carries_schema_version():
    payload = Spec().to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Vision-aligned reproducibility golden
# ---------------------------------------------------------------------------


def test_reproducibility_golden_design_round_trips_to_identical_spec():
    """VISION.md:99-103 -- reconstruct the same analysis from a
    checked-in design specification alone.

    This test simulates the cross-machine workflow: the original
    researcher writes the design in Python, serialises it to a
    JSON string (which is what would be committed to git), and a
    second researcher loads the JSON string on a different machine.
    The reloaded Spec must be structurally indistinguishable from
    the original -- not just `==`, but with an empty `spec_diff`,
    so any downstream compile step would emit the same design
    matrix and any downstream fit would emit the same betas.
    """
    original = (
        hrf("trial_type", basis="spmg3", durations="duration", lag=0.0)
        + hrf("block", subset={"task": "memory"}, prefix="memblk")
        + drift("cosine", cutoff=128)
        + intercept(per="run")
        + confounds("trans_x", "trans_y", "rot_z", "framewise_displacement")
    )

    # Researcher 1: serialise.
    on_disk = json.dumps(original.to_dict(), sort_keys=True, indent=2)

    # Researcher 2 (different process / machine -- simulated by a
    # fresh JSON parse): rebuild.
    rebuilt = Spec.from_dict(json.loads(on_disk))

    # The rebuilt Spec must be byte-for-byte the same shape.
    assert rebuilt == original
    diff = spec_diff(original, rebuilt)
    assert diff.is_empty, diff.summary()

    # And the serialisation must be deterministic so the on-disk
    # artifact compares cleanly in version control.
    again = json.dumps(rebuilt.to_dict(), sort_keys=True, indent=2)
    assert again == on_disk


# ---------------------------------------------------------------------------
# Negative path: shapes that must NOT silently round-trip
# ---------------------------------------------------------------------------


def test_hrf_instance_is_rejected_with_helpful_message():
    """An HRF instance carries dataclass-private params that Spec/v1
    can't generically rebuild; users must use a registry-key string."""
    spec = Spec() + hrf("trial_type", basis=GammaHRF(name="custom"))
    with pytest.raises(SpecSerializationError, match="registry-key strings"):
        spec.to_dict()


def test_callable_subset_is_rejected():
    spec = Spec() + hrf("trial_type", subset=lambda df: df["block"] == 1)
    with pytest.raises(SpecSerializationError, match="callable .subset."):
        spec.to_dict()


def test_non_empty_contrasts_are_rejected():
    """A loaded Spec rebuilds with `contrasts=()`. To avoid silently
    dropping user-attached contrasts, serialization refuses
    non-empty contrast tuples outright."""
    # Inject a sentinel object into the contrasts slot (the type
    # annotation is Tuple[ContrastSpec, ...] but the dataclass is
    # frozen at the value level, not the type-checker level).
    term = HrfTerm(variables=("trial_type",), contrasts=("__sentinel__",))
    spec = Spec(events=(term,))
    with pytest.raises(SpecSerializationError, match="contrasts"):
        spec.to_dict()


def test_confounds_with_inline_dataframe_is_rejected():
    df = pd.DataFrame({"trans_x": [0.1, 0.2]})
    spec = Spec() + confounds("trans_x", source=df)
    with pytest.raises(SpecSerializationError, match="Confounds"):
        spec.to_dict()


def test_schema_version_mismatch_is_rejected():
    bad = {
        "schema_version": "Spec/v0",
        "events": [],
        "baseline": [],
    }
    with pytest.raises(SpecSerializationError, match="schema_version"):
        Spec.from_dict(bad)


def test_unknown_term_kind_is_rejected():
    bad = {
        "schema_version": SCHEMA_VERSION,
        "events": [{"kind": "DefinitelyNotATerm", "variables": ["x"]}],
        "baseline": [],
    }
    with pytest.raises(SpecSerializationError, match="Unknown term kind"):
        Spec.from_dict(bad)


def test_missing_kind_discriminator_is_rejected():
    bad = {
        "schema_version": SCHEMA_VERSION,
        "events": [{"variables": ["x"]}],
        "baseline": [],
    }
    with pytest.raises(SpecSerializationError, match="kind"):
        Spec.from_dict(bad)


def test_subset_malformed_kind_is_rejected():
    bad = {
        "schema_version": SCHEMA_VERSION,
        "events": [
            {
                "kind": "HrfTerm",
                "variables": ["x"],
                "subset": {"kind": "regex", "value": ".*"},
            }
        ],
        "baseline": [],
    }
    with pytest.raises(SpecSerializationError, match="unknown 'kind'"):
        Spec.from_dict(bad)


def test_variables_must_be_strings():
    bad = {
        "schema_version": SCHEMA_VERSION,
        "events": [{"kind": "HrfTerm", "variables": [42]}],
        "baseline": [],
    }
    with pytest.raises(SpecSerializationError, match="variables"):
        Spec.from_dict(bad)


# ---------------------------------------------------------------------------
# JSON-safety crosscheck
# ---------------------------------------------------------------------------


def test_to_dict_output_is_json_serialisable_without_default_hook():
    """The payload must round-trip through ``json.dumps`` *without* a
    `default=` hook, i.e. without falling back to ``repr``. This catches
    accidental tuples, numpy scalars, or other non-JSON-native values
    leaking into the encoder."""
    spec = (
        hrf("trial_type", durations=np.float64(0.5))  # numpy scalar
        + drift("poly", degree=3)
        + intercept()
        + confounds("trans_x")
    )
    text = json.dumps(spec.to_dict())
    payload = json.loads(text)
    rebuilt = Spec.from_dict(payload)
    # Spec must compare equal after the round-trip, with the
    # numpy scalar coerced to a Python float during JSON-encoding.
    assert spec_diff(spec, rebuilt).is_empty
