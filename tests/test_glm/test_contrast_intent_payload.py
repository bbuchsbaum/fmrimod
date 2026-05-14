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


# -- Phase 2a: production-site population ------------------------------------
#
# bd-01KRK7YGR5Y7E2CPV9T9R316T7 Phase 2a populates `weights` and
# `provenance_id` at every ``ContrastIntent`` construction site reachable from
# the public seam, including the explicit ``fit.contrast(spec)`` dispatch in
# :mod:`fmrimod.glm.fmri_lm` and the BIDS Stats Model contrast translator. The
# tests below assert the population is non-null and survives the to_dict()
# round-trip, plus that two fits with the same provenance produce the same
# ``provenance_id`` (the load-bearing equality invariant). ``basis_label`` and
# ``design_id`` remain deferred to a follow-up Phase 2b slice because their
# derivation requires reach-in to the fit's event-model basis state and the
# realized design matrix accessor respectively.


def _seam_fit():
    """Construct a small public-seam fit via the canonical four-stage entry.

    Kept inline to make the test self-contained against the public ``fm.*``
    surface; if any of these imports stop resolving, the smoke gate from the
    first-ten-minutes acceptance bead (bd-01KRK8X32P40YCAHERP5B4W68N) will
    also break, so the failure mode is shared on purpose.
    """
    import numpy as np
    import pandas as pd

    import fmrimod as fm
    from fmrimod.spec import hrf

    rng = np.random.default_rng(7)
    bold = rng.standard_normal((60, 6))
    events = pd.DataFrame(
        {
            "onset": [6.0, 18.0, 30.0, 42.0],
            "duration": [2.0, 2.0, 2.0, 2.0],
            "trial_type": ["a", "b", "a", "b"],
        }
    )
    dataset = fm.fmri_dataset(bold, tr=2.0, events=events)
    fit = fm.fmri_lm(hrf("trial_type", norm="spm"), dataset)
    return fit


def test_array_contrast_carries_weights_and_provenance_id_payload() -> None:
    """A raw-array contrast through ``fit.contrast(np.array(...))`` must
    populate the two production-derivable payload fields. This is the bottom
    of the dispatch table — every other contrast kind funnels through
    ``_compute_contrast`` and inherits the same invariant.
    """
    import numpy as np

    fit = _seam_fit()
    n_cols = len(fit.design_columns().names)
    weights = np.zeros(n_cols, dtype=np.float64)
    weights[0] = 1.0

    result = fit.contrast(weights, name="first_column")

    intent = result.intent
    assert intent.weights is not None, "weights payload must be non-null"
    assert intent.weights == ((1.0,) + (0.0,) * (n_cols - 1),)
    assert intent.provenance_id is not None, "provenance_id must be non-null"
    assert intent.provenance_id.startswith("fitprov:sha256:")


def test_omnibus_contrast_carries_weights_and_provenance_id_payload() -> None:
    """OmnibusContrast goes through the typed branch of ``fit.contrast``;
    payload-equality fields must be populated there too.
    """
    from fmrimod.contrast import OmnibusContrast

    fit = _seam_fit()
    omnibus = OmnibusContrast(term="trial_type", levels=("a", "b"))
    result = fit.contrast(omnibus)

    intent = result.intent
    assert intent.kind == "omnibus"
    assert intent.weights is not None
    # OmnibusContrast for a two-level factor resolves to a 1-row F-matrix.
    assert len(intent.weights) >= 1
    assert all(isinstance(row, tuple) for row in intent.weights)
    assert intent.provenance_id is not None
    assert intent.provenance_id.startswith("fitprov:sha256:")


def test_provenance_id_is_stable_for_equivalent_fits() -> None:
    """The same fit configuration must produce the same ``provenance_id``;
    that's the load-bearing payload-equality property the bead's red check
    relies on for first-level -> group transit.
    """
    fit_a = _seam_fit()
    fit_b = _seam_fit()
    # Hash the same provenance content, even though the Python objects differ.
    assert fit_a.provenance == fit_b.provenance

    from fmrimod.glm.contrasts import provenance_id

    assert provenance_id(fit_a) == provenance_id(fit_b)
    assert provenance_id(fit_a) is not None


def test_contrast_intent_payload_round_trips_after_seam_fit() -> None:
    """The full intent payload from a real fit must survive to_dict/from_dict
    so that group reducers can compare intents by value across serialization.
    """
    import numpy as np

    fit = _seam_fit()
    weights = np.zeros(len(fit.design_columns().names), dtype=np.float64)
    weights[0] = 1.0
    intent = fit.contrast(weights, name="round_trip").intent
    rehydrated = ContrastIntent.from_dict(intent.to_dict())
    assert rehydrated == intent


def test_provenance_id_returns_none_when_provenance_absent() -> None:
    """Defensive: the helper must tolerate fits without provenance (legacy
    test paths or partially-constructed objects) by returning ``None`` rather
    than raising — release-time gates can then refuse ``None`` for flagship
    rows while debug paths keep working.
    """
    from fmrimod.glm.contrasts import provenance_id

    class _Stub:
        pass

    assert provenance_id(_Stub()) is None
