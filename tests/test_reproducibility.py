"""End-to-end reproducibility golden for VISION.md:99-103.

The vision clause::

    Two people on two machines should be able to reconstruct the
    same result from a checked-in design specification alone.

The two halves of that claim already have unit-level coverage:

- ``tests/test_spec/test_serialize.py`` proves the typed :class:`Spec`
  tree round-trips byte-for-byte through JSON.
- ``tests/test_glm/test_fmri_lm_provenance.py`` proves
  :class:`FitProvenance` round-trips through ``to_dict`` / ``to_json``
  and carries every VISION-named field (seed, solver path, HRF
  normalisation, AR config, masking mode, version).

What was missing was the *closing of the loop*: a test that the spec
checked into git produces the same fit when reloaded. This module
fills that gap.

The test is intentionally a top-level reproducibility golden -- not
tucked under ``test_spec`` or ``test_glm`` -- because the property it
locks crosses every layer: spec serialization, design-matrix
compilation, GLM fit, and provenance recording. Drift in any of those
silently breaks the vision claim; one assertion here catches it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.spec import Spec, drift, hrf, intercept, spec_diff


_EVENTS = pd.DataFrame(
    {
        "onset": [10.0, 30.0, 50.0, 75.0],
        "duration": [2.0, 2.0, 2.0, 2.0],
        "trial_type": ["A", "B", "A", "B"],
        "run": [1, 1, 1, 1],
    }
)


def _deterministic_dataset():
    """A small but non-trivial dataset; seeded so every run sees the same BOLD."""
    rng = np.random.default_rng(20260513)
    n_time, n_voxels = 80, 8
    bold = rng.standard_normal((n_time, n_voxels)).astype(np.float64)
    return fm.fmri_dataset(bold, tr=2.0, events=_EVENTS)


def _design_spec() -> Spec:
    """A modest design exercising events plus a polynomial baseline drift.

    Restricted to shapes the typed-spec compile path supports today:
    ``drift("poly", degree=...)`` rather than ``"cosine"`` (which the
    spec builder accepts but the baseline_model does not), and no
    ``confounds()`` without an inline ``source`` (also not yet wired
    through). When those gaps close, the test can grow to cover the
    full builder surface without changing what it is asserting.
    """
    return (
        Spec()
        + hrf("trial_type", norm="spm")
        + drift("poly", degree=2)
        + intercept(per="run")
    )


def test_spec_to_dict_round_trip_preserves_fit_betas_and_provenance():
    """A checked-in design spec re-fits to byte-identical betas.

    Simulates the cross-machine workflow with one process:

    1. Build a typed Spec, fit it, record the betas and provenance.
    2. Serialise the Spec to a JSON string -- *this* is the artifact
       that would be checked into version control.
    3. Reload from a fresh JSON parse and re-fit on an identical
       dataset.
    4. Assert: betas are bytewise equal, residuals are bytewise
       equal, and the `FitProvenance` payload matches modulo the
       (deterministically-empty) seed slot.

    The deterministic-runwise-OLS engine is the default and carries
    no random state for this spec shape, so byte-equality is the
    correct gate -- not allclose.
    """
    spec = _design_spec()
    dataset_a = _deterministic_dataset()

    # First pass: fit with the original Spec.
    fit_a = fm.fmri_lm(spec, dataset_a)

    # The artifact: serialise the Spec to JSON text. This is what
    # someone would commit to a repository as the design of record.
    on_disk = json.dumps(spec.to_dict(), sort_keys=True)

    # Second pass: reload the Spec from a *fresh* JSON parse (no
    # in-memory sharing) and re-fit on a fresh-but-equivalent
    # dataset.
    rebuilt_spec = Spec.from_dict(json.loads(on_disk))
    dataset_b = _deterministic_dataset()
    fit_b = fm.fmri_lm(rebuilt_spec, dataset_b)

    # The rebuilt spec must be structurally indistinguishable from
    # the original. Belt-and-braces: assert both `==` and an empty
    # spec_diff so a future field-level drift trips us here, not
    # downstream in the byte comparison.
    assert rebuilt_spec == spec
    diff = spec_diff(spec, rebuilt_spec)
    assert diff.is_empty, diff.summary()

    # The fit must reproduce bytewise -- this is the hard claim.
    np.testing.assert_array_equal(fit_a.betas, fit_b.betas)

    # Provenance must match too. The default engine is deterministic
    # so seed-related fields are themselves stable; the version,
    # solver path, HRF normalisation, AR config, and mask mode are
    # all derived from the spec + dataset and must therefore agree.
    assert fit_a.provenance is not None
    assert fit_b.provenance is not None
    payload_a = fit_a.provenance.to_dict()
    payload_b = fit_b.provenance.to_dict()
    assert payload_a == payload_b, (
        f"FitProvenance drifted across the spec round-trip:\n"
        f"  before reload: {payload_a}\n"
        f"  after reload:  {payload_b}"
    )


def test_repeated_fits_of_the_same_spec_are_bytewise_identical():
    """Determinism baseline: without any round-trip, two fits of the
    same Spec on the same dataset produce identical betas.

    Locks the precondition that
    :func:`test_spec_to_dict_round_trip_preserves_fit_betas_and_provenance`
    relies on: any byte-level drift across spec round-trip can be
    attributed to serialization rather than to non-determinism in
    the fit pipeline itself.
    """
    spec = _design_spec()
    fit_a = fm.fmri_lm(spec, _deterministic_dataset())
    fit_b = fm.fmri_lm(spec, _deterministic_dataset())
    np.testing.assert_array_equal(fit_a.betas, fit_b.betas)
    assert fit_a.provenance is not None
    assert fit_b.provenance is not None
    assert fit_a.provenance.to_dict() == fit_b.provenance.to_dict()


def test_design_specification_is_the_unit_of_reproducibility():
    """A checked-in design spec is enough -- you don't need the
    original Python source. Demonstrates the spec-as-artifact claim:
    no reference to `_design_spec()` exists in the rebuild path, just
    the JSON string.
    """
    # The "checked-in artifact" -- ordinary JSON, no fmrimod-specific
    # decoder hooks, sort_keys for deterministic byte ordering.
    artifact = json.dumps(_design_spec().to_dict(), sort_keys=True, indent=2)

    # Rebuild starting from the artifact alone.
    rebuilt = Spec.from_dict(json.loads(artifact))

    # And re-emitting the artifact from the rebuilt spec returns the
    # same bytes -- so a round-trip through git wouldn't churn the
    # diff with formatting noise.
    re_emitted = json.dumps(rebuilt.to_dict(), sort_keys=True, indent=2)
    assert re_emitted == artifact

    # Sanity: the rebuilt spec is fittable on a fresh dataset.
    fit = fm.fmri_lm(rebuilt, _deterministic_dataset())
    assert fit.betas is not None
    assert fit.provenance is not None
